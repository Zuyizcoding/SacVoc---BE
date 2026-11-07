"""Exercise the real FastAPI routes using an isolated, disposable catalog copy.

Intuition: direct ranker tests do not check routing, request validation or JSON
serialization. This check sends ASGI requests through the unchanged application,
without a listening server or network. Profiles are explicit test fixtures, not
Gemini predictions and not measured users.

Schema: the report contains scenario checks and one result per supported profile.
Pipeline: verify source hashes -> SQLite backup into a temporary directory ->
start a child with explicit DB/query settings -> execute routes -> remove copy ->
verify source hashes again. Import app.main ONLY inside that configured child.
"""

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from .encoder import file_sha256
from .integrity import verify_manifest
from .workflow import PROJECT_ROOT, write_json

DEVICE_ID = hashlib.sha256(b"isolated-f3-integration-scenario").hexdigest()
RECOMMEND_PATH = "/api/v1/outfits/recommend"


def run_isolated_api_verification(bundle: Path, output: Path) -> dict:
    """The parent never imports application settings or opens a writable source."""
    verify_manifest(bundle)
    source = bundle.resolve(strict=True)
    before = {
        name: file_sha256(source / name)
        for name in (
            "catalog.db",
            "query_vectors.json",
            "query_vectors.meta.json",
            "manifest.json",
        )
    }
    output.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix="sacvoc-api-") as temporary:
        copy = Path(temporary) / "catalog-copy.db"
        with closing(
            sqlite3.connect((source / "catalog.db").as_uri() + "?mode=ro", uri=True)
        ) as original:
            with closing(sqlite3.connect(copy)) as target:
                original.backup(target)
        environment = {
            **os.environ,
            "DATABASE_URL": "sqlite:///" + copy.as_posix(),
            "F3_QUERY_VECTORS_PATH": str(source / "query_vectors.json"),
            "GEMINI_API_KEY": "",
            "DEBUG": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "app.catalog_preparation.api_verification",
                "--database-copy",
                str(copy),
                "--report",
                str(output / "report.json"),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
        )
        if process.returncode:
            raise RuntimeError(f"Isolated API verification failed:\n{process.stderr}")
    if any(file_sha256(source / name) != digest for name, digest in before.items()):
        raise ValueError("API verification modified the source bundle")
    verify_manifest(source)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    report.update(
        source_unchanged=True,
        temporary_database_removed=not copy.exists(),
        source_sha256=before,
    )
    write_json(output / "report.json", report)
    return report


def exercise_routes(database_copy: Path) -> dict:
    """Worker-only entry: test real routes and real recommendation, with no AI."""
    from fastapi.testclient import TestClient

    from app.core.config import settings
    from app.db.database import SessionLocal, engine
    from app.db.models import AnalysisHistory, Outfit
    from app.main import app
    from app.services import gemini_client
    from app.services.recommendation import embedding_store
    from app.services.recommendation.query_builder import (
        BODY_KEYWORDS,
        OCCASION_KEYWORDS,
        SEASON_KEYWORDS,
        build_outfit_query,
    )

    expected_url = "sqlite:///" + database_copy.resolve().as_posix()
    if settings.DATABASE_URL != expected_url:
        raise ValueError(
            "Worker DB configuration does not point to the disposable copy"
        )

    def forbid_external_encoder(*args, **kwargs):
        raise AssertionError("This verification must use only saved query vectors")

    # Startup does not need an AI client; guard both fallback entry points.
    gemini_client.get_gemini_client = forbid_external_encoder
    gemini_client.GeminiClient.generate_json = forbid_external_encoder
    embedding_store._get_runtime_text_encoder = forbid_external_encoder
    logging.getLogger("app.services.outfit_service").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    headers = {"X-Device-ID": DEVICE_ID}
    checks = []
    profiles = []

    def set_profile(season, shape):
        with SessionLocal() as db:
            row = db.get(AnalysisHistory, DEVICE_ID)
            if row is None:
                row = AnalysisHistory(device_id=DEVICE_ID)
                db.add(row)
            row.color_sub_season, row.body_shape = season, shape
            db.commit()

    def check_response(client, name, path, status, **kwargs):
        response = client.get(path, **kwargs)
        if response.status_code != status:
            raise ValueError(f"{name}: expected {status}, got {response.status_code}")
        checks.append({"scenario": name, "http_status": status})
        return response.json()

    try:
        with TestClient(app) as client:
            for name, season, shape, error, missing in (
                ("missing_both", None, None, "missing_analysis", ["color", "body"]),
                (
                    "missing_color",
                    None,
                    "Rectangle",
                    "missing_color_analysis",
                    ["color"],
                ),
                (
                    "missing_body",
                    "Soft Autumn",
                    None,
                    "missing_body_analysis",
                    ["body"],
                ),
            ):
                set_profile(season, shape)
                body = check_response(
                    client, name, RECOMMEND_PATH, 400, headers=headers
                )
                if (
                    body["detail"]["error"] != error
                    or body["detail"]["missing"] != missing
                ):
                    raise ValueError(f"{name}: missing-analysis detail changed")
            for name, request_headers, parameters in (
                ("missing_header", {}, {}),
                ("short_device_id", {"X-Device-ID": "short"}, {}),
                ("invalid_occasion", headers, {"occasion": "unknown"}),
                ("unsupported_gender", headers, {"gender": "female"}),
            ):
                check_response(
                    client,
                    name,
                    RECOMMEND_PATH,
                    422,
                    headers=request_headers,
                    params=parameters,
                )
            readiness = check_response(client, "ready", "/health/ready", 200)
            if not readiness["recommendation"]["ready"]:
                raise ValueError("v2 bundle did not pass API readiness")
            with SessionLocal() as db:
                records = {row.id: row for row in db.query(Outfit).all()}
            for season in SEASON_KEYWORDS:
                for shape in BODY_KEYWORDS:
                    set_profile(season, shape)
                    for occasion in OCCASION_KEYWORDS:
                        first = client.get(
                            RECOMMEND_PATH,
                            headers=headers,
                            params={"occasion": occasion},
                        )
                        second = client.get(
                            RECOMMEND_PATH,
                            headers=headers,
                            params={"occasion": occasion},
                        )
                        if first.status_code != 200 or first.json() != second.json():
                            raise ValueError("Recommendation response is not stable")
                        payload = first.json()
                        if not payload["success"]:
                            raise ValueError(
                                "Expected successful recommendation envelope"
                            )
                        data = payload["data"]
                        if data["query_source"] != "precomputed_query_vector":
                            raise ValueError(
                                "Recommendation did not use the saved query"
                            )
                        if data["query_used"] != build_outfit_query(
                            season, shape, occasion
                        ):
                            raise ValueError("API query disagrees with the profile")
                        items = data["outfits"]
                        if len(items) != 6 or len(
                            {o["image_url"] for o in items}
                        ) != len(items):
                            raise ValueError("Expected six image-distinct results")
                        for item in items:
                            row = records[item["id"]]
                            if (
                                row.occasion != occasion
                                or row.gender != "male"
                                or row.vector_blob is None
                            ):
                                raise ValueError("API returned an ineligible outfit")
                            if not 0 <= item["score"] <= 1:
                                raise ValueError("API match score is out of bounds")
                        profiles.append(
                            {
                                "season": season,
                                "shape": shape,
                                "occasion": occasion,
                                "outfit_ids": [item["id"] for item in items],
                            }
                        )
            # Detail is static catalog data, including rows whose image is unavailable.
            for outfit_id, row in records.items():
                detail = client.get(f"/api/v1/outfits/{outfit_id}")
                if (
                    detail.status_code != 200
                    or detail.json()["data"]["name"] != row.name
                ):
                    raise ValueError("Outfit detail did not preserve catalog identity")
            check_response(client, "missing_detail", "/api/v1/outfits/999999", 404)
            set_profile("Soft Autumn", "Rectangle")
            default = check_response(
                client, "default_occasion", RECOMMEND_PATH, 200, headers=headers
            )
            if default["data"]["query_used"] != build_outfit_query(
                "Soft Autumn", "Rectangle", "daily"
            ):
                raise ValueError("Default occasion is no longer daily")
            with SessionLocal() as db:
                row = (
                    db.query(Outfit)
                    .filter(Outfit.occasion == "daily", Outfit.vector_blob.isnot(None))
                    .first()
                )
                original_version = row.embedding_version
                row.embedding_version = "deliberately-mixed-test-version"
                db.commit()
                mixed_id = row.id
            mixed = check_response(
                client, "mixed_catalog_versions", RECOMMEND_PATH, 200, headers=headers
            )
            if mixed["success"] or mixed["error_code"] != "RECOMMENDATION_NOT_READY":
                raise ValueError("A mixed-version catalog was accepted")
            with SessionLocal() as db:
                db.get(Outfit, mixed_id).embedding_version = original_version
                db.commit()
            with SessionLocal() as db:
                db.query(Outfit).filter(Outfit.vector_blob.isnot(None)).update(
                    {
                        Outfit.embedding_version: "deliberately-incompatible-test-version"
                    },
                    synchronize_session=False,
                )
                db.commit()
            mismatch = check_response(
                client,
                "query_catalog_version_mismatch",
                RECOMMEND_PATH,
                200,
                headers=headers,
            )
            if (
                mismatch["success"]
                or mismatch["error_code"] != "RECOMMENDATION_NOT_READY"
            ):
                raise ValueError(
                    "Mismatched versions must fail in the existing envelope"
                )
            with SessionLocal() as db:
                db.query(Outfit).delete()
                db.commit()
            empty = check_response(
                client, "empty_catalog", RECOMMEND_PATH, 200, headers=headers
            )
            if empty["success"] or empty["error_code"] != "RECOMMENDATION_NOT_READY":
                raise ValueError("An empty catalog was accepted")
        return {
            "passed": True,
            "profiles_checked": len(profiles),
            "detail_rows_checked": len(records),
            "scenarios": checks,
            "profiles": profiles,
            "gemini_calls": 0,
            "runtime_encoder_calls": 0,
            "transport": "in-process ASGI TestClient; no listening HTTP server",
            "profile_origin": "hypothetical fixtures in disposable DB only",
        }
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal isolated API worker")
    parser.add_argument("--database-copy", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    # pytest-socket is already part of the lightweight test environment. Permit
    # loopback for AnyIO's event loop while rejecting external TCP connections.
    from pytest_socket import socket_allow_hosts

    socket_allow_hosts(["127.0.0.1", "::1"], allow_unix_socket=True)
    write_json(args.report, exercise_routes(args.database_copy))


if __name__ == "__main__":
    main()
