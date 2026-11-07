"""Offline preparation regressions with generated images and a temporary bundle.

The fixture contains explicit test-only vectors, one unavailable image and one
shared-image metadata conflict. It is software test input, not an AI benchmark.
The API subprocess uses a disposable backup, never this fixture's source DB.
"""

import csv
import io
import json
import zipfile
from collections import defaultdict
from html.parser import HTMLParser

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.catalog_preparation.api_verification import run_isolated_api_verification
from app.catalog_preparation.bundle import BUNDLE_IDENTITY
from app.catalog_preparation.catalog_review import prepare_catalog_review
from app.catalog_preparation.encoder import file_sha256
from app.catalog_preparation.pilot import (
    create_pilot_plan,
    prepare_relevance_pilot,
    validate_blank_forms,
)
from app.catalog_preparation.review_assets import (
    cached_image_path,
    copy_local_images,
    load_sources,
)
from app.db.database import Base
from app.db.models import Outfit
from app.evaluation.annotation_csv import read_ratings
from app.evaluation.annotation_schema import AnnotationPacket
from app.services.recommendation.query_builder import (
    _all_query_templates,
    _current_query_signature,
)


class HTMLAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images, self.scripts = [], []
        self.policy = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "img":
            self.images.append(values["src"])
        if tag == "script":
            self.scripts.append(values)
        if tag == "meta" and values.get("http-equiv") == "Content-Security-Policy":
            self.policy = values["content"]


@pytest.fixture
def review_bundle(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    (directory / "images").mkdir()
    csv_path = directory / "catalog.csv"
    fields = ["name", "category", "image_url", "shopee_url", "price_range", "occasion"]
    seed, grouped = [], defaultdict(list)
    database = directory / "catalog.db"
    engine = create_engine("sqlite:///" + database.as_posix())
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        for index in range(1, 29):
            occasion = ["daily", "work", "party", "date"][(index - 1) // 7]
            image_id = 1 if index == 2 else index
            image_url = f"https://example.org/image/{image_id}"
            row = {
                "name": "<script>injected()</script>"
                if index == 1
                else f"Test item {index}",
                "category": "top" if index % 2 else "bottom",
                "image_url": image_url,
                "shopee_url": f"https://example.org/item/{index}",
                "price_range": "100k",
                "occasion": occasion,
            }
            seed.append(row)
            grouped[image_url].append(index)
            db.add(
                Outfit(
                    id=index,
                    **row,
                    gender="male",
                    fit="regular",
                    formality="casual",
                    fabric="cotton",
                    color_family="earth",
                    season_bias="Soft Autumn",
                    body_shape_bias=None if index == 3 else "Rectangle",
                    vector_blob=None
                    if index == 28
                    else np.array([1.0] + [0.0] * 511, dtype="<f4").tobytes(),
                    embedding_backend=None
                    if index == 28
                    else BUNDLE_IDENTITY["backend"],
                    embedding_model=None if index == 28 else BUNDLE_IDENTITY["model"],
                    embedding_version=None
                    if index == 28
                    else BUNDLE_IDENTITY["version"],
                )
            )
        db.commit()
    engine.dispose()
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(seed)
    sources = []
    for url, indices in grouped.items():
        if indices == [28]:
            sources.append(
                {
                    "url": url,
                    "csv_rows": indices,
                    "status": "http_error",
                    "http_status": 403,
                }
            )
            continue
        buffer = io.BytesIO()
        Image.new("RGB", (320, 400), (indices[0] * 7 % 255, 20, 90)).save(
            buffer, format="PNG"
        )
        filename = f"test-{indices[0]}.image"
        image = directory / "images" / filename
        image.write_bytes(buffer.getvalue())
        sources.append(
            {
                "url": url,
                "csv_rows": indices,
                "status": "ready",
                "cache_file": filename,
                "file_sha256": file_sha256(image),
                "width": 320,
                "height": 400,
            }
        )
    (directory / "sources.json").write_text(json.dumps({"sources": sources}))
    (directory / "query_vectors.json").write_text(
        json.dumps({text: [1.0] + [0.0] * 511 for text in _all_query_templates()})
    )
    (directory / "query_vectors.meta.json").write_text(
        json.dumps(
            {
                **BUNDLE_IDENTITY,
                "query_count": 240,
                "query_signature": _current_query_signature(),
            }
        )
    )
    files = (
        "catalog.db",
        "sources.json",
        "query_vectors.json",
        "query_vectors.meta.json",
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "embedding": BUNDLE_IDENTITY,
                "artifact_sha256": {
                    name: file_sha256(directory / name) for name in files
                },
                "baseline_sha256": {str(csv_path): file_sha256(csv_path)},
            }
        )
    )
    return directory, csv_path


def test_api_checks_real_routes_and_leaves_source_unchanged(review_bundle, tmp_path):
    directory, _ = review_bundle
    before = (directory / "catalog.db").read_bytes()
    report = run_isolated_api_verification(directory, tmp_path / "api")
    assert report["profiles_checked"] == 240
    assert report["detail_rows_checked"] == 28
    assert report["source_unchanged"] and report["temporary_database_removed"]
    assert report["gemini_calls"] == report["runtime_encoder_calls"] == 0
    scenarios = {s["scenario"]: s["http_status"] for s in report["scenarios"]}
    assert scenarios["missing_color"] == scenarios["missing_body"] == 400
    assert (
        scenarios["mixed_catalog_versions"]
        == scenarios["query_catalog_version_mismatch"]
        == 200
    )
    assert scenarios["empty_catalog"] == 200
    assert scenarios["unsupported_gender"] == 422
    assert (directory / "catalog.db").read_bytes() == before


def test_catalog_audit_and_gallery_preserve_bytes_and_escape_text(
    review_bundle, tmp_path
):
    directory, csv_path = review_bundle
    before = (directory / "catalog.db").read_bytes()
    output = tmp_path / "gallery"
    report = prepare_catalog_review(directory, csv_path, output)
    assert report["row_count"] == 28
    assert report["issue_counts"]["missing_metadata"] == 1
    assert report["issue_counts"]["image_unavailable"] == 1
    conflict = next(
        i for i in report["issues"] if i["code"] == "shared_image_metadata_conflict"
    )
    assert conflict["row_ids"] == [1, 2] and "category" in conflict["fields"]
    html = (output / "gallery.html").read_text(encoding="utf-8")
    parser = HTMLAssets()
    parser.feed(html)
    assert not parser.scripts and "default-src 'none'" in parser.policy
    assert "file:" in parser.policy
    assert "&lt;script&gt;" in html
    assert all(
        not path.startswith(("http:", "https:")) and (output / path).is_file()
        for path in parser.images
    )
    assert len(set(parser.images)) == 26
    assert (directory / "catalog.db").read_bytes() == before
    with pytest.raises(FileExistsError):
        prepare_catalog_review(directory, csv_path, output)


@pytest.mark.parametrize("failure", ["modified", "escape"])
def test_cached_images_must_match_and_stay_inside_bundle(
    review_bundle, tmp_path, failure
):
    directory, _ = review_bundle
    source = next(s for s in load_sources(directory) if s.status == "ready")
    if failure == "modified":
        (directory / "images" / source.cache_file).write_bytes(b"changed")
    else:
        from dataclasses import replace

        outside = directory / "outside.image"
        outside.write_bytes(b"outside")
        source = replace(
            source, cache_file="../outside.image", file_sha256=file_sha256(outside)
        )
    with pytest.raises(ValueError):
        cached_image_path(directory, source)


def test_copied_images_are_exact_original_bytes(review_bundle, tmp_path):
    directory, _ = review_bundle
    output = tmp_path / "offline"
    output.mkdir()
    mapping = copy_local_images(directory, output)
    for source in load_sources(directory):
        if source.status == "ready":
            assert file_sha256(output / mapping[source.url]) == source.file_sha256


def test_scenario_plan_coverage_is_explicitly_hypothetical_dev():
    plan = create_pilot_plan()
    assert len(plan.cases) == 20
    assert {case.split for case in plan.cases} == {"dev"}
    assert all(
        case.subject_id.startswith("hypothetical-profile-") for case in plan.cases
    )
    assert len({case.profile.color_sub_season for case in plan.cases}) == 12
    assert (
        len({(case.profile.body_shape, case.profile.occasion) for case in plan.cases})
        == 20
    )
    assert "hypothetical" in plan.annotation_protocol


def prepare_pilot_fixture(review_bundle, tmp_path):
    directory, csv_path = review_bundle
    audit = tmp_path / "audit"
    prepare_catalog_review(directory, csv_path, audit)
    output = tmp_path / "pilot"
    report = prepare_relevance_pilot(directory, audit / "report.json", output)
    return output, report


def test_offline_pilot_is_blinded_blank_and_portable(review_bundle, tmp_path):
    output, report = prepare_pilot_fixture(review_bundle, tmp_path)
    packet_dir = output / "packet"
    packet = AnnotationPacket.model_validate_json(
        (packet_dir / "private/packet.json").read_text()
    )
    assert report["ratings_completed"] == 0
    assert report["form_check"]["different_orders"]
    for reviewer in packet.reviewers:
        directory = packet_dir / reviewer
        archive = output / report["reviewer_archives"][reviewer]["filename"]
        assert file_sha256(archive) == report["reviewer_archives"][reviewer]["sha256"]
        with zipfile.ZipFile(archive) as zipped:
            assert all(name.startswith(reviewer + "/") for name in zipped.namelist())
            assert not any("private/" in name for name in zipped.namelist())
        html = (directory / "reference.html").read_text(encoding="utf-8")
        parser = HTMLAssets()
        parser.feed(html)
        assert not parser.scripts
        assert all((directory / path).is_file() for path in parser.images)
        assert "season_bias" not in html and "body_shape_bias" not in html
        assert "retrieval_score" not in html and "confidence" not in html
        with pytest.raises(ValueError, match="blank grades"):
            read_ratings(directory / "ratings.csv", packet, reviewer)


@pytest.mark.parametrize(
    "field, value", [("grade", "3"), ("outfit_name", "edited name")]
)
def test_preparation_rejects_filled_grades_or_changed_context(
    review_bundle, tmp_path, field, value
):
    output, _ = prepare_pilot_fixture(review_bundle, tmp_path)
    packet_dir = output / "packet"
    packet = AnnotationPacket.model_validate_json(
        (packet_dir / "private/packet.json").read_text()
    )
    path = packet_dir / packet.reviewers[0] / "ratings.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    rows[0][field] = value
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        validate_blank_forms(packet_dir, packet)


@pytest.mark.parametrize("problem", ["bundle", "structural"])
def test_pilot_rejects_wrong_or_structurally_invalid_audit(
    review_bundle, tmp_path, problem
):
    directory, csv_path = review_bundle
    audit = tmp_path / "audit"
    report = prepare_catalog_review(directory, csv_path, audit)
    if problem == "bundle":
        report["integrity"]["manifest_sha256"] = "0" * 64
    else:
        report["severity_counts"]["error"] = 1
    (audit / "report.json").write_text(json.dumps(report))
    output = tmp_path / "bad-pilot"
    with pytest.raises(ValueError):
        prepare_relevance_pilot(directory, audit / "report.json", output)
    assert not output.exists()
