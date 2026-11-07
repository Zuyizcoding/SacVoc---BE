"""Prepare a separate catalog using a strict compatibility gate.

Intuition: preparation must never repair a model mismatch by relabeling vectors.
Schema: catalog.db retains every CSV row; unavailable images have NULL vectors.
reports contain source availability, encoder provenance and all query comparisons.
Pipeline: validate inputs -> audit images -> seed NEW DB -> verify text geometry
-> encode available images only if every query passed -> commit one transaction.

The legacy seed CLI runs in a child process with an explicit DATABASE_URL.
This reuses its metadata inference/upsert rules without importing its configured
engine in this process. CSV keys must be unique, so no row can be merged away.
"""

import csv
import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .compatibility import (
    IDENTITY,
    normalized_vectors,
    read_reference_queries,
    verify_compatibility,
)
from .encoder import file_sha256
from .sources import ImageSource, audit_sources

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ImageEncoder(Protocol):
    def encode_image(self, image_path: Path) -> np.ndarray: ...


class CatalogEncoder(ImageEncoder, Protocol):
    provenance: dict

    def encode_text(self, texts: list[str]) -> np.ndarray: ...


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_seed_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "name",
            "category",
            "image_url",
            "shopee_url",
            "price_range",
            "occasion",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Catalog CSV lacks required seed columns")
        rows = [{key: value.strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError("Catalog CSV is empty")
    for key in ("name", "shopee_url"):
        values = [row[key] for row in rows]
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f"Separate catalog requires unique, non-empty {key}")
    return rows


def seed_separate_database(database: Path, csv_path: Path, rows: list[dict]) -> None:
    """Use the actual ORM schema and seed rules, with no user DB connection."""
    if database.exists():
        raise FileExistsError(database)
    environment = {
        **os.environ,
        "DATABASE_URL": "sqlite:///" + database.resolve().as_posix(),
        "DEBUG": "false",
        "PYTHONIOENCODING": "utf-8",
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.seed_outfits_csv",
            "--csv",
            str(csv_path.resolve()),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
    )
    with closing(sqlite3.connect(database)) as connection:
        stored = connection.execute(
            "SELECT id, name, image_url FROM outfits ORDER BY id"
        ).fetchall()
        expected = [
            (index, row["name"], row["image_url"])
            for index, row in enumerate(rows, start=1)
        ]
        if stored != expected:
            raise ValueError("Seeded rows do not retain the CSV order and identity")
        if connection.execute("SELECT count(*) FROM analysis_history").fetchone()[0]:
            raise ValueError("Separate catalog must not contain device history")


def encode_available_images(
    database: Path,
    encoder: ImageEncoder,
    sources: list[ImageSource],
    cache: Path,
    *,
    identity: dict[str, str] | None = None,
) -> int:
    """Commit every successful image vector together, leaving failed URLs NULL."""
    encoded = 0
    embedding_identity = IDENTITY if identity is None else identity
    timestamp = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            for source in sources:
                if source.status != "ready":
                    continue
                if not source.cache_file or not source.file_sha256:
                    raise ValueError("Ready images need a cache path and checksum")
                path = cache / source.cache_file
                if file_sha256(path) != source.file_sha256:
                    raise ValueError("Cached image changed after source inspection")
                vector = normalized_vectors(encoder.encode_image(path), 1)[0]
                blob = vector.astype("<f4").tobytes()
                for row_id in source.csv_rows:
                    connection.execute(
                        "UPDATE outfits SET vector_blob=?, embedding_backend=?, "
                        "embedding_model=?, embedding_version=?, embedding_generated_at=? "
                        "WHERE id=?",
                        (
                            blob,
                            embedding_identity["backend"],
                            embedding_identity["model"],
                            embedding_identity["version"],
                            timestamp,
                            row_id,
                        ),
                    )
                    encoded += 1
    return encoded


def prepare_catalog(
    csv_path: Path,
    query_path: Path,
    metadata_path: Path,
    output: Path,
    encoder_factory: Callable[[], CatalogEncoder],
) -> dict:
    """A fresh output directory is required; reruns cannot overwrite a prior DB."""
    rows = read_seed_rows(csv_path)
    reference = read_reference_queries(query_path, metadata_path)
    inputs = {
        str(path.resolve()): file_sha256(path)
        for path in (csv_path, query_path, metadata_path)
    }
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "preparing",
        "input_sha256": inputs,
        "csv_rows": len(rows),
        "database": "catalog.db",
        "encoded_rows": 0,
    }
    write_json(output / "summary.json", summary)
    print(f"Auditing {len(rows)} catalog rows...", flush=True)
    sources = audit_sources(rows, output / "images")
    write_json(output / "sources.json", {"sources": [s.to_dict() for s in sources]})
    summary.update(
        unique_urls=len(sources),
        source_status_counts=dict(Counter(s.status for s in sources)),
        available_rows=sum(len(s.csv_rows) for s in sources if s.status == "ready"),
        unique_file_hashes=len({s.file_sha256 for s in sources if s.status == "ready"}),
        unique_rgb_hashes=len({s.rgb_sha256 for s in sources if s.status == "ready"}),
    )
    database = output / "catalog.db"
    seed_separate_database(database, csv_path, rows)
    print("Checking encoder against every committed query...", flush=True)
    encoder = encoder_factory()
    write_json(output / "encoder.json", encoder.provenance)
    compatibility = verify_compatibility(encoder, reference)
    write_json(output / "compatibility.json", compatibility)
    if not compatibility["compatible"]:
        summary["status"] = "blocked_incompatible_encoder"
    else:
        print("Compatibility passed. Encoding available images...", flush=True)
        summary["encoded_rows"] = encode_available_images(
            database, encoder, sources, output / "images"
        )
        summary["status"] = (
            "encoded" if summary["encoded_rows"] else "blocked_no_available_images"
        )
    summary["unencoded_rows"] = len(rows) - summary["encoded_rows"]
    summary["input_files_unchanged"] = all(
        file_sha256(Path(path)) == digest for path, digest in inputs.items()
    )
    if not summary["input_files_unchanged"]:
        raise ValueError("Input files changed during preparation; discard this run")
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "summary.json", summary)
    return summary
