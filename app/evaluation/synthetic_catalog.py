"""Create explicitly synthetic SQLite/query fixtures under a caller-owned directory.

This is shared by software smoke checks, never by production data preparation.
No model is encoded and no committed vector file is overwritten.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from app.services.recommendation.query_builder import (
    _all_query_templates,
    _current_query_signature,
)

from .provenance import PROJECT_ROOT
from .sqlite_catalog import CATALOG_FIELDS


def write_synthetic_catalog_sources(directory: Path) -> tuple[Path, Path, Path, Path]:
    source = json.loads(
        (PROJECT_ROOT / "evals/synthetic/retrieval.snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    if not source["synthetic"]:
        raise ValueError("Smoke sources must be explicitly synthetic")
    database = directory / "synthetic-catalog.db"
    with closing(sqlite3.connect(database)) as connection:
        definitions = [
            f"{field} {'INTEGER PRIMARY KEY' if field == 'id' else 'TEXT'}"
            for field in CATALOG_FIELDS
        ]
        connection.execute(
            "CREATE TABLE outfits ("
            + ", ".join(definitions)
            + ", vector_blob BLOB, embedding_backend TEXT, embedding_model TEXT, embedding_version TEXT)"
        )
        fields = (
            *CATALOG_FIELDS,
            "vector_blob",
            "embedding_backend",
            "embedding_model",
            "embedding_version",
        )
        for outfit in source["outfits"]:
            values = [outfit.get(field, "") for field in CATALOG_FIELDS]
            values.extend(
                [
                    np.asarray(outfit["vector"], dtype="<f4").tobytes(),
                    source["embedding"]["backend"],
                    source["embedding"]["model"],
                    source["embedding"]["version"],
                ]
            )
            connection.execute(
                f"INSERT INTO outfits ({', '.join(fields)}) VALUES ({', '.join('?' for _ in fields)})",
                values,
            )
        connection.commit()
    request = {
        key: source[key]
        for key in ("dataset_id", "version", "synthetic", "annotation_protocol")
    }
    request["cases"] = [
        {
            key: case[key]
            for key in ("case_id", "subject_id", "split", "profile", "relevance")
        }
        for case in source["cases"]
    ]
    request_path = directory / "export-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    queries = directory / "query_vectors.json"
    queries.write_text(
        json.dumps({text: [1.0] + [0.0] * 511 for text in _all_query_templates()}),
        encoding="utf-8",
    )
    metadata = directory / "query_vectors.meta.json"
    metadata.write_text(
        json.dumps(
            {
                **source["embedding"],
                "query_count": 240,
                "query_signature": _current_query_signature(),
            }
        ),
        encoding="utf-8",
    )
    return request_path, database, queries, metadata
