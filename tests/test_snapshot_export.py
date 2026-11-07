"""Exporter contract tests use isolated SQLite and explicitly synthetic vectors."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.evaluation.preflight import preflight_retrieval
from app.evaluation.retrieval import evaluate_snapshot
from app.evaluation.snapshot_export import export_snapshot
from app.evaluation.sqlite_catalog import CATALOG_FIELDS, readonly_catalog
from app.services.recommendation.query_builder import (
    _all_query_templates,
    _current_query_signature,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def export_inputs(tmp_path):
    snapshot = json.loads(
        (ROOT / "evals/synthetic/retrieval.snapshot.json").read_text(encoding="utf-8")
    )
    database = tmp_path / "catalog # unicode-é.db"
    with sqlite3.connect(database) as connection:
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
        for outfit in snapshot["outfits"]:
            values = [outfit.get(field, "") for field in CATALOG_FIELDS]
            values += [
                np.asarray(outfit["vector"], dtype="<f4").tobytes(),
                "synthetic",
                "handcrafted-unit-vectors",
                "1",
            ]
            connection.execute(
                f"INSERT INTO outfits ({', '.join(fields)}) VALUES ({', '.join('?' for _ in fields)})",
                values,
            )
        connection.execute("CREATE TABLE private_history (secret TEXT)")
        connection.execute(
            "INSERT INTO private_history VALUES ('must not be exported')"
        )
    request = {
        key: snapshot[key]
        for key in ("dataset_id", "version", "synthetic", "annotation_protocol")
    }
    request["cases"] = [
        {
            key: case[key]
            for key in ("case_id", "subject_id", "split", "profile", "relevance")
        }
        for case in snapshot["cases"]
    ]
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    vectors = tmp_path / "query_vectors.json"
    vectors.write_text(
        json.dumps({query: [1.0] + [0.0] * 511 for query in _all_query_templates()}),
        encoding="utf-8",
    )
    metadata = tmp_path / "query_vectors.meta.json"
    metadata.write_text(
        json.dumps(
            {
                "backend": "synthetic",
                "model": "handcrafted-unit-vectors",
                "version": "1",
                "query_count": 240,
                "query_signature": _current_query_signature(),
            }
        ),
        encoding="utf-8",
    )
    return request_path, database, vectors, metadata


def test_export_is_read_only_and_enters_existing_preflight_ablation(
    export_inputs, tmp_path
):
    original = {path: path.read_bytes() for path in export_inputs}
    output = tmp_path / "export"
    provenance = export_snapshot(*export_inputs, output)
    for path, content in original.items():
        assert path.read_bytes() == content
    assert provenance["synthetic"] is True
    assert provenance["encoded_outfit_count"] == 13
    assert preflight_retrieval(output / "snapshot.json")["ready"]
    assert set(evaluate_snapshot(output / "snapshot.json", split="dev")["summary"]) == {
        "filtered_random",
        "vector_only",
        "hybrid",
        "hybrid_diverse",
    }
    assert "must not be exported" not in (output / "snapshot.json").read_text()
    assert not list(tmp_path.glob("*-journal"))


def test_connection_rejects_sql_writes_and_does_not_create_missing_db(
    export_inputs, tmp_path
):
    with readonly_catalog(export_inputs[1]) as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM outfits")
    missing = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        with readonly_catalog(missing):
            pass
    assert not missing.exists()


@pytest.mark.parametrize(
    "failure", ["mixed", "dummy", "short", "zero", "nan", "no_vectors"]
)
def test_invalid_catalog_never_produces_snapshot(export_inputs, tmp_path, failure):
    with sqlite3.connect(export_inputs[1]) as connection:
        if failure == "mixed":
            connection.execute("UPDATE outfits SET embedding_model='other' WHERE id=1")
        elif failure == "dummy":
            connection.execute("UPDATE outfits SET embedding_backend='dummy'")
        elif failure == "no_vectors":
            connection.execute("UPDATE outfits SET vector_blob=NULL")
        else:
            vector = np.zeros(512, dtype=np.float32)
            if failure == "short":
                vector = np.ones(2, dtype=np.float32)
            elif failure == "nan":
                vector[0] = np.nan
            connection.execute(
                "UPDATE outfits SET vector_blob=? WHERE id=1", (vector.tobytes(),)
            )
    output = tmp_path / "export"
    with pytest.raises(ValueError):
        export_snapshot(*export_inputs, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "failure",
    [
        "signature",
        "identity",
        "count",
        "missing_query",
        "bad_query_vector",
        "unknown_judgment",
    ],
)
def test_invalid_saved_queries_or_request_never_falls_back(
    export_inputs, tmp_path, failure
):
    request, _database, vectors, metadata = export_inputs
    if failure == "unknown_judgment":
        data = json.loads(request.read_text())
        data["cases"][0]["relevance"]["999"] = 3
        request.write_text(json.dumps(data))
    elif failure in {"missing_query", "bad_query_vector"}:
        data = json.loads(vectors.read_text())
        first = next(iter(data))
        if failure == "missing_query":
            del data[first]
        else:
            data[first] = [0.0] * 512
        vectors.write_text(json.dumps(data))
    else:
        data = json.loads(metadata.read_text())
        data[
            {
                "signature": "query_signature",
                "identity": "model",
                "count": "query_count",
            }[failure]
        ] = 239 if failure == "count" else "wrong"
        metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        export_snapshot(*export_inputs, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_missing_schema_does_not_trigger_migration(export_inputs, tmp_path):
    database = tmp_path / "empty.db"
    with sqlite3.connect(database):
        pass
    original = database.read_bytes()
    with pytest.raises(ValueError, match="no migration"):
        export_snapshot(
            export_inputs[0],
            database,
            export_inputs[2],
            export_inputs[3],
            tmp_path / "export",
        )
    assert database.read_bytes() == original


def test_existing_output_is_not_overwritten(export_inputs, tmp_path):
    with pytest.raises(FileExistsError):
        export_snapshot(*export_inputs, tmp_path)


def test_unencoded_rows_are_counted_without_becoming_fake_candidates(
    export_inputs, tmp_path
):
    with sqlite3.connect(export_inputs[1]) as connection:
        connection.execute("INSERT INTO outfits (id) VALUES (100)")
    provenance = export_snapshot(*export_inputs, tmp_path / "export")
    assert provenance["excluded_unencoded_count"] == 1
    assert provenance["encoded_outfit_count"] == 13
    payload = json.loads(
        (tmp_path / "export/snapshot.json").read_text(encoding="utf-8")
    )
    assert 100 not in {outfit["id"] for outfit in payload["outfits"]}


def test_wal_committed_rows_are_visible_without_checkpointing_main_database(
    export_inputs, tmp_path
):
    database = export_inputs[1]
    with sqlite3.connect(database) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("UPDATE outfits SET name='committed in WAL' WHERE id=1")
        writer.commit()
        original = database.read_bytes()
        export_snapshot(*export_inputs, tmp_path / "export")
        exported = json.loads(
            (tmp_path / "export/snapshot.json").read_text(encoding="utf-8")
        )
        assert exported["outfits"][0]["name"] == "committed in WAL"
        assert database.read_bytes() == original


def test_cli_ignores_application_db_configuration_and_needs_no_key(
    export_inputs, tmp_path
):
    environment = os.environ.copy()
    environment.pop("GEMINI_API_KEY", None)
    environment["DATABASE_URL"] = "invalid-url-that-must-never-be-used"
    code = (
        "import sys; from scripts.export_retrieval_snapshot import main; main(); "
        "assert 'app.db.database' not in sys.modules; "
        "assert 'app.services.recommendation.embedding_store' not in sys.modules"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "--request",
            str(export_inputs[0]),
            "--database",
            str(export_inputs[1]),
            "--query-vectors",
            str(export_inputs[2]),
            "--query-metadata",
            str(export_inputs[3]),
            "--output-dir",
            str(tmp_path / "export"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
