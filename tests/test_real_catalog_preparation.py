"""Offline regression checks: fake HTTP/encoder, actual seed, temporary DB only."""

import csv
import io
import json
import sqlite3
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pytest
from PIL import Image

from app.catalog_preparation import sources, workflow
from app.catalog_preparation.compatibility import (
    normalized_vectors,
    read_reference_queries,
    verify_compatibility,
)
from app.catalog_preparation.encoder import file_sha256
from app.catalog_preparation.sources import ImageSource
from app.services.recommendation.query_builder import _all_query_templates

ROOT = Path(__file__).resolve().parents[1]


def reference_queries():
    return read_reference_queries(
        ROOT / "data/query_vectors.json", ROOT / "data/query_vectors.meta.json"
    )


class FakeEncoder:
    provenance = {"test_only": True}

    def __init__(self, compatible=True):
        self.compatible = compatible
        self.image_calls = 0

    def encode_text(self, texts):
        reference = reference_queries()
        vectors = np.array([reference[text] for text in texts])
        return vectors if self.compatible else -vectors

    def encode_image(self, path):
        self.image_calls += 1
        return np.array([[1.0] + [0.0] * 511])


def test_all_queries_must_match_not_just_model_names():
    reference = reference_queries()
    encoder = FakeEncoder()
    assert verify_compatibility(encoder, reference)["passed_queries"] == 240
    damaged = {key: list(value) for key, value in reference.items()}
    damaged[_all_query_templates()[-1]] = [
        -x for x in damaged[_all_query_templates()[-1]]
    ]
    report = verify_compatibility(encoder, damaged)
    assert not report["compatible"]
    assert report["passed_queries"] == 239


@pytest.mark.parametrize(
    "vector",
    [
        np.zeros((1, 512)),
        np.ones((1, 511)),
        np.full((1, 512), np.nan),
    ],
)
def test_invalid_vectors_are_rejected(vector):
    with pytest.raises(ValueError):
        normalized_vectors(vector, 1)


@pytest.mark.parametrize("damage", ["signature", "missing_query", "norm"])
def test_committed_query_contract_is_checked(tmp_path, damage):
    reference = reference_queries()
    metadata = json.loads((ROOT / "data/query_vectors.meta.json").read_text())
    if damage == "signature":
        metadata["query_signature"] = "stale"
    elif damage == "missing_query":
        reference.pop(next(iter(reference)))
    else:
        reference[next(iter(reference))] = [1.0] * 512
    queries = tmp_path / "queries.json"
    meta = tmp_path / "meta.json"
    queries.write_text(json.dumps(reference))
    meta.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        read_reference_queries(queries, meta)


class FakeResponse(io.BytesIO):
    status = 200
    url = "https://example.org/final.png"
    headers = {"Content-Type": "image/png"}


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 24), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def test_get_decodes_bytes_and_deduplicates_urls(tmp_path, monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return FakeResponse(png_bytes())

    monkeypatch.setattr(sources, "urlopen", get)
    rows = [{"image_url": "https://example.org/image"}] * 2
    results = sources.audit_sources(rows, tmp_path)
    assert len(calls) == len(results) == 1
    source = results[0]
    assert source.status == "ready"
    assert source.csv_rows == [1, 2]
    assert (source.width, source.height) == (16, 24)
    assert file_sha256(tmp_path / source.cache_file) == source.file_sha256


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("html", "invalid_image"),
        ("403", "http_error"),
        ("timeout", "network_error"),
    ],
)
def test_unavailable_sources_stay_explicit(tmp_path, monkeypatch, kind, expected):
    def get(url, **kwargs):
        if kind == "403":
            raise HTTPError(url, 403, "Forbidden", {}, None)
        if kind == "timeout":
            raise TimeoutError("read timed out")
        return FakeResponse(b"<html>not an image</html>")

    monkeypatch.setattr(sources, "urlopen", get)
    result = sources.inspect_source("https://example.org/image", [1], tmp_path)
    assert result.status == expected
    assert result.cache_file is None
    assert not list(tmp_path.iterdir())


def test_cache_write_errors_fail_instead_of_marking_source_unavailable(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sources, "urlopen", lambda *args, **kwargs: FakeResponse(png_bytes())
    )
    cache = tmp_path / "not-a-directory"
    cache.write_text("occupied")
    with pytest.raises(FileExistsError):
        sources.inspect_source("https://example.org/image", [1], cache)


def seed_csv(tmp_path):
    path = tmp_path / "outfits.csv"
    fields = ["name", "category", "image_url", "shopee_url", "price_range", "occasion"]
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for index in (1, 2):
            writer.writerow(
                dict(
                    name=f"Item {index}",
                    category="top",
                    image_url="https://example.org/image",
                    shopee_url=f"https://example.org/item/{index}",
                    price_range="100k",
                    occasion="daily",
                )
            )
    return path


@pytest.mark.parametrize("compatible", [False, True])
def test_gate_preserves_rows_and_never_encodes_on_mismatch(
    tmp_path, monkeypatch, compatible
):
    csv_path = seed_csv(tmp_path)
    original = csv_path.read_bytes()
    encoder = FakeEncoder(compatible)
    output = tmp_path / "prepared"

    def audit(rows, cache):
        cache.mkdir()
        image = cache / "source.image"
        image.write_bytes(png_bytes())
        return [
            ImageSource(
                url=rows[0]["image_url"],
                csv_rows=[1, 2],
                status="ready",
                file_sha256=file_sha256(image),
                rgb_sha256="test-only",
                cache_file=image.name,
            )
        ]

    monkeypatch.setattr(workflow, "audit_sources", audit)
    summary = workflow.prepare_catalog(
        csv_path,
        ROOT / "data/query_vectors.json",
        ROOT / "data/query_vectors.meta.json",
        output,
        lambda: encoder,
    )
    assert encoder.image_calls == (1 if compatible else 0)
    assert summary["encoded_rows"] == (2 if compatible else 0)
    assert csv_path.read_bytes() == original
    assert summary["input_files_unchanged"]
    with sqlite3.connect(output / "catalog.db") as db:
        assert db.execute("SELECT count(*) FROM outfits").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM analysis_history").fetchone()[0] == 0
        encoded = db.execute(
            "SELECT vector_blob, embedding_backend FROM outfits"
        ).fetchall()
    if compatible:
        assert all(
            len(blob) == 2048 and backend == "fashion_clip" for blob, backend in encoded
        )
    else:
        assert all(blob is None and backend is None for blob, backend in encoded)
        assert summary["status"] == "blocked_incompatible_encoder"
    with pytest.raises(FileExistsError):
        workflow.prepare_catalog(
            csv_path,
            ROOT / "data/query_vectors.json",
            ROOT / "data/query_vectors.meta.json",
            output,
            lambda: encoder,
        )


def test_image_encoding_rolls_back_all_vectors_on_late_failure(tmp_path):
    csv_path = seed_csv(tmp_path)
    db = tmp_path / "new.db"
    workflow.seed_separate_database(db, csv_path, workflow.read_seed_rows(csv_path))
    image = tmp_path / "image"
    image.write_bytes(png_bytes())
    source = ImageSource(
        url="https://example.org/image",
        csv_rows=[1],
        status="ready",
        file_sha256=file_sha256(image),
        cache_file=image.name,
    )

    class BrokenEncoder(FakeEncoder):
        def encode_image(self, path):
            if self.image_calls:
                raise ValueError("second image failed")
            return super().encode_image(path)

    with pytest.raises(ValueError, match="second image"):
        workflow.encode_available_images(
            db, BrokenEncoder(), [source, source], tmp_path
        )
    with sqlite3.connect(db) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM outfits WHERE vector_blob IS NOT NULL"
            ).fetchone()[0]
            == 0
        )
