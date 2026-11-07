"""Exercise the complete new-bundle pipeline with fake AI/HTTP and a temp DB."""

import csv
import io
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.catalog_preparation import bundle
from app.catalog_preparation.encoder import file_sha256
from app.catalog_preparation.sources import ImageSource
from app.catalog_preparation.validation import verify_serving_paths
from scripts.verify_embedding_bundle import verify_manifest

ROOT = Path(__file__).resolve().parents[1]


class FakeReferenceEncoder:
    provenance = {"test_only": True, "no_real_model": True}

    def __init__(self, *, broken_text=False, broken_image=False):
        self.broken_text = broken_text
        self.broken_image = broken_image
        self.image_calls = 0

    def encode_text(self, texts):
        return np.tile([1.0] + [0.0] * 511, (len(texts), 1))

    def encode_reference_text(self, texts):
        return self.encode_text(texts) * (-1 if self.broken_text else 1)

    def encode_image(self, image_path):
        self.image_calls += 1
        return self.encode_text(["image"])

    def encode_reference_image(self, image_path):
        sign = -1 if self.broken_image and self.image_calls == 2 else 1
        return self.encode_text(["image"]) * sign


@pytest.fixture
def catalog_csv(tmp_path):
    path = tmp_path / "catalog.csv"
    fields = [
        "name",
        "category",
        "image_url",
        "shopee_url",
        "price_range",
        "occasion",
        "gender",
    ]
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for index, occasion in enumerate(
            ["daily", "work", "party", "date"] * 2, start=1
        ):
            writer.writerow(
                {
                    "name": f"Item {index}",
                    "category": "top",
                    "image_url": f"https://example.org/image/{index}",
                    "shopee_url": f"https://example.org/product/{index}",
                    "price_range": "100k",
                    "occasion": occasion,
                    "gender": "male",
                }
            )
    return path


def fake_audit(rows, cache, *, missing_last=False):
    cache.mkdir()
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    result = []
    for index, row in enumerate(rows, start=1):
        if missing_last and index == len(rows):
            result.append(
                ImageSource(
                    url=row["image_url"],
                    csv_rows=[index],
                    status="http_error",
                    http_status=403,
                )
            )
            continue
        path = cache / f"{index}.image"
        path.write_bytes(buffer.getvalue())
        result.append(
            ImageSource(
                url=row["image_url"],
                csv_rows=[index],
                status="ready",
                cache_file=path.name,
                file_sha256=file_sha256(path),
            )
        )
    return result


def build(csv_path, output, encoder):
    return bundle.build_corrected_bundle(
        csv_path,
        ROOT / "data/query_vectors.json",
        ROOT / "data/query_vectors.meta.json",
        output,
        lambda: encoder,
    )


def test_new_bundle_is_separate_versioned_partial_and_servable(
    tmp_path, monkeypatch, catalog_csv
):
    monkeypatch.setattr(
        bundle,
        "audit_sources",
        lambda rows, cache: fake_audit(rows, cache, missing_last=True),
    )
    baseline = (ROOT / "data/query_vectors.json").read_bytes()
    output = tmp_path / "v2"
    summary = build(catalog_csv, output, FakeReferenceEncoder())
    assert summary["status"] == "verified_bundle"
    assert summary["encoded_rows"] == 7 and summary["unencoded_rows"] == 1
    assert summary["unavailable_csv_rows"] == [8]
    assert summary["input_files_unchanged"] and not summary["application_switched"]
    assert (ROOT / "data/query_vectors.json").read_bytes() == baseline
    assert verify_manifest(output)["baseline_unchanged"]
    before = (output / "catalog.db").read_bytes()
    assert verify_serving_paths(output)["profiles_checked"] == 240
    assert (output / "catalog.db").read_bytes() == before
    with sqlite3.connect(output / "catalog.db") as db:
        assert db.execute(
            "SELECT DISTINCT embedding_version FROM outfits WHERE vector_blob IS NOT NULL"
        ).fetchall() == [(bundle.BUNDLE_IDENTITY["version"],)]
        assert (
            db.execute("SELECT vector_blob FROM outfits WHERE id=8").fetchone()[0]
            is None
        )
    with pytest.raises(FileExistsError):
        build(catalog_csv, output, FakeReferenceEncoder())
    (output / "query_vectors.json").write_text("tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_manifest(output)


def test_text_reference_mismatch_prevents_image_audit_and_db_creation(
    tmp_path, monkeypatch, catalog_csv
):
    def forbidden_audit(*args):
        pytest.fail("Image work must wait for the independent query reference check")

    monkeypatch.setattr(bundle, "audit_sources", forbidden_audit)
    encoder = FakeReferenceEncoder(broken_text=True)
    output = tmp_path / "bad-text"
    with pytest.raises(ValueError, match="one-projection reference"):
        build(catalog_csv, output, encoder)
    assert encoder.image_calls == 0
    assert not (output / "catalog.db").exists()
    assert not (output / "manifest.json").exists()
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"


def test_late_image_reference_mismatch_rolls_back_and_never_publishes(
    tmp_path, monkeypatch, catalog_csv
):
    monkeypatch.setattr(bundle, "audit_sources", fake_audit)
    output = tmp_path / "bad-image"
    with pytest.raises(ValueError, match="one-projection reference"):
        build(catalog_csv, output, FakeReferenceEncoder(broken_image=True))
    with sqlite3.connect(output / "catalog.db") as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM outfits WHERE vector_blob IS NOT NULL"
            ).fetchone()[0]
            == 0
        )
    assert not (output / "manifest.json").exists()
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"
