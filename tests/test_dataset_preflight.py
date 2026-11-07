"""Use generated images and explicit reference fixtures, never personal photos."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app.evaluation.image_inspection import inspect_image_path
from app.evaluation.preflight import (
    preflight_analysis,
    preflight_retrieval,
    write_preflight_report,
)

ROOT = Path(__file__).resolve().parents[1]


def write_manifest(directory, cases):
    path = directory / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "dataset_id": "generated-image-unit-fixture",
                "version": "1",
                "synthetic": False,
                "reference_kind": "human_consensus",
                "annotation_protocol": "Unit-test schema fixture only: generated images and artificial labels.",
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return path


def color_case(ident, *, split="dev", image="one.png", subject=None):
    return {
        "case_id": ident,
        "subject_id": subject or ident,
        "feature": "color",
        "split": split,
        "reference": {"primary_season": "Autumn"},
        "image_path": image,
        "consent_confirmed": True,
    }


def test_readiness_and_distinct_subject_support_are_separate_from_case_count(tmp_path):
    Image.new("RGB", (200, 200), "blue").save(tmp_path / "one.png")
    Image.new("RGB", (200, 201), "red").save(tmp_path / "two.png")
    manifest = write_manifest(
        tmp_path,
        [
            color_case("one", subject="same-person"),
            color_case("two", image="two.png", subject="same-person"),
        ],
    )
    report = preflight_analysis(manifest)
    assert report["ready"]
    assert report["support"]["dev"]["case_count"] == 2
    assert report["support"]["dev"]["subject_count"] == 1
    assert report["support"]["dev"]["labels"]["color.primary_season"]["Autumn"] == {
        "cases": 2,
        "subjects": 1,
    }


def test_cross_split_byte_duplicates_block_even_with_different_subject_ids(tmp_path):
    path = tmp_path / "one.png"
    Image.new("RGB", (200, 200), "blue").save(path)
    original = path.read_bytes()
    manifest = write_manifest(
        tmp_path,
        [
            color_case("one"),
            color_case("two", split="test"),
        ],
    )
    report = preflight_analysis(manifest)
    assert not report["ready"]
    assert {group["fingerprint"] for group in report["duplicate_groups"]} == {
        "encoded_sha256",
        "rgb_sha256",
    }
    assert all(group["cross_split"] for group in report["duplicate_groups"])
    assert path.read_bytes() == original


def test_different_lossless_metadata_same_pixels_is_detected(tmp_path):
    image = Image.new("RGB", (200, 200), "blue")
    image.save(tmp_path / "one.png")
    metadata = PngInfo()
    metadata.add_text("note", "Different bytes, identical pixels")
    image.save(tmp_path / "two.png", pnginfo=metadata)
    manifest = write_manifest(
        tmp_path,
        [
            color_case("one"),
            color_case("two", split="test", image="two.png"),
        ],
    )
    report = preflight_analysis(manifest)
    assert not report["ready"]
    assert [group["fingerprint"] for group in report["duplicate_groups"]] == [
        "rgb_sha256"
    ]


def test_within_split_duplicates_are_reported_as_warnings(tmp_path):
    Image.new("RGB", (200, 200), "blue").save(tmp_path / "one.png")
    report = preflight_analysis(
        write_manifest(tmp_path, [color_case("one"), color_case("two")])
    )
    assert report["ready"]
    assert not report["errors"]
    assert {warning["code"] for warning in report["warnings"]} == {
        "WITHIN_SPLIT_DUPLICATE"
    }


@pytest.mark.parametrize(
    "kind,code",
    [
        ("missing", "IMAGE_MISSING"),
        ("corrupt", "PHOTO_UNREADABLE"),
        ("tiny", "PHOTO_TOO_SMALL"),
    ],
)
def test_image_failures_stay_in_preflight_report(tmp_path, kind, code):
    if kind == "corrupt":
        (tmp_path / "one.png").write_bytes(b"not an image")
    elif kind == "tiny":
        Image.new("RGB", (10, 10)).save(tmp_path / "one.png")
    report = preflight_analysis(write_manifest(tmp_path, [color_case("one")]))
    assert not report["ready"]
    assert report["errors"][0]["code"] == code
    assert len(report["images"]) == 1


def test_decoding_is_forced_for_truncated_images(tmp_path):
    image = tmp_path / "one.bmp"
    Image.new("RGB", (200, 200), "blue").save(image)
    image.write_bytes(image.read_bytes()[:100])
    assert inspect_image_path(image).issues == ["PHOTO_UNREADABLE"]


def test_exif_orientation_changes_fingerprint_layout_without_changing_file(tmp_path):
    source = Image.new("RGB", (200, 300), "red")
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "oriented.png"
    source.save(path, exif=exif)
    original = path.read_bytes()
    inspection = inspect_image_path(path)
    assert (inspection.width, inspection.height) == (300, 200)
    assert path.read_bytes() == original


def test_invalid_manifest_reports_schema_errors_without_copying_input(tmp_path):
    case = color_case("one")
    case["image_path"] = "private-sensitive-path.png"
    case["consent_confirmed"] = False
    report = preflight_analysis(write_manifest(tmp_path, [case]))
    assert not report["ready"]
    assert not report["inspection_complete"]
    assert report["errors"][0]["code"] == "SCHEMA_INVALID"
    assert "private-sensitive-path" not in json.dumps(report)


def test_synthetic_analysis_does_not_pretend_to_check_human_image_paths():
    report = preflight_analysis(ROOT / "evals/synthetic/analysis.dataset.json")
    assert report["ready"]
    assert report["images"] == []
    assert report["warnings"][0]["code"] == "SYNTHETIC_IMAGES_NOT_APPLICABLE"


def test_retrieval_preflight_checks_full_judgments(tmp_path):
    source = ROOT / "evals/synthetic/retrieval.snapshot.json"
    assert preflight_retrieval(source)["ready"]
    payload = json.loads(source.read_text(encoding="utf-8"))
    del payload["cases"][0]["relevance"]["1"]
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = preflight_retrieval(path)
    assert not report["ready"]
    assert report["errors"][0]["code"] == "INVALID_CANDIDATE_POOL"


def test_report_cannot_overwrite_existing_experiment(tmp_path):
    with pytest.raises(FileExistsError):
        write_preflight_report(
            tmp_path, preflight_analysis(ROOT / "evals/synthetic/analysis.dataset.json")
        )


def test_blocked_cli_writes_report_and_exits_two(tmp_path):
    path = write_manifest(tmp_path, [color_case("one")])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.preflight_dataset",
            "--kind",
            "analysis",
            "--dataset",
            str(path),
            "--output-dir",
            str(tmp_path / "report"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert (tmp_path / "report/report.json").is_file()
