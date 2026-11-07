"""Independent-rating workflow tested with explicitly synthetic catalog records."""

import csv
import json

import pytest

from app.evaluation.annotation_packet import RATING_FIELDS, prepare_annotation_packet
from app.evaluation.annotation_review import review_annotations
from app.evaluation.snapshot_export import export_snapshot
from app.evaluation.synthetic_catalog import write_synthetic_catalog_sources


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def prepared(tmp_path):
    inputs = write_synthetic_catalog_sources(tmp_path)
    request = json.loads(inputs[0].read_text(encoding="utf-8"))
    plan = {
        key: request[key]
        for key in ("dataset_id", "version", "synthetic", "annotation_protocol")
    }
    plan["cases"] = [
        {key: value for key, value in case.items() if key != "relevance"}
        for case in request["cases"]
    ]
    plan["cases"][0]["profile"]["color_confidence"] = 0.1111
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    packet_dir = tmp_path / "packet"
    packet = prepare_annotation_packet(plan_path, inputs[1], packet_dir)
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    for reviewer, path in zip(packet.reviewers, (first, second)):
        fields, rows = read_csv(packet_dir / reviewer / "ratings.csv")
        for row in rows:
            row["grade"] = "2"
            row["notes"] = "Synthetic independent-file fixture, not real human judgment"
        write_csv(path, fields, rows)
    return {
        "inputs": inputs,
        "plan": plan_path,
        "packet_dir": packet_dir,
        "packet": packet,
        "packet_path": packet_dir / "private/packet.json",
        "first": first,
        "second": second,
        "root": tmp_path,
    }


def review(prepared, name="review", resolutions=None):
    return review_annotations(
        prepared["packet_path"],
        prepared["first"],
        prepared["second"],
        prepared["root"] / name,
        resolution_path=resolutions,
    )


def add_disagreement(prepared):
    fields, rows = read_csv(prepared["second"])
    rows[0]["grade"] = "3"
    write_csv(prepared["second"], fields, rows)
    return rows[0]["case_id"], rows[0]["outfit_id"]


def test_packet_has_blank_blinded_forms_with_different_reviewer_order(prepared):
    packet = prepared["packet"]
    key_orders = []
    for reviewer in packet.reviewers:
        fields, rows = read_csv(prepared["packet_dir"] / reviewer / "ratings.csv")
        assert tuple(fields) == RATING_FIELDS
        assert all(row["grade"] == "" and row["notes"] == "" for row in rows)
        assert not {
            "vector",
            "season_bias",
            "body_shape_bias",
            "confidence",
            "score",
        } & set(fields)
        assert (
            "0.1111"
            not in (prepared["packet_dir"] / reviewer / "reference.html").read_text()
        )
        key_orders.append([(row["case_id"], row["outfit_id"]) for row in rows])
    assert set(key_orders[0]) == set(key_orders[1])
    assert key_orders[0] != key_orders[1]


def test_full_agreement_emits_bound_request_and_preserves_private_profile(prepared):
    report = review(prepared)
    assert report["ready"]
    assert report["exact_agreement_rate"] == 1
    request = json.loads((prepared["root"] / "review/export-request.json").read_text())
    assert request["expected_catalog_sha256"] == prepared["packet"].catalog_sha256
    assert request["cases"][0]["profile"]["color_confidence"] == 0.1111
    assert request["annotation_packet_id"] == prepared["packet"].packet_id
    export_snapshot(
        prepared["root"] / "review/export-request.json",
        *prepared["inputs"][1:],
        prepared["root"] / "export",
    )


def test_disagreement_never_auto_averages_or_exports_partial_labels(prepared):
    add_disagreement(prepared)
    report = review(prepared)
    assert not report["ready"]
    assert report["disagreement_count"] == 1
    assert not (prepared["root"] / "review/export-request.json").exists()
    _, rows = read_csv(prepared["root"] / "review/resolution.csv")
    assert len(rows) == 1 and rows[0]["grade"] == ""
    _, differences = read_csv(prepared["root"] / "review/disagreements.csv")
    assert (differences[0]["grade_one"], differences[0]["grade_two"]) == ("2", "3")


def test_explicit_human_resolution_completes_request(prepared):
    key = add_disagreement(prepared)
    review(prepared, "pending")
    fields, rows = read_csv(prepared["root"] / "pending/resolution.csv")
    rows[0].update(grade="1", reason="Synthetic adjudication rationale")
    path = prepared["root"] / "resolved.csv"
    write_csv(path, fields, rows)
    report = review(prepared, "resolved", path)
    resolved = next(
        row for row in report["tasks"] if (row["case_id"], row["outfit_id"]) == key
    )
    assert report["ready"]
    assert resolved["consensus_grade"] == 1
    assert resolved["resolution_reason"] == "Synthetic adjudication rationale"


@pytest.mark.parametrize(
    "mutation",
    [
        "blank",
        "fraction",
        "missing",
        "duplicate",
        "unknown",
        "context",
        "reviewer",
        "packet",
    ],
)
def test_invalid_or_incomplete_ratings_are_rejected(prepared, mutation):
    fields, rows = read_csv(prepared["second"])
    if mutation == "blank":
        rows[0]["grade"] = ""
    elif mutation == "fraction":
        rows[0]["grade"] = "2.5"
    elif mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "unknown":
        rows[0]["outfit_id"] = "999"
    elif mutation == "context":
        rows[0]["season"] = "Changed"
    elif mutation == "reviewer":
        rows[0]["reviewer_id"] = "rater-a"
    else:
        rows[0]["packet_id"] = "other-packet"
    write_csv(prepared["second"], fields, rows)
    with pytest.raises(ValueError):
        review(prepared)
    assert not (prepared["root"] / "review").exists()


@pytest.mark.parametrize(
    "mutation", ["blank_reason", "missing", "duplicate", "wrong_packet", "unknown"]
)
def test_invalid_resolutions_are_rejected(prepared, mutation):
    add_disagreement(prepared)
    review(prepared, "pending")
    fields, rows = read_csv(prepared["root"] / "pending/resolution.csv")
    rows[0].update(grade="2", reason="Synthetic reason")
    if mutation == "blank_reason":
        rows[0]["reason"] = ""
    elif mutation == "missing":
        rows.clear()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "wrong_packet":
        rows[0]["packet_id"] = "other"
    else:
        rows[0]["outfit_id"] = "999"
    path = prepared["root"] / "bad-resolution.csv"
    write_csv(path, fields, rows)
    with pytest.raises(ValueError):
        review(prepared, "invalid", path)


def test_changed_catalog_cannot_reuse_old_annotations(prepared):
    import sqlite3

    review(prepared)
    with sqlite3.connect(prepared["inputs"][1]) as db:
        db.execute("UPDATE outfits SET name='Changed after review' WHERE id=1")
    with pytest.raises(ValueError, match="Catalog changed"):
        export_snapshot(
            prepared["root"] / "review/export-request.json",
            *prepared["inputs"][1:],
            prepared["root"] / "export",
        )
    assert not (prepared["root"] / "export").exists()


def test_reordered_completed_rows_are_joined_by_identity(prepared):
    fields, rows = read_csv(prepared["second"])
    write_csv(prepared["second"], fields, list(reversed(rows)))
    assert review(prepared)["ready"]


def test_reviewer_ids_and_output_directory_are_validated(prepared):
    with pytest.raises(ValueError):
        prepare_annotation_packet(
            prepared["plan"],
            prepared["inputs"][1],
            prepared["root"] / "other",
            reviewer_two="rater-a",
        )
    with pytest.raises(FileExistsError):
        prepare_annotation_packet(
            prepared["plan"], prepared["inputs"][1], prepared["packet_dir"]
        )


def test_html_and_csv_keep_untrusted_catalog_display_text_literal(prepared):
    import sqlite3

    name = '=FORMULA() <script>alert("fixture")</script>'
    with sqlite3.connect(prepared["inputs"][1]) as db:
        db.execute(
            "UPDATE outfits SET name=?, image_url='javascript:alert(1)' WHERE id=1",
            (name,),
        )
    output = prepared["root"] / "escaped-packet"
    prepare_annotation_packet(prepared["plan"], prepared["inputs"][1], output)
    html = (output / "rater-a/reference.html").read_text(encoding="utf-8")
    assert "<script>" not in html
    assert "javascript:" not in html
    _, rows = read_csv(output / "rater-a/ratings.csv")
    row = next(row for row in rows if row["outfit_id"] == "1")
    assert row["outfit_name"] == "'" + name
