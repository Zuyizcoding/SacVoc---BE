"""Synthetic demonstration of packet -> ratings -> adjudication -> export.

Only this explicitly synthetic example fills grades automatically from fixtures.
The actual annotation tools always write blank grades and require human files.
"""

import argparse
import csv
import json
from pathlib import Path

from app.evaluation.annotation_packet import prepare_annotation_packet
from app.evaluation.annotation_review import review_annotations
from app.evaluation.preflight import preflight_retrieval
from app.evaluation.snapshot_export import export_snapshot
from app.evaluation.synthetic_catalog import write_synthetic_catalog_sources


def _write_fixture_ratings(
    source: Path, target: Path, grades: dict, *, disputed=None
) -> None:
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        key = row["case_id"], row["outfit_id"]
        grade = grades[key]
        row["grade"] = (grade + 1) % 4 if key == disputed else grade
        row["notes"] = "Synthetic fixture rating, not an independent human judgment"
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_smoke(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    request_path, database, vectors, metadata = write_synthetic_catalog_sources(
        output_dir
    )
    original_database = database.read_bytes()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not request["synthetic"]:
        raise ValueError(
            "Automatic fixture grades are allowed only for synthetic smoke data"
        )
    plan = {
        key: request[key]
        for key in ("dataset_id", "version", "synthetic", "annotation_protocol")
    }
    plan["cases"] = [
        {key: value for key, value in case.items() if key != "relevance"}
        for case in request["cases"]
    ]
    plan_path = output_dir / "annotation-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    packet_dir = output_dir / "packet"
    packet = prepare_annotation_packet(plan_path, database, packet_dir)
    grades = {
        (case["case_id"], outfit_id): grade
        for case in request["cases"]
        for outfit_id, grade in case["relevance"].items()
    }
    disputed = packet.tasks[0].key
    first_path, second_path = (
        output_dir / "synthetic-a.csv",
        output_dir / "synthetic-b.csv",
    )
    _write_fixture_ratings(packet_dir / "rater-a/ratings.csv", first_path, grades)
    _write_fixture_ratings(
        packet_dir / "rater-b/ratings.csv", second_path, grades, disputed=disputed
    )
    packet_path = packet_dir / "private/packet.json"
    pending = review_annotations(
        packet_path, first_path, second_path, output_dir / "pending"
    )
    if pending["ready"] or pending["disagreement_count"] != 1:
        raise RuntimeError(
            "One intentional fixture disagreement must remain unresolved"
        )
    resolution_path = output_dir / "pending/resolution.csv"
    with resolution_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0]["grade"] = grades[disputed]
    rows[0]["reason"] = "Synthetic adjudication fixture only"
    resolutions = output_dir / "synthetic-resolution.csv"
    with resolutions.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    complete = review_annotations(
        packet_path,
        first_path,
        second_path,
        output_dir / "complete",
        resolution_path=resolutions,
    )
    if not complete["ready"]:
        raise RuntimeError("Synthetic resolution did not complete consensus")
    provenance = export_snapshot(
        output_dir / "complete/export-request.json",
        database,
        vectors,
        metadata,
        output_dir / "export",
    )
    if not preflight_retrieval(output_dir / "export/snapshot.json")["ready"]:
        raise RuntimeError("Consensus request must produce a valid retrieval snapshot")
    if database.read_bytes() != original_database:
        raise RuntimeError("Annotation/export changed the source catalog")
    (output_dir / "smoke-summary.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "tasks_per_reviewer": len(packet.tasks),
                "intentional_disagreements": 1,
                "consensus_ready": True,
                "exported_cases": provenance["case_count"],
                "database_unchanged": True,
                "interpretation": "Software workflow checks only; synthetic agreement is not human annotation evidence.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_smoke(args.output_dir)
    print(f"Synthetic annotation checks passed: {args.output_dir}")


if __name__ == "__main__":
    main()
