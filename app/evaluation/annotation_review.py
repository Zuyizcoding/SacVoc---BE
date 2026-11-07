"""Compare independent grades, then require explicit adjudication for differences.

Agreement copies the common grade; disagreement never becomes an average or an
AI-generated answer. A human resolution needs both a grade and a written reason.
A final exporter request is written only when every task has a consensus grade.
"""

import csv
import json
from pathlib import Path

from app.services.analysis.tracing import content_hash

from .annotation_csv import parse_grade, read_csv_rows, read_ratings
from .annotation_packet import csv_text
from .annotation_schema import AnnotationPacket
from .snapshot_export import ExportCase, ExportRequest

RESOLUTION_FIELDS = ("packet_id", "case_id", "outfit_id", "grade", "reason")


def review_annotations(
    packet_path: Path,
    first_path: Path,
    second_path: Path,
    output_dir: Path,
    *,
    resolution_path: Path | None = None,
) -> dict:
    packet_bytes = packet_path.read_bytes()
    packet = AnnotationPacket.model_validate_json(packet_bytes.decode("utf-8-sig"))
    first = read_ratings(first_path, packet, packet.reviewers[0])
    second = read_ratings(second_path, packet, packet.reviewers[1])
    disputed = {key for key in first if first[key].grade != second[key].grade}
    resolutions = {}
    if resolution_path is not None:
        for row in read_csv_rows(resolution_path, RESOLUTION_FIELDS):
            key = row["case_id"], row["outfit_id"]
            if (
                row["packet_id"] != packet.packet_id
                or key not in disputed
                or key in resolutions
            ):
                raise ValueError("Unknown/duplicate resolution or wrong packet ID")
            if not row["reason"].strip():
                raise ValueError("Each human resolution needs a non-empty reason")
            resolutions[key] = (parse_grade(row["grade"]), row["reason"].strip())
        if set(resolutions) != disputed:
            raise ValueError("Resolve every disagreement exactly once")
    rows = []
    for task in packet.tasks:
        key = task.key
        agree = key not in disputed
        grade = first[key].grade if agree else resolutions.get(key, (None, ""))[0]
        reason = (
            "independent_agreement" if agree else resolutions.get(key, (None, ""))[1]
        )
        rows.append(
            {
                "case_id": task.case_id,
                "outfit_id": task.outfit_id,
                "grade_one": first[key].grade,
                "grade_two": second[key].grade,
                "notes_one": first[key].notes,
                "notes_two": second[key].notes,
                "agreed": agree,
                "consensus_grade": grade,
                "resolution_reason": reason,
            }
        )
    ready = all(row["consensus_grade"] is not None for row in rows)
    report = {
        "packet_id": packet.packet_id,
        "synthetic": packet.plan.synthetic,
        "ready": ready,
        "task_count": len(rows),
        "agreement_count": len(rows) - len(disputed),
        "disagreement_count": len(disputed),
        "exact_agreement_rate": (len(rows) - len(disputed)) / len(rows),
        "packet_sha256": content_hash(packet_bytes),
        "reviewer_file_sha256": [
            content_hash(path.read_bytes()) for path in (first_path, second_path)
        ],
        "resolution_sha256": content_hash(resolution_path.read_bytes())
        if resolution_path
        else None,
        "reviewers": packet.reviewers,
        "tasks": rows,
        "interpretation": "Agreement between submitted files does not prove fashion accuracy or reviewer independence.",
    }
    request = None
    if ready:
        request = ExportRequest(
            **packet.plan.model_dump(exclude={"cases"}),
            cases=[
                ExportCase(
                    **case.model_dump(),
                    relevance={
                        row["outfit_id"]: row["consensus_grade"]
                        for row in rows
                        if row["case_id"] == case.case_id
                    },
                )
                for case in packet.plan.cases
            ],
            expected_catalog_sha256=packet.catalog_sha256,
            annotation_packet_id=packet.packet_id,
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "review.md").write_text(
        "# Annotation review\n\n"
        f"- Ready for export: {ready}\n- Tasks: {len(rows)}\n"
        f"- Agreements: {report['agreement_count']}\n- Disagreements: {len(disputed)}\n\n"
        "Review resolution.csv for disputed pairs. No grades were averaged or inferred.\n"
        "See review.json for both independent grades, notes and provenance.\n",
        encoding="utf-8",
    )
    with (output_dir / "disagreements.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fields = (
            "case_id",
            "outfit_id",
            "grade_one",
            "grade_two",
            "notes_one",
            "notes_two",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if not row["agreed"]:
                writer.writerow(
                    {
                        field: csv_text(str(row[field]))
                        if field.startswith("notes_")
                        else row[field]
                        for field in fields
                    }
                )
    with (output_dir / "resolution.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=RESOLUTION_FIELDS)
        writer.writeheader()
        for row in rows:
            if not row["agreed"]:
                writer.writerow(
                    {
                        "packet_id": packet.packet_id,
                        "case_id": row["case_id"],
                        "outfit_id": row["outfit_id"],
                        "grade": row["consensus_grade"],
                        "reason": csv_text(row["resolution_reason"]),
                    }
                )
    if request is not None:
        (output_dir / "export-request.json").write_text(
            request.model_dump_json(indent=2), encoding="utf-8"
        )
    return report
