"""Strict completed-CSV reader: only grade and notes are editable.

Task identity is the (case_id, outfit_id) pair, never the row position. A reviewer
may reorder rows, but changing context, packet IDs or reviewer assignments makes
the file invalid. Blank/missing grades are incomplete work, not zero relevance.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from .annotation_packet import RATING_FIELDS, task_csv_fields
from .annotation_schema import AnnotationPacket, TaskKey


@dataclass(frozen=True)
class Rating:
    grade: int
    notes: str


def read_csv_rows(path: Path, fields) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if (
            reader.fieldnames is None
            or len(reader.fieldnames) != len(set(reader.fieldnames))
            or set(reader.fieldnames) != set(fields)
        ):
            raise ValueError(f"Unexpected or duplicate CSV columns in {path.name}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV row length does not match the header")
    return rows


def parse_grade(text: str) -> int:
    if text.strip() not in {"0", "1", "2", "3"}:
        raise ValueError(
            "Every grade must be an integer 0..3; blank grades are incomplete"
        )
    return int(text.strip())


def read_ratings(
    path: Path, packet: AnnotationPacket, reviewer: str
) -> dict[TaskKey, Rating]:
    expected = {task.key: task for task in packet.tasks}
    ratings = {}
    for row in read_csv_rows(path, RATING_FIELDS):
        key = row["case_id"], row["outfit_id"]
        if key not in expected or key in ratings:
            raise ValueError(f"Unknown/duplicate annotation task: {key}")
        if row["packet_id"] != packet.packet_id or row["reviewer_id"] != reviewer:
            raise ValueError("Packet/reviewer ID does not match the assigned file")
        fields = task_csv_fields(expected[key])
        if any(row[name] != value for name, value in fields.items()):
            raise ValueError(f"Read-only task context was edited: {key}")
        ratings[key] = Rating(parse_grade(row["grade"]), row["notes"])
    if set(ratings) != set(expected):
        raise ValueError("Every assigned task needs exactly one completed rating")
    return ratings
