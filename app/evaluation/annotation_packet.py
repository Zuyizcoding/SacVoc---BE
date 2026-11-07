"""Prepare two blinded reviewer folders from a read-only catalog.

Pipeline: validate profiles -> filter the existing catalog -> freeze task keys ->
shuffle each reviewer's display order -> write blank CSV and an HTML reference.
No ranker runs here. The private packet stays with the coordinator; distribute
only the individual reviewer folder to preserve procedural blinding.
"""

import csv
import random
import re
from html import escape
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from app.services.analysis.tracing import content_hash

from .annotation_schema import AnnotationPacket, AnnotationPlan, AnnotationTask
from .retrieval import candidates_for_profile
from .sqlite_catalog import catalog_fingerprint, read_catalog

DISPLAY_FIELDS = tuple(AnnotationTask.model_fields)
RATING_FIELDS = ("packet_id", "reviewer_id", *DISPLAY_FIELDS, "grade", "notes")


def csv_text(value: str) -> str:
    """Keep display text literal when a CSV is opened in a spreadsheet."""
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def task_csv_fields(task: AnnotationTask) -> dict[str, str]:
    return {name: csv_text(value) for name, value in task.model_dump().items()}


def _review_html(tasks: list[AnnotationTask]) -> str:
    cards = []
    for task in tasks:
        image = urlparse(task.image_url)
        link = (
            f'<a href="{escape(task.image_url, quote=True)}" rel="noreferrer">Open image</a>'
            if image.scheme in {"https", "http"}
            else "Image link unavailable"
        )
        cards.append(
            "<article><h2>"
            + escape(f"{task.case_id} / outfit {task.outfit_id}")
            + "</h2>"
            "<p>" + escape(f"{task.season} / {task.shape} / {task.occasion}") + "</p>"
            "<p>"
            + escape(f"{task.outfit_name} ({task.category})")
            + "</p><p>"
            + link
            + "</p></article>"
        )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Relevance annotation</title>'
        "<style>body{font:16px system-ui;max-width:900px;margin:32px auto;padding:0 16px}"
        "article{border:1px solid #ccc;border-radius:8px;padding:16px;margin:12px 0}</style>"
        "<h1>Independent relevance annotation</h1>"
        "<p>Fill ratings.csv: 0 irrelevant, 1 weakly relevant, 2 relevant, 3 strongly relevant.</p>"
        "<p>Use the fixed profile context. Complete your ratings independently before discussing them. "
        "Image links load only when opened. Confidence, model scores and bias tags are omitted.</p>"
        + "".join(cards)
        + "</html>"
    )


def prepare_annotation_packet(
    plan_path: Path,
    database_path: Path,
    output_dir: Path,
    *,
    reviewer_one: str = "rater-a",
    reviewer_two: str = "rater-b",
    seed: int = 42,
) -> AnnotationPacket:
    reviewers = [reviewer_one, reviewer_two]
    if len(set(reviewers)) != 2 or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", reviewer)
        for reviewer in reviewers
    ):
        raise ValueError("Use two distinct alphanumeric reviewer IDs")
    plan = AnnotationPlan.model_validate_json(plan_path.read_text(encoding="utf-8-sig"))
    catalog = read_catalog(database_path)
    if plan.synthetic != (catalog.embedding.backend == "synthetic"):
        raise ValueError("Plan and catalog synthetic declarations disagree")
    tasks = []
    for case in plan.cases:
        pool = candidates_for_profile(catalog.outfits, case.profile)
        if not pool:
            raise ValueError(f"No eligible candidates for {case.case_id}")
        for outfit in pool:
            tasks.append(
                AnnotationTask(
                    case_id=case.case_id,
                    outfit_id=str(outfit.id),
                    season=case.profile.color_sub_season,
                    shape=case.profile.body_shape,
                    occasion=case.profile.occasion,
                    outfit_name=outfit.name,
                    category=outfit.category,
                    image_url=outfit.image_url,
                )
            )
    packet = AnnotationPacket(
        packet_id=str(uuid4()),
        plan=plan,
        catalog_sha256=catalog_fingerprint(catalog.outfits),
        reviewers=reviewers,
        tasks=tasks,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    private = output_dir / "private"
    private.mkdir()
    (private / "packet.json").write_text(
        packet.model_dump_json(indent=2), encoding="utf-8"
    )
    for reviewer in reviewers:
        directory = output_dir / reviewer
        directory.mkdir()
        ordered = tasks.copy()
        random.Random(int(content_hash(f"{seed}:{reviewer}"), 16)).shuffle(ordered)
        with (directory / "ratings.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=RATING_FIELDS)
            writer.writeheader()
            for task in ordered:
                writer.writerow(
                    {
                        "packet_id": packet.packet_id,
                        "reviewer_id": reviewer,
                        **task_csv_fields(task),
                        "grade": "",
                        "notes": "",
                    }
                )
        (directory / "reference.html").write_text(
            _review_html(ordered), encoding="utf-8"
        )
    return packet
