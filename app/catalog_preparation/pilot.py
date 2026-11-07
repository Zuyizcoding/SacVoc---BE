"""Prepare a small, blinded, offline F3 annotation pilot with blank ratings.

Intuition: before collecting judgments, freeze what each question means. This
pilot uses hypothetical profile scenarios, not measured subjects or AI-derived
F1/F2 labels. Twenty scenarios cover each body-shape/occasion pair exactly once;
all twelve color seasons appear. All cases belong to dev for rubric calibration.

Schema: existing AnnotationPlan/Packet/CSV contracts are reused. synthetic=false
describes the real catalog embeddings, NOT the existence of real human subjects.
Pipeline: verify bundle/audit -> create scenario plan -> existing blinded packet
builder -> copy local images -> render only permitted display fields -> validate
blank ratings and independent task orders -> publish coordinator summary.
"""

import csv
import json
import zipfile
from collections import Counter
from html import escape
from pathlib import Path

from app.evaluation.annotation_packet import (
    RATING_FIELDS,
    prepare_annotation_packet,
    task_csv_fields,
)
from app.evaluation.annotation_schema import (
    AnnotationCase,
    AnnotationPlan,
    AnnotationTask,
)
from app.evaluation.retrieval_schema import RetrievalProfile
from app.services.recommendation.query_builder import (
    BODY_KEYWORDS,
    OCCASION_KEYWORDS,
    SEASON_KEYWORDS,
)

from .encoder import file_sha256
from .integrity import verify_manifest
from .review_assets import copy_local_images, image_html, review_document
from .workflow import write_json

PROTOCOL_ID = "f3-hypothetical-scenario-pilot-v1"
REVIEW_GUIDE = """# Independent F3 relevance pilot — reviewer guide

## Intuition
Judge how useful this clothing item would be as part of an outfit for the
displayed color season, body shape and occasion. These are hypothetical scenarios,
not actual people's measured profiles. Your ratings describe this fixed question.

## What you receive
Your folder contains ratings.csv, reference.html, images/ and this guide.
Open reference.html locally; images work without the network. Match questions
by case_id and outfit_id, never by spreadsheet row number.

## Rubric v1 (draft for calibration)
0 — Irrelevant: clearly unsuitable for the stated occasion/context, or not useful
    as a clothing component for the scenario.
1 — Weakly relevant: some plausible use, but substantial visible/contextual
    limitations prevent recommending it confidently.
2 — Relevant: a reasonable component for the scenario with no major evident
    conflict; not an especially strong match.
3 — Strongly relevant: clear positive reasons for this item in the stated context,
    supported by what the image and product description actually show.

Consider occasion, visible color and item silhouette/fit. Do not infer a complete
outfit or a wearer's proportions from a single isolated product. For shoes or
accessories, body-shape evidence may be limited; explain uncertainty in notes.
Use an agreed neutral reference for the named season/shape rather than inventing
rules during rating. If the rubric is unclear, ask the coordinator before starting.

## Pipeline
1. Coordinator reviews catalog issues and approves/calibrates this draft protocol.
2. Work independently before discussing any items with the other reviewer.
3. Edit ONLY grade and notes. Grades are integers 0, 1, 2 or 3.
4. If an image/context cannot be judged, leave grade blank and explain why in
   notes; ask the coordinator to resolve the task/protocol before final submission.
   Blank is incomplete, never an implicit zero. Do not invent a grade.
5. Save UTF-8 CSV. Keep packet/reviewer IDs and display fields unchanged.
6. Give your completed folder/file to the coordinator. Do not view model scores,
   rankings, bias tags, the coordinator gallery or the other reviewer's answers.

## Limits
All twenty scenarios are dev cases for pilot calibration. This is not a held-out
test benchmark, a population sample or measured F1/F2 accuracy. The coordinator
must document disagreements and adjudication rather than silently average grades.
"""


def create_pilot_plan() -> AnnotationPlan:
    """Deterministic coverage, not a statistically representative sample."""
    seasons = list(SEASON_KEYWORDS)
    cases = []
    for shape in BODY_KEYWORDS:
        for occasion in OCCASION_KEYWORDS:
            index = len(cases)
            cases.append(
                AnnotationCase(
                    case_id=f"scenario-{index + 1:03}",
                    subject_id=f"hypothetical-profile-{index + 1:03}",
                    split="dev",
                    profile=RetrievalProfile(
                        color_sub_season=seasons[index % len(seasons)],
                        body_shape=shape,
                        occasion=occasion,
                    ),
                )
            )
    return AnnotationPlan(
        dataset_id="sacvoc-f3-hypothetical-pilot",
        version="1",
        synthetic=False,
        annotation_protocol=(
            f"{PROTOCOL_ID}; hypothetical profiles, real catalog embeddings; "
            "all dev; draft rubric requires coordinator approval and independent human ratings"
        ),
        cases=cases,
    )


def render_reviewer_reference(
    tasks: list[AnnotationTask], images: dict[str, str]
) -> str:
    """Render the allowlisted task schema only: no ranking metadata or scores."""
    cards = []
    for task in tasks:
        cards.append(
            f"<article><h2>{escape(task.case_id)} / item {escape(task.outfit_id)}</h2>"
            f"<p>{escape(task.season)} · {escape(task.shape)} · {escape(task.occasion)}</p>"
            + image_html(images.get(task.image_url), task.outfit_name)
            + f"<p>{escape(task.outfit_name)} ({escape(task.category)})</p></article>"
        )
    return review_document(
        "Independent F3 relevance pilot",
        "Draft pilot: begin after coordinator approval. Use README.md and fill only "
        "grade/notes in ratings.csv. Scenario profiles are hypothetical; work independently.",
        '<div class="grid">' + "".join(cards) + "</div>",
    )


def validate_blank_forms(output: Path, packet) -> dict:
    """Check task/context completeness and blinding before any human work."""
    expected = {task.key for task in packet.tasks}
    tasks = {task.key: task for task in packet.tasks}
    orders = []
    for reviewer in packet.reviewers:
        with (output / reviewer / "ratings.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != RATING_FIELDS:
                raise ValueError("Reviewer CSV leaked or lost fields")
            rows = list(reader)
        keys = [(row["case_id"], row["outfit_id"]) for row in rows]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("Reviewer task set is incomplete")
        if any(row["grade"] or row["notes"] for row in rows):
            raise ValueError("Prepared reviewer grades and notes must remain blank")
        for row, key in zip(rows, keys):
            if (
                row["packet_id"] != packet.packet_id
                or row["reviewer_id"] != reviewer
                or any(
                    row[name] != value
                    for name, value in task_csv_fields(tasks[key]).items()
                )
            ):
                raise ValueError(
                    "Prepared reviewer context does not match the private packet"
                )
        orders.append(keys)
    if orders[0] == orders[1]:
        raise ValueError("Reviewer task order must be independently shuffled")
    return {"blank_grades": True, "same_task_sets": True, "different_orders": True}


def prepare_relevance_pilot(bundle: Path, audit_path: Path, output: Path) -> dict:
    integrity = verify_manifest(bundle)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["integrity"]["manifest_sha256"] != integrity["manifest_sha256"]:
        raise ValueError("Catalog audit belongs to a different bundle")
    if audit["severity_counts"].get("error", 0):
        raise ValueError(
            "Resolve structural catalog errors before preparing annotation"
        )
    output.mkdir(parents=True, exist_ok=False)
    plan = create_pilot_plan()
    plan_path = output / "annotation-plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    packet_dir = output / "packet"
    packet = prepare_annotation_packet(plan_path, bundle / "catalog.db", packet_dir)
    task_by_key = {task.key: task for task in packet.tasks}
    for reviewer in packet.reviewers:
        directory = packet_dir / reviewer
        images = copy_local_images(bundle, directory)
        with (directory / "ratings.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            ordered = [
                task_by_key[(row["case_id"], row["outfit_id"])]
                for row in csv.DictReader(stream)
            ]
        (directory / "reference.html").write_text(
            render_reviewer_reference(ordered, images), encoding="utf-8"
        )
        (directory / "README.md").write_text(REVIEW_GUIDE, encoding="utf-8")
    form_check = validate_blank_forms(packet_dir, packet)
    archives = {}
    for reviewer in packet.reviewers:
        archive = output / f"{reviewer}.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as target:
            for path in sorted((packet_dir / reviewer).rglob("*")):
                if path.is_file():
                    target.write(path, path.relative_to(packet_dir).as_posix())
        archives[reviewer] = {"filename": archive.name, "sha256": file_sha256(archive)}
    report = {
        "status": "draft_awaiting_coordinator_and_independent_ratings",
        "profile_origin": "hypothetical scenarios; no observed human subjects",
        "scenario_count": len(plan.cases),
        "split_counts": dict(Counter(c.split for c in plan.cases)),
        "coverage": {
            field: dict(Counter(getattr(c.profile, field) for c in plan.cases))
            for field in ("color_sub_season", "body_shape", "occasion")
        },
        "tasks_per_reviewer": len(packet.tasks),
        "reviewers": packet.reviewers,
        "packet_id": packet.packet_id,
        "catalog_sha256": packet.catalog_sha256,
        "bundle_manifest_sha256": integrity["manifest_sha256"],
        "audit_sha256": file_sha256(audit_path),
        "catalog_issue_counts": audit["issue_counts"],
        "form_check": form_check,
        "reviewer_archives": archives,
        "ratings_completed": 0,
        "interpretation": "Ready-to-review draft materials, not measured model quality.",
    }
    write_json(output / "summary.json", report)
    (output / "COORDINATOR.md").write_text(
        "# Coordinator handoff\n\n"
        "Review the separate catalog audit first, especially shared-image product "
        "conflicts and missing body-shape metadata. Confirm/correct source data "
        "and agree on the rubric before distributing any tasks. If the catalog "
        "changes, regenerate the bundle/audit/packet; do not edit frozen CSV context.\n\n"
        "These 20 profiles are hypothetical and all dev. Share ONLY one assigned "
        "reviewer folder per person. Never share private/packet.json, the coordinator "
        "gallery or model rankings. Each folder contains its own offline image files.\n\n"
        "After independent completion, use scripts.review_relevance_annotations "
        "to validate ratings and obtain an explicit disagreement/adjudication list. "
        "Only completed and adjudicated labels can enter snapshot export/ablation.\n",
        encoding="utf-8",
    )
    verify_manifest(bundle)
    return report
