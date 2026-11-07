"""Dataset preflight: validate schema -> inspect inputs -> summarize readiness.

Intuition: a measured score is untrustworthy if the test split repeats training
images or one person's many photos dominate a class. Count cases and distinct
subjects separately, and detect exact cross-split duplicates even when subject
IDs differ. Preflight diagnoses data; it never changes labels, splits, or images.
"""

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from app.services.analysis.tracing import content_hash

from .image_inspection import inspect_image_path
from .retrieval import candidate_pool, eligible_judgments
from .retrieval_schema import RetrievalSnapshot
from .schemas import AnalysisCase, AnalysisDataset


def _schema_error_report(exc: ValidationError) -> dict:
    # Pydantic's full error includes input values. Keep only location/category/
    # message so reports do not copy private input paths or entire manifests.
    return {
        "ready": False,
        "inspection_complete": False,
        "errors": [
            {
                "code": "SCHEMA_INVALID",
                "details": [
                    {
                        "location": list(error["loc"]),
                        "type": error["type"],
                        "message": error["msg"],
                    }
                    for error in exc.errors(include_input=False, include_context=False)
                ],
            }
        ],
        "warnings": [],
    }


def _label_support(cases: Sequence[AnalysisCase]) -> dict:
    support = {}
    for split in ("dev", "test"):
        selected = [case for case in cases if case.split == split]
        labels: dict = {}
        for case in selected:
            for field, label in case.reference.model_dump(exclude_none=True).items():
                key = f"{case.feature}.{field}"
                entry = labels.setdefault(key, {}).setdefault(
                    label, {"cases": 0, "subjects": set()}
                )
                entry["cases"] += 1
                entry["subjects"].add(case.subject_id)
        support[split] = {
            "case_count": len(selected),
            "subject_count": len({case.subject_id for case in selected}),
            "labels": {
                key: {
                    label: {"cases": entry["cases"], "subjects": len(entry["subjects"])}
                    for label, entry in entries.items()
                }
                for key, entries in labels.items()
            },
        }
    return support


def _duplicate_groups(rows: list[dict]) -> list[dict]:
    groups = []
    for fingerprint in ("encoded_sha256", "rgb_sha256"):
        by_hash = defaultdict(list)
        for row in rows:
            value = row["image"][fingerprint]
            if value is not None:
                by_hash[value].append(row)
        for matches in by_hash.values():
            if len(matches) < 2:
                continue
            splits = sorted({row["split"] for row in matches})
            groups.append(
                {
                    "fingerprint": fingerprint,
                    "case_ids": [row["case_id"] for row in matches],
                    "subject_ids": sorted({row["subject_id"] for row in matches}),
                    "splits": splits,
                    "cross_split": len(splits) > 1,
                }
            )
    return groups


def preflight_analysis(manifest_path: Path) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    try:
        dataset = AnalysisDataset.model_validate_json(
            manifest_bytes.decode("utf-8-sig")
        )
    except ValidationError as exc:
        return {"kind": "analysis", **_schema_error_report(exc)}
    rows = []
    errors = []
    warnings = []
    if dataset.synthetic:
        warnings.append(
            {
                "code": "SYNTHETIC_IMAGES_NOT_APPLICABLE",
                "message": "Synthetic analysis uses generated runner images; no image paths are inspected.",
            }
        )
    else:
        for case in dataset.cases:
            image = inspect_image_path(manifest_path.parent / case.image_path)
            rows.append(
                {
                    "case_id": case.case_id,
                    "subject_id": case.subject_id,
                    "feature": case.feature,
                    "split": case.split,
                    "image": image.model_dump(),
                }
            )
            errors.extend(
                {"code": code, "case_id": case.case_id} for code in image.issues
            )
    duplicates = _duplicate_groups(rows)
    for group in duplicates:
        issue = {
            "code": "CROSS_SPLIT_DUPLICATE"
            if group["cross_split"]
            else "WITHIN_SPLIT_DUPLICATE",
            **group,
        }
        (errors if group["cross_split"] else warnings).append(issue)
    return {
        "kind": "analysis",
        "manifest_sha256": content_hash(manifest_bytes),
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.version,
        "synthetic": dataset.synthetic,
        "ready": not errors,
        "inspection_complete": True,
        "image_inspection": "not_applicable" if dataset.synthetic else "completed",
        "support": _label_support(dataset.cases),
        "images": rows,
        "duplicate_groups": duplicates,
        "errors": errors,
        "warnings": warnings,
    }


def preflight_retrieval(snapshot_path: Path) -> dict:
    snapshot_bytes = snapshot_path.read_bytes()
    try:
        snapshot = RetrievalSnapshot.model_validate_json(
            snapshot_bytes.decode("utf-8-sig")
        )
    except ValidationError as exc:
        return {"kind": "retrieval", **_schema_error_report(exc)}
    errors = []
    rows = []
    for case in snapshot.cases:
        pool = candidate_pool(snapshot, case)
        try:
            judgments = eligible_judgments(pool, case)
        except ValueError as exc:
            errors.append(
                {
                    "code": "INVALID_CANDIDATE_POOL",
                    "case_id": case.case_id,
                    "message": str(exc),
                }
            )
            judgments = {}
        rows.append(
            {
                "case_id": case.case_id,
                "subject_id": case.subject_id,
                "split": case.split,
                "candidate_count": len(pool),
                "judged_count": len(judgments),
                "positive_count": sum(grade > 0 for grade in judgments.values()),
            }
        )
    return {
        "kind": "retrieval",
        "snapshot_sha256": content_hash(snapshot_bytes),
        "dataset_id": snapshot.dataset_id,
        "dataset_version": snapshot.version,
        "synthetic": snapshot.synthetic,
        "ready": not errors,
        "inspection_complete": True,
        "case_count": len(rows),
        "subject_count": len({case.subject_id for case in snapshot.cases}),
        "cases": rows,
        "errors": errors,
        "warnings": [],
    }


def write_preflight_report(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Dataset preflight",
        "",
        f"- Kind: {report['kind']}",
        f"- Ready: {report['ready']}",
        f"- Inspection complete: {report['inspection_complete']}",
        f"- Errors: {len(report['errors'])}",
        f"- Warnings: {len(report['warnings'])}",
        "",
        "Readiness checks structure, exact duplication and available inputs.",
        "It does not verify consent, human annotation independence, or AI accuracy.",
        "",
        "See report.json for case-level issues, support and duplicate groups.",
        "",
    ]
    for level in ("errors", "warnings"):
        lines += [f"## {level.title()}", ""]
        lines += [
            f"- {issue['code']}: {issue.get('case_id', issue.get('case_ids', 'dataset'))}"
            for issue in report[level]
        ] or ["None."]
        lines.append("")
    if "support" in report:
        lines += [
            "## Label support",
            "",
            "| Split | Field / label | Cases | Subjects |",
            "| --- | --- | ---: | ---: |",
        ]
        for split, summary in report["support"].items():
            lines.append(
                f"| {split} | All cases | {summary['case_count']} | {summary['subject_count']} |"
            )
            for field, labels in summary["labels"].items():
                for label, support in labels.items():
                    lines.append(
                        f"| {split} | {field} / {label} | {support['cases']} | {support['subjects']} |"
                    )
        lines.append("")
    if "cases" in report:
        lines += [
            "## Candidate pools",
            "",
            "| Case | Eligible | Judged | Positive |",
            "| --- | ---: | ---: | ---: |",
        ]
        for case in report["cases"]:
            lines.append(
                f"| {case['case_id']} | {case['candidate_count']} | {case['judged_count']} | {case['positive_count']} |"
            )
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
