"""Aggregate case evidence into reports without discarding failed cases."""

import csv
import json
from pathlib import Path

from app.data.body_shapes import MALE_BODY_SHAPES
from app.data.season_data import SEASON_DATA

from .metrics import classification_metrics, operational_metrics
from .schemas import PRIMARY_SEASONS, AnalysisDataset


def label_agreement_metrics(rows: list[dict]) -> dict:
    """Shared aggregation for a single run and a paired comparison."""
    quality = {}
    for feature in ("color", "body"):
        selected = [row for row in rows if row["feature"] == feature]
        if not selected:
            continue
        dimensions = (
            {"primary_season": PRIMARY_SEASONS, "sub_season": tuple(SEASON_DATA)}
            if feature == "color"
            else {"shape": MALE_BODY_SHAPES}
        )
        quality[feature] = {}
        for field, labels in dimensions.items():
            labeled = [
                row for row in selected if row["reference"].get(field) is not None
            ]
            quality[feature][field] = classification_metrics(
                [row["reference"][field] for row in labeled],
                [
                    row["final_result"].get(field) if row["final_result"] else None
                    for row in labeled
                ],
                labels=labels,
            )
    return quality


def build_report(dataset: AnalysisDataset, rows: list[dict], metadata: dict) -> dict:
    return {
        "metadata": metadata,
        "interpretation": (
            "Synthetic software smoke check; these are not AI quality or performance measurements."
            if dataset.synthetic
            else "Pilot agreement with reference annotations; inspect support and failures before drawing conclusions."
        ),
        "operations": operational_metrics(rows),
        "quality": label_agreement_metrics(rows),
    }


def write_case_summary(output_dir: Path, rows: list[dict]) -> None:
    """A flat inspection view; JSONL remains the complete evidence source."""
    fields = [
        "case_id",
        "feature",
        "status",
        "synthetic",
        "error_type",
        "error_code",
        "reference_primary_season",
        "predicted_primary_season",
        "reference_sub_season",
        "predicted_sub_season",
        "reference_shape",
        "predicted_shape",
        "logical_model_calls",
        "provider_attempts",
        "latency_ms",
    ]
    with (output_dir / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            error = row["error"] or {}
            final = row["final_result"] or {}
            calls = row["trace"]["calls"]
            summary = {
                name: row[name]
                for name in ("case_id", "feature", "status", "synthetic", "latency_ms")
            }
            summary.update(
                error_type=error.get("type"),
                error_code=error.get("code"),
                logical_model_calls=len(calls),
                provider_attempts=sum(len(call["attempts"]) for call in calls),
            )
            for field in ("primary_season", "sub_season", "shape"):
                summary[f"reference_{field}"] = row["reference"].get(field)
                summary[f"predicted_{field}"] = final.get(field)
            writer.writerow(summary)


def write_report(output_dir: Path, report: dict) -> None:
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    operations = report["operations"]
    lines = [
        "# Analysis evaluation",
        "",
        report["interpretation"],
        "",
        f"- Dataset: {report['metadata']['dataset_id']} ({report['metadata']['dataset_version']})",
        f"- Split: {report['metadata']['split']}",
        f"- Provider: {report['metadata']['provider']}",
        f"- Cases: {operations['case_count']}",
        f"- Final success rate: {operations['final_success_rate']:.3f}",
        f"- First-pass schema validity: {operations['first_pass_schema_valid_rate']:.3f}",
        f"- Repair rate: {operations['repair_rate']:.3f}",
        "",
        "## Label agreement",
        "",
        "| Feature / field | Cases | Coverage | Accuracy | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for feature, dimensions in report["quality"].items():
        for field, metrics in dimensions.items():
            values = [metrics[key] for key in ("coverage", "accuracy", "macro_f1")]
            cells = ["N/A" if value is None else f"{value:.3f}" for value in values]
            lines.append(
                f"| {feature} / {field} | {metrics['case_count']} | {' | '.join(cells)} |"
            )
    lines += [
        "",
        "Macro F1 includes every declared taxonomy class, including absent classes.",
        "Failures count as misses. Sub-season metrics include only cases with sub-season references.",
        "See report.json for support/confusion matrices and cases.jsonl for raw, repaired, and final evidence.",
        "Latency includes retries, backoff, image processing, and failures; small-sample p95 is descriptive.",
        "Token counts include observed usage only. Missing provider usage is not zero cost.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
