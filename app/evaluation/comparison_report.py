"""Write a reviewable paired report; keep individual failures visible."""

import csv
import json
from pathlib import Path


def write_comparison_report(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Paired analysis comparison",
        "",
        report["interpretation"],
        "",
        "All deltas are candidate minus baseline. Errors are retained.",
        "",
        f"- Cases: {report['paired']['case_count']}",
        f"- Status transitions: {report['paired']['status_transitions']}",
        f"- Complete usage pairs: {report['paired']['complete_usage_pair_count']}",
        "",
        "## Label changes",
        "",
        "| Field | Improved | Regressed | Unchanged correct | Unchanged incorrect |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for field, counts in report["paired"]["label_changes"].items():
        cells = [
            str(counts.get(name, 0))
            for name in (
                "improved",
                "regressed",
                "unchanged_correct",
                "unchanged_incorrect",
            )
        ]
        lines.append(f"| {field} | {' | '.join(cells)} |")
    lines += ["", "## Configuration differences", ""]
    lines += [
        f"- {change['field']}: see comparison.json for both values."
        for change in report["changed_settings"]
    ] or ["No recorded configuration differences."]
    if report["warnings"]:
        lines += [
            "",
            "## Input verification warnings",
            "",
            *[f"- {warning}" for warning in report["warnings"]],
        ]
    lines += [
        "",
        "Compare latency alongside success/error transitions. A failed request can be faster.",
        "Missing usage is unknown, not zero. Token deltas use only pairs with complete observed usage.",
        "A single paired run does not establish statistical significance or isolate a prompt's causal effect.",
        "",
    ]
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    fields = [
        "case_id",
        "feature",
        "status_transition",
        "label_changes",
        "baseline_error",
        "candidate_error",
        "baseline_calls",
        "candidate_calls",
        "baseline_repair",
        "candidate_repair",
        "latency_ms_delta",
        "total_tokens_delta",
    ]
    with (output_dir / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["per_case"]:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "feature": row["feature"],
                    "status_transition": row["status_transition"],
                    "label_changes": json.dumps(row["label_changes"]),
                    "baseline_error": json.dumps(row["baseline"]["error"]),
                    "candidate_error": json.dumps(row["candidate"]["error"]),
                    "baseline_calls": row["baseline"]["logical_calls"],
                    "candidate_calls": row["candidate"]["logical_calls"],
                    "baseline_repair": row["baseline"]["repair"],
                    "candidate_repair": row["candidate"]["repair"],
                    "latency_ms_delta": row["latency_ms_delta"],
                    "total_tokens_delta": row["total_tokens_delta"],
                }
            )
