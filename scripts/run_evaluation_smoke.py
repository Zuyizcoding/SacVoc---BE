"""Run the CI evaluation commands locally without credentials or model downloads.

Pipeline: two synthetic analysis runs -> paired comparison -> F3 ablation.
Identical scripted runs must have identical quality outcomes, not identical timing.
Only loopback/Unix sockets are allowed in this process while these checks run.
Package installation and GitHub artifact upload happen outside this offline phase.
"""

import argparse
import asyncio
import json
import socket
from pathlib import Path

from pytest_socket import socket_allow_hosts

from app.evaluation.comparison import compare_analysis_runs
from app.evaluation.comparison_report import write_comparison_report
from app.evaluation.provenance import PROJECT_ROOT
from app.evaluation.retrieval import evaluate_snapshot, write_ablation_report
from app.evaluation.runner import run_dataset


async def _run_checks(output_dir: Path) -> None:
    fixtures = PROJECT_ROOT / "evals/synthetic"
    for name in ("baseline", "candidate"):
        report = await run_dataset(
            fixtures / "analysis.dataset.json",
            output_dir / name,
            provider="synthetic",
            split="dev",
        )
        if (
            report["operations"]["case_count"] != 10
            or not report["metadata"]["synthetic"]
        ):
            raise RuntimeError("Unexpected analysis smoke dataset/result")
    paired = compare_analysis_runs(output_dir / "baseline", output_dir / "candidate")
    if any(
        row["status_transition"] in {"success_to_error", "error_to_success"}
        or any(
            value in {"improved", "regressed"}
            for value in row["label_changes"].values()
        )
        for row in paired["per_case"]
    ):
        raise RuntimeError("Identical scripted runs produced different outcomes")
    write_comparison_report(output_dir / "comparison", paired)
    retrieval = evaluate_snapshot(fixtures / "retrieval.snapshot.json", split="dev")
    write_ablation_report(output_dir / "retrieval", retrieval)
    if any(item["case_count"] != 2 for item in retrieval["summary"].values()):
        raise RuntimeError("Unexpected retrieval case count")
    (output_dir / "smoke-summary.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "analysis_cases_per_run": 10,
                "paired_cases": paired["paired"]["case_count"],
                "retrieval_cases_per_variant": 2,
                "interpretation": "Offline software checks only; no model quality claim.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_smoke(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    original_connect = socket.socket.connect
    socket_allow_hosts(["127.0.0.1", "::1"], allow_unix_socket=True)
    try:
        asyncio.run(_run_checks(output_dir))
    finally:
        socket.socket.connect = original_connect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_smoke(args.output_dir)
    print(f"Synthetic smoke checks passed: {args.output_dir}")


if __name__ == "__main__":
    main()
