"""Offline example: generated image -> preflight -> read-only export -> ablation.

Every source artifact is generated under a new output directory and explicitly
synthetic. This script never opens the application's database or encodes a model.
Its artificial query bundle tests the 240-template contract; committed vectors
remain untouched.
"""

import argparse
import json
from pathlib import Path

from PIL import Image
from pytest_socket import socket_allow_hosts

from app.evaluation.image_inspection import inspect_image_path
from app.evaluation.preflight import (
    preflight_analysis,
    preflight_retrieval,
    write_preflight_report,
)
from app.evaluation.provenance import PROJECT_ROOT
from app.evaluation.retrieval import evaluate_snapshot, write_ablation_report
from app.evaluation.snapshot_export import export_snapshot
from app.evaluation.synthetic_catalog import write_synthetic_catalog_sources


def run_smoke(output_dir: Path) -> None:
    import socket

    output_dir.mkdir(parents=True, exist_ok=False)
    original_connect = socket.socket.connect
    socket_allow_hosts(["127.0.0.1", "::1"], allow_unix_socket=True)
    try:
        image_path = output_dir / "generated.png"
        Image.new("RGB", (320, 480), "green").save(image_path)
        image = inspect_image_path(image_path)
        if image.issues or not image.rgb_sha256:
            raise RuntimeError("Generated image inspection failed")
        (output_dir / "image-inspection.json").write_text(
            image.model_dump_json(indent=2), encoding="utf-8"
        )
        analysis = preflight_analysis(
            PROJECT_ROOT / "evals/synthetic/analysis.dataset.json"
        )
        write_preflight_report(output_dir / "analysis-preflight", analysis)
        inputs = write_synthetic_catalog_sources(output_dir)
        original_database = inputs[1].read_bytes()
        provenance = export_snapshot(*inputs, output_dir / "export")
        if inputs[1].read_bytes() != original_database:
            raise RuntimeError("Read-only export changed the synthetic database")
        retrieval = preflight_retrieval(output_dir / "export/snapshot.json")
        write_preflight_report(output_dir / "retrieval-preflight", retrieval)
        if not analysis["ready"] or not retrieval["ready"]:
            raise RuntimeError("Synthetic preflight failed")
        report = evaluate_snapshot(output_dir / "export/snapshot.json", split="dev")
        write_ablation_report(output_dir / "ablation", report)
        (output_dir / "smoke-summary.json").write_text(
            json.dumps(
                {
                    "synthetic": True,
                    "image_inspected": True,
                    "database_unchanged": True,
                    "exported_cases": provenance["case_count"],
                    "exported_outfits": provenance["encoded_outfit_count"],
                    "interpretation": "Software/data-contract checks only; no AI quality claim.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        socket.socket.connect = original_connect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_smoke(args.output_dir)
    print(f"Data preparation smoke checks passed: {args.output_dir}")


if __name__ == "__main__":
    main()
