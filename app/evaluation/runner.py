"""Run a versioned F1/F2 benchmark without HTTP endpoints or database sessions.

Read this module in pipeline order:
1. validate the manifest and select one split;
2. capture reproducibility metadata before any inference;
3. process every case through the production analysis workflow, without demos;
4. flush evidence after each case, then aggregate metrics.

One failed case is evidence, not a reason to silently shrink the denominator.
Live runs share a client/breaker as the API does. Synthetic cases use independent
scripts and must never be compared to real latency, cost, or quality measurements.
"""

import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Literal
from uuid import uuid4

from app.services.analysis.tracing import capture_analysis_trace, content_hash

from .provenance import PROJECT_ROOT, git_output, source_fingerprint
from .reporting import build_report, write_case_summary, write_report
from .schemas import AnalysisCase, AnalysisDataset, load_dataset


def _metadata(
    dataset: AnalysisDataset, manifest_path: Path, provider: str, split: str
) -> dict:
    from app.core.config import settings

    git_status = git_output("status", "--porcelain")
    return {
        "run_id": str(uuid4()),
        "artifact_schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.version,
        "manifest_sha256": content_hash(manifest_path.read_bytes()),
        "synthetic": dataset.synthetic,
        "reference_kind": dataset.reference_kind,
        "split": split,
        "selected_case_ids": [
            case.case_id for case in dataset.cases if case.split == split
        ],
        "provider": provider,
        "execution_mode": "evaluation",
        "model_requested": settings.GEMINI_MODEL
        if provider == "gemini"
        else "synthetic",
        "sdk_version": version("google-genai"),
        "python_version": platform.python_version(),
        "pydantic_version": version("pydantic"),
        "pillow_version": version("Pillow"),
        "code_commit": git_output("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git_status) if git_status is not None else None,
        "source_sha256": source_fingerprint(),
        "prompt_bundle_sha256": content_hash(
            (PROJECT_ROOT / "app/core/prompts.py").read_bytes()
            + (PROJECT_ROOT / "app/services/analysis/prompts.py").read_bytes()
        ),
        "runtime_dependencies": {
            package: version(package)
            for package in ("google-genai", "httpx", "pydantic", "Pillow", "numpy")
        },
        "inference_config": {
            "api_version": "v1beta",
            "timeout_seconds_per_attempt": settings.GEMINI_TIMEOUT,
            "max_attempts_per_call": settings.GEMINI_MAX_RETRIES,
            "sdk_attempts": 1,
            "thinking_budget": settings.GEMINI_THINKING_BUDGET,
            "temperature": "provider_default",
            "safety_thresholds": {
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
            },
            "preprocess_max_kb": 1000,
        },
    }


async def _run_case(case: AnalysisCase, client, image: bytes | Path) -> dict:
    from app.services.analysis.workflows import analyze_image

    started = time.perf_counter()
    result = None
    error = None
    image_hash = None
    with capture_analysis_trace() as trace:
        try:
            image_bytes = image.read_bytes() if isinstance(image, Path) else image
            image_hash = content_hash(image_bytes)
            result = (
                await analyze_image(
                    image_bytes, client, feature=case.feature, mode="evaluation"
                )
            ).model_dump(mode="json")
        except Exception as exc:
            # Store a category/code, not arbitrary SDK exception text (which can
            # include transport details). Rejections remain visible as failures.
            error = {
                "type": type(exc).__name__,
                "code": getattr(exc, "error_code", None),
            }
    return {
        "case_id": case.case_id,
        "subject_id": case.subject_id,
        "feature": case.feature,
        "split": case.split,
        "reference": case.reference.model_dump(exclude_none=True),
        "input_image_sha256": image_hash,
        "status": "success" if result is not None else "error",
        "final_result": result,
        "error": error,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "trace": trace.model_dump(mode="json"),
    }


async def run_dataset(
    manifest_path: Path,
    output_dir: Path,
    *,
    provider: Literal["synthetic", "gemini"],
    split: Literal["dev", "test"],
) -> dict:
    """Write to a new output directory; never overwrite previous experiment data."""
    if provider not in {"synthetic", "gemini"}:
        raise ValueError("Unknown evaluation provider")
    dataset = load_dataset(manifest_path)
    if dataset.synthetic != (provider == "synthetic"):
        raise ValueError(
            "Synthetic manifests require synthetic provider; human manifests require Gemini"
        )
    selected = [case for case in dataset.cases if case.split == split]
    if not selected:
        raise ValueError(f"No cases in split: {split}")
    if provider == "gemini":
        for case in selected:
            if not (manifest_path.parent / case.image_path).is_file():
                raise ValueError(f"Image file missing for case: {case.case_id}")

    from app.core.config import settings
    from app.services.gemini_client import GeminiClient

    from .synthetic import generated_image, scripted_sdk

    # Construct before creating artifacts so missing credentials fail immediately.
    shared_client = GeminiClient() if provider == "gemini" else None
    try:
        metadata = _metadata(dataset, manifest_path, provider, split)
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "run.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        synthetic_image = generated_image() if dataset.synthetic else None
        rows = []
        with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as evidence:
            for case in selected:
                client = shared_client or GeminiClient(
                    scripted_sdk(
                        case.synthetic_replies, timeout_seconds=settings.GEMINI_TIMEOUT
                    )
                )
                try:
                    image = (
                        synthetic_image
                        if dataset.synthetic
                        else (manifest_path.parent / case.image_path)
                    )
                    row = await _run_case(case, client, image)
                finally:
                    if shared_client is None:
                        client.close()
                row.update(run_id=metadata["run_id"], synthetic=dataset.synthetic)
                rows.append(row)
                evidence.write(json.dumps(row, ensure_ascii=False) + "\n")
                evidence.flush()
        report = build_report(dataset, rows, metadata)
        write_case_summary(output_dir, rows)
        write_report(output_dir, report)
        return report
    finally:
        if shared_client is not None:
            shared_client.close()
