"""Legacy flat-export CLI, backed by the analysis runner's shared metrics."""

import argparse
import json
import sys
from pathlib import Path

# Retain direct-script execution as well as python -m.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.analysis_exports import (  # noqa: E402
    evaluate_f1,
    evaluate_f2,
    index_rows,
)


def _load_json_list(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise ValueError(f"Expected a JSON list of objects in {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--task", choices=["f1", "f2"], required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        benchmark_path = Path(args.benchmark).resolve()
        predictions_path = Path(args.predictions).resolve()
        benchmark = _load_json_list(str(benchmark_path))
        predictions = index_rows(_load_json_list(str(predictions_path)))
        evaluate = evaluate_f1 if args.task == "f1" else evaluate_f2
        metrics = evaluate(benchmark, predictions)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    report = {
        "task": args.task,
        "benchmark": str(benchmark_path),
        "predictions": str(predictions_path),
        "case_count": len(benchmark),
        "metrics": metrics,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        # Retain the legacy explicit --output overwrite behavior.
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
