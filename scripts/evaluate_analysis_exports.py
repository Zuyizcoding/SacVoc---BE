"""Compare exported F1/F2 predictions against labeled benchmarks."""

import argparse
import json
import os
from typing import Any


def _load_json_list(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def evaluate_f1(
    benchmark: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    total = len(benchmark) or 1
    season_hits = 0
    primary_hits = 0
    missing_predictions: list[str] = []

    for case in benchmark:
        case_id = str(case["case_id"])
        prediction = predictions.get(case_id)
        if prediction is None:
            missing_predictions.append(case_id)
            continue
        if str(prediction.get("predicted_sub_season")) == str(
            case.get("expected_sub_season")
        ):
            season_hits += 1
        if str(prediction.get("predicted_primary_season")) == str(
            case.get("expected_primary_season")
        ):
            primary_hits += 1

    return {
        "sub_season_accuracy": round(season_hits / total, 4),
        "primary_season_accuracy": round(primary_hits / total, 4),
        "missing_predictions": missing_predictions,
    }


def evaluate_f2(
    benchmark: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    total = len(benchmark) or 1
    shape_hits = 0
    missing_predictions: list[str] = []

    for case in benchmark:
        case_id = str(case["case_id"])
        prediction = predictions.get(case_id)
        if prediction is None:
            missing_predictions.append(case_id)
            continue
        if str(prediction.get("predicted_shape")) == str(case.get("expected_shape")):
            shape_hits += 1

    return {
        "shape_accuracy": round(shape_hits / total, 4),
        "missing_predictions": missing_predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exported F1/F2 predictions")
    parser.add_argument("--benchmark", required=True, help="Benchmark JSON path")
    parser.add_argument("--predictions", required=True, help="Predictions JSON path")
    parser.add_argument("--task", choices=["f1", "f2"], required=True)
    parser.add_argument("--output", help="Optional path to write JSON report")
    args = parser.parse_args()

    benchmark = _load_json_list(os.path.abspath(args.benchmark))
    prediction_rows = _load_json_list(os.path.abspath(args.predictions))
    predictions = {
        str(item["case_id"]): item for item in prediction_rows if "case_id" in item
    }

    if args.task == "f1":
        metrics = evaluate_f1(benchmark, predictions)
    else:
        metrics = evaluate_f2(benchmark, predictions)

    report = {
        "task": args.task,
        "benchmark": os.path.abspath(args.benchmark),
        "predictions": os.path.abspath(args.predictions),
        "case_count": len(benchmark),
        "metrics": metrics,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output:
        output_path = os.path.abspath(args.output)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
