"""Offline evaluation for the male-focused F3 recommendation flow."""

import argparse
import json
import os
import sys
from statistics import mean
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.database import SessionLocal, init_db
from app.services.outfit_service import recommend_outfits


def _load_cases(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Benchmark file must be a JSON list")
    return [item for item in data if isinstance(item, dict)]


def _name_hit(expected_terms: list[str], returned_names: list[str]) -> bool:
    if not expected_terms:
        return False
    haystack = " | ".join(returned_names).lower()
    return any(term.lower() in haystack for term in expected_terms)


def _category_hit(
    expected_categories: list[str], returned_categories: list[str]
) -> bool:
    if not expected_categories:
        return False
    actual = {item.lower() for item in returned_categories}
    return any(category.lower() in actual for category in expected_categories)


def evaluate(benchmark_path: str, *, top_k: int = 6) -> dict[str, Any]:
    init_db()
    db = SessionLocal()
    try:
        cases = _load_cases(benchmark_path)
        per_case: list[dict[str, Any]] = []

        for case in cases:
            result = recommend_outfits(
                db=db,
                color_sub_season=str(case["color_sub_season"]),
                body_shape=str(case["body_shape"]),
                occasion=str(case.get("occasion") or "daily"),
                gender="male",
                top_k=top_k,
            )
            names = [item.name for item in result.outfits]
            categories = [item.category for item in result.outfits]
            scores = [item.score for item in result.outfits]

            per_case.append(
                {
                    "case_id": case["case_id"],
                    "name_hit": _name_hit(case.get("expected_any_names", []), names),
                    "category_hit": _category_hit(
                        case.get("expected_any_categories", []), categories
                    ),
                    "avg_score": round(mean(scores), 4) if scores else 0.0,
                    "query_used": result.query_used,
                    "returned_names": names,
                    "returned_categories": categories,
                }
            )

        total = len(per_case) or 1
        name_hit_rate = sum(1 for item in per_case if item["name_hit"]) / total
        category_hit_rate = sum(1 for item in per_case if item["category_hit"]) / total
        return {
            "benchmark": benchmark_path,
            "cases": len(per_case),
            "metrics": {
                "name_hit_rate": round(name_hit_rate, 4),
                "category_hit_rate": round(category_hit_rate, 4),
            },
            "per_case": per_case,
        }
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate F3 recommendation quality")
    parser.add_argument("--benchmark", required=True, help="Path to F3 benchmark JSON")
    parser.add_argument(
        "--top-k", type=int, default=6, help="Number of results to evaluate"
    )
    parser.add_argument("--output", help="Optional path to write JSON report")
    args = parser.parse_args()

    report = evaluate(os.path.abspath(args.benchmark), top_k=args.top_k)
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
