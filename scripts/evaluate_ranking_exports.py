"""Score F3 ranking exports against independent item-level relevance judgments.

This tool reads files only: no database, model loading, or recommender mutation.
Prediction format: [{"case_id": "...", "outfit_ids": ["1", "2", ...]}].
Existing name/category smoke checks in evaluate_f3.py remain a separate tool.
"""

import argparse
import json
from pathlib import Path

from app.evaluation.metrics import ranking_metrics
from app.evaluation.schemas import StrictModel, load_ranking_dataset


class RankedPrediction(StrictModel):
    case_id: str
    outfit_ids: list[str]


def evaluate_exports(dataset_path: Path, predictions_path: Path, *, k: int = 6) -> dict:
    dataset = load_ranking_dataset(dataset_path)
    predictions = [
        RankedPrediction.model_validate(item)
        for item in json.loads(predictions_path.read_text(encoding="utf-8-sig"))
    ]
    by_id = {item.case_id: item for item in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate prediction case IDs")
    unknown = by_id.keys() - {case.case_id for case in dataset.cases}
    if unknown:
        raise ValueError(f"Unknown prediction case IDs: {sorted(unknown)}")
    rows = []
    for case in dataset.cases:
        prediction = by_id.get(case.case_id)
        rows.append(
            {
                "case_id": case.case_id,
                "missing_prediction": prediction is None,
                **ranking_metrics(
                    prediction.outfit_ids if prediction else [], case.relevance, k=k
                ),
            }
        )
    comparable = [row["ndcg_at_k"] for row in rows if row["ndcg_at_k"] is not None]
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.version,
        "synthetic": dataset.synthetic,
        "interpretation": "Synthetic formula check only"
        if dataset.synthetic
        else "Agreement with independent relevance judgments",
        "metrics": {
            "precision_at_k": sum(row["precision_at_k"] for row in rows) / len(rows),
            "hit_at_k": sum(row["hit_at_k"] for row in rows) / len(rows),
            "ndcg_at_k": sum(comparable) / len(comparable) if comparable else None,
            "ndcg_defined_cases": len(comparable),
            "case_count": len(rows),
        },
        "per_case": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_exports(args.dataset, args.predictions, k=args.top_k)
    # Exclusive creation prevents accidentally replacing previous reports.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
