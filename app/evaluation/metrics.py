"""Small, dependency-free metrics with explicit denominators.

Classification: failures count as misses; coverage is reported separately.
Macro F1 gives each declared class equal weight (including zero-support classes).
Ranking: relevance comes from independent judgments, never from our match score.
Missing recommendation slots count against Precision@k; unknown relevance is an
error rather than an implicit zero. These choices make comparisons reviewable.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence


def classification_metrics(
    references: Sequence[str],
    predictions: Sequence[str | None],
    *,
    labels: Sequence[str],
) -> dict:
    """Return row=true, column=predicted confusion counts and per-class support."""
    if len(references) != len(predictions):
        raise ValueError("Each reference needs one prediction, including failures")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Declare a non-empty, unique label set")
    if any(label not in labels for label in references):
        raise ValueError("Reference label is outside the declared taxonomy")

    missing = "<missing>"
    unexpected = sorted({p for p in predictions if p is not None and p not in labels})
    columns = [*labels, *unexpected, missing]
    matrix = {label: {column: 0 for column in columns} for label in labels}
    per_class = {}
    for actual, predicted in zip(references, predictions):
        matrix[actual][predicted if predicted is not None else missing] += 1
    for label in labels:
        true_positive = sum(
            a == label and p == label for a, p in zip(references, predictions)
        )
        false_positive = sum(
            a != label and p == label for a, p in zip(references, predictions)
        )
        false_negative = sum(
            a == label and p != label for a, p in zip(references, predictions)
        )
        support = references.count(label)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    total = len(references)
    return {
        "case_count": total,
        "coverage": sum(p is not None for p in predictions) / total if total else None,
        "accuracy": sum(a == p for a, p in zip(references, predictions)) / total
        if total
        else None,
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(labels)
        if total
        else None,
        "macro_labels": list(labels),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def ranking_metrics(
    returned_ids: Sequence[str], relevance: Mapping[str, int], *, k: int = 6
) -> dict:
    """Precision@k, Hit@k, and linear-gain NDCG@k for one judged query.

    DCG = sum(grade / log2(rank + 1)); IDCG uses the ideal order of the judgment
    pool. With no positive judgments, NDCG is undefined and reported as null.
    """
    if k <= 0 or not relevance:
        raise ValueError("k and the judgment pool must be non-empty/positive")
    if any(type(g) is not int or not 0 <= g <= 3 for g in relevance.values()):
        raise ValueError("Relevance grades must be integers from 0 to 3")
    top_ids = list(returned_ids[:k])
    if len(top_ids) != len(set(top_ids)):
        raise ValueError("Duplicate recommendation IDs cannot earn extra credit")
    unknown = set(top_ids) - relevance.keys()
    if unknown:
        raise ValueError(f"Unjudged recommendation IDs: {sorted(unknown)}")
    grades = [relevance[item_id] for item_id in top_ids]
    hits = sum(grade > 0 for grade in grades)
    dcg = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(grades))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    return {
        "precision_at_k": hits / k,
        "hit_at_k": int(hits > 0),
        "ndcg_at_k": dcg / idcg if idcg else None,
        "returned_count": len(top_ids),
        "k": k,
    }


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Linearly interpolate between adjacent sorted observations."""
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def operational_metrics(rows: Sequence[dict]) -> dict:
    """Include every selected case, even preprocessing/provider failures."""
    total = len(rows)
    if not total:
        return {"case_count": 0}
    calls = [call for row in rows for call in row["trace"]["calls"]]
    attempts = [attempt for call in calls for attempt in call["attempts"]]
    first_attempts = [
        row["trace"]["calls"][0]["attempts"][0]
        for row in rows
        if row["trace"]["calls"] and row["trace"]["calls"][0]["attempts"]
    ]
    replies = [attempt for attempt in attempts if attempt["raw_text"] is not None]
    token_totals: Counter = Counter()
    for attempt in attempts:
        token_totals.update(
            {
                name: value
                for name, value in attempt["usage"].items()
                if value is not None
            }
        )
    return {
        "case_count": total,
        "final_success_rate": sum(row["status"] == "success" for row in rows) / total,
        "first_pass_schema_valid_rate": sum(a["schema_valid"] for a in first_attempts)
        / total,
        "schema_valid_reply_rate": sum(a["schema_valid"] for a in replies)
        / len(replies)
        if replies
        else None,
        "repair_rate": sum(
            any(c["stage"].startswith("repair") for c in row["trace"]["calls"])
            for row in rows
        )
        / total,
        "clothing_best_effort_rate": sum(
            any(c["stage"] == "clothing_best_effort" for c in row["trace"]["calls"])
            for row in rows
        )
        / total,
        "logical_model_calls": len(calls),
        "provider_attempts": len(attempts),
        "latency_ms_p50": percentile([row["latency_ms"] for row in rows], 0.5),
        "latency_ms_p95": percentile([row["latency_ms"] for row in rows], 0.95),
        "observed_token_totals": dict(token_totals),
        "attempts_with_usage": sum(bool(a["usage"]) for a in attempts),
        "errors": dict(Counter(row["error"]["type"] for row in rows if row["error"])),
    }
