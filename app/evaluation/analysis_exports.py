"""Strict adapter for the legacy flat F1/F2 export format.

Missing reference labels are invalid benchmark data. Missing model predictions
are failures and stay in the denominator. Validate references first, then delegate
scoring to shared metrics; comparing two missing strings cannot establish a hit.
"""

from collections.abc import Mapping

from app.data.body_shapes import MALE_BODY_SHAPES
from app.data.season_data import SEASON_DATA

from .metrics import classification_metrics
from .schemas import PRIMARY_SEASONS, ReferenceLabels, validate_reference_labels


def case_id(value: object) -> str:
    """Retain numeric legacy IDs; reject null, blank and boolean IDs."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("case_id must be a non-empty string or integer")
    result = str(value).strip()
    if not result:
        raise ValueError("case_id cannot be blank")
    return result


def index_rows(rows: list[dict]) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every input row must be a JSON object")
        ident = case_id(row.get("case_id"))
        if ident in indexed:
            raise ValueError(f"Duplicate case_id: {ident}")
        indexed[ident] = row
    return indexed


def _validated_predictions(predictions: Mapping, expected_ids: set[str]) -> dict:
    indexed = {}
    for key, row in predictions.items():
        ident = case_id(key)
        if ident in indexed:
            raise ValueError(f"Duplicate prediction case_id: {ident}")
        if not isinstance(row, dict):
            raise ValueError(f"Prediction must be an object: {ident}")
        if "case_id" in row and case_id(row["case_id"]) != ident:
            raise ValueError(f"Prediction row/key disagree: {ident}")
        indexed[ident] = row
    if set(indexed) - expected_ids:
        raise ValueError(
            f"Unknown prediction case IDs: {sorted(set(indexed) - expected_ids)}"
        )
    return indexed


def _prediction_label(
    row: dict, field: str, labels, ident: str, invalid: list
) -> str | None:
    value = row.get(field)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        invalid.append({"case_id": ident, "field": field, "reason": "invalid_type"})
        return None
    if value not in labels:
        invalid.append({"case_id": ident, "field": field, "reason": "unknown_label"})
    # Unexpected strings remain visible in confusion matrices and count as misses.
    return value


def _evaluate(benchmark: list[dict], predictions: Mapping, *, feature: str) -> dict:
    references = index_rows(benchmark)
    if not references:
        raise ValueError("Benchmark cannot be empty")
    indexed = _validated_predictions(predictions, set(references))
    for row in references.values():
        labels = ReferenceLabels(
            primary_season=row.get("expected_primary_season")
            if feature == "color"
            else None,
            sub_season=row.get("expected_sub_season") if feature == "color" else None,
            shape=row.get("expected_shape") if feature == "body" else None,
        )
        validate_reference_labels(feature, labels)
    dimensions = (
        {"primary_season": PRIMARY_SEASONS, "sub_season": tuple(SEASON_DATA)}
        if feature == "color"
        else {"shape": MALE_BODY_SHAPES}
    )
    metrics, invalid, incomplete = {}, [], set()
    for field, taxonomy in dimensions.items():
        expected, actual = [], []
        for ident, reference in references.items():
            label = reference.get(f"expected_{field}")
            if label is None:
                continue
            expected.append(label)
            prediction = _prediction_label(
                indexed.get(ident, {}), f"predicted_{field}", taxonomy, ident, invalid
            )
            actual.append(prediction)
            if ident in indexed and prediction is None:
                incomplete.add(ident)
        metrics[field] = classification_metrics(expected, actual, labels=taxonomy)
    result = {
        f"{field}_accuracy": round(metric["accuracy"], 4)
        if metric["accuracy"] is not None
        else None
        for field, metric in metrics.items()
    }
    result.update(
        missing_predictions=[ident for ident in references if ident not in indexed],
        incomplete_predictions=sorted(incomplete),
        invalid_predictions=invalid,
        classification=metrics,
    )
    return result


def evaluate_f1(benchmark: list[dict], predictions: Mapping) -> dict:
    """Require primary season; optional sub-season uses its own labeled subset."""
    return _evaluate(benchmark, predictions, feature="color")


def evaluate_f2(benchmark: list[dict], predictions: Mapping) -> dict:
    return _evaluate(benchmark, predictions, feature="body")
