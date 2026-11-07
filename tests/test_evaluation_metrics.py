"""Hand-computed examples protect formulas and failure denominators."""

import math

import pytest

from app.evaluation.metrics import classification_metrics, percentile, ranking_metrics


def test_classification_keeps_failures_and_zero_support_classes():
    result = classification_metrics(
        ["A", "A", "B", "B"], ["A", None, "A", "B"], labels=["A", "B", "C"]
    )
    assert result["accuracy"] == 0.5
    assert result["coverage"] == 0.75
    assert result["per_class"]["A"] == {
        "support": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert result["per_class"]["B"]["f1"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 3)
    assert result["confusion_matrix"]["A"]["<missing>"] == 1
    assert result["per_class"]["C"]["support"] == 0


def test_unexpected_prediction_is_a_miss_and_visible_in_matrix():
    result = classification_metrics(["A"], ["unknown"], labels=["A"])
    assert result["accuracy"] == 0
    assert result["confusion_matrix"]["A"]["unknown"] == 1


def test_empty_classification_is_undefined_not_perfect():
    result = classification_metrics([], [], labels=["A"])
    assert result["accuracy"] is None
    assert result["macro_f1"] is None


@pytest.mark.parametrize(
    "references,predictions,labels",
    [
        (["A"], [], ["A"]),
        (["B"], ["A"], ["A"]),
        (["A"], ["A"], ["A", "A"]),
    ],
)
def test_classification_rejects_inconsistent_input(references, predictions, labels):
    with pytest.raises(ValueError):
        classification_metrics(references, predictions, labels=labels)


def test_ranking_uses_independent_graded_judgments_and_fixed_k():
    result = ranking_metrics(
        ["zero", "best", "okay"], {"best": 3, "okay": 2, "zero": 0}, k=6
    )
    expected_dcg = 3 / math.log2(3) + 2 / math.log2(4)
    ideal_dcg = 3 + 2 / math.log2(3)
    assert result["precision_at_k"] == pytest.approx(2 / 6)
    assert result["hit_at_k"] == 1
    assert result["ndcg_at_k"] == pytest.approx(expected_dcg / ideal_dcg)


def test_missing_ranking_is_counted_as_zero():
    result = ranking_metrics([], {"one": 2}, k=6)
    assert result["precision_at_k"] == 0
    assert result["hit_at_k"] == 0
    assert result["ndcg_at_k"] == 0


def test_no_relevant_items_makes_ndcg_undefined():
    assert ranking_metrics(["one"], {"one": 0})["ndcg_at_k"] is None


@pytest.mark.parametrize(
    "ids,judgments,k",
    [
        (["one", "one"], {"one": 1}, 6),
        (["unknown"], {"one": 1}, 6),
        (["one"], {"one": 1}, 0),
        (["one"], {"one": -1}, 6),
        ([], {}, 6),
    ],
)
def test_ranking_rejects_invalid_evidence(ids, judgments, k):
    with pytest.raises(ValueError):
        ranking_metrics(ids, judgments, k=k)


def test_percentile_interpolation_and_empty_input():
    assert percentile([0, 100], 0.95) == 95
    assert percentile([20], 0.95) == 20
    assert percentile([], 0.95) is None
