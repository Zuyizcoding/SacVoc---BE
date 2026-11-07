"""The legacy adapter must not manufacture accuracy from missing data."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.evaluation.analysis_exports import evaluate_f1, evaluate_f2, index_rows

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("evaluate", [evaluate_f1, evaluate_f2])
def test_missing_reference_and_prediction_cannot_produce_perfect_accuracy(evaluate):
    with pytest.raises(ValueError):
        evaluate([{"case_id": "one"}], {"one": {}})


def test_missing_predictions_are_misses_with_original_denominator():
    benchmark = [
        {"case_id": "one", "expected_shape": "Rectangle"},
        {"case_id": "two", "expected_shape": "Oval"},
    ]
    result = evaluate_f2(benchmark, {"one": {"predicted_shape": "Rectangle"}})
    assert result["shape_accuracy"] == 0.5
    assert result["missing_predictions"] == ["two"]
    assert result["classification"]["shape"]["coverage"] == 0.5


def test_primary_only_reference_has_undefined_subseason_accuracy():
    result = evaluate_f1(
        [{"case_id": "one", "expected_primary_season": "Autumn"}],
        {"one": {"predicted_primary_season": "Autumn"}},
    )
    assert result["primary_season_accuracy"] == 1
    assert result["sub_season_accuracy"] is None
    assert result["classification"]["sub_season"]["case_count"] == 0


def test_partial_prediction_is_not_a_hit_for_missing_field():
    result = evaluate_f1(
        [
            {
                "case_id": "one",
                "expected_primary_season": "Autumn",
                "expected_sub_season": "Soft Autumn",
            }
        ],
        {"one": {"predicted_sub_season": "Soft Autumn"}},
    )
    assert result["primary_season_accuracy"] == 0
    assert result["sub_season_accuracy"] == 1
    assert result["incomplete_predictions"] == ["one"]


@pytest.mark.parametrize("value", ["Unknown", 17, None, ""])
def test_invalid_or_missing_shape_counts_as_miss(value):
    result = evaluate_f2(
        [{"case_id": "one", "expected_shape": "Rectangle"}],
        {"one": {"predicted_shape": value}},
    )
    assert result["shape_accuracy"] == 0


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"case_id": "one", "expected_shape": "unknown"}],
        [{"case_id": "one", "expected_shape": "Rectangle"}] * 2,
    ],
)
def test_invalid_benchmark_is_rejected(rows):
    with pytest.raises(ValueError):
        evaluate_f2(rows, {})


@pytest.mark.parametrize(
    "rows",
    [
        [{"case_id": "one"}] * 2,
        [{"case_id": None}],
        [1],
        [{"case_id": 1}, {"case_id": "1"}],
    ],
)
def test_prediction_index_does_not_silently_overwrite_or_drop_rows(rows):
    with pytest.raises(ValueError):
        index_rows(rows)


def test_unknown_prediction_case_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        evaluate_f2([{"case_id": "one", "expected_shape": "Rectangle"}], {"extra": {}})


@pytest.mark.parametrize("direct_script", [False, True])
def test_both_legacy_cli_invocations_keep_flags_and_report_keys(
    tmp_path, direct_script
):
    benchmark, predictions = tmp_path / "benchmark.json", tmp_path / "predictions.json"
    benchmark.write_text(
        '[{"case_id":"one","expected_shape":"Rectangle"}]', encoding="utf-8"
    )
    predictions.write_text(
        '[{"case_id":"one","predicted_shape":"Rectangle"}]', encoding="utf-8"
    )
    command = (
        [str(ROOT / "scripts/evaluate_analysis_exports.py")]
        if direct_script
        else ["-m", "scripts.evaluate_analysis_exports"]
    )
    completed = subprocess.run(
        [
            sys.executable,
            *command,
            "--task",
            "f2",
            "--benchmark",
            str(benchmark),
            "--predictions",
            str(predictions),
            "--output",
            str(tmp_path / "report.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    report = json.loads(completed.stdout)
    assert report["metrics"]["shape_accuracy"] == 1
    assert report["metrics"]["missing_predictions"] == []
    assert report["case_count"] == 1


def test_cli_rejects_missing_reference_without_writing_a_misleading_report(tmp_path):
    benchmark, predictions = tmp_path / "reference.json", tmp_path / "predictions.json"
    benchmark.write_text('[{"case_id":"one"}]', encoding="utf-8")
    predictions.write_text('[{"case_id":"one"}]', encoding="utf-8")
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_analysis_exports",
            "--task",
            "f1",
            "--benchmark",
            str(benchmark),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert not output.exists()
