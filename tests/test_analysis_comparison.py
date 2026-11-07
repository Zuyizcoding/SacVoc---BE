"""Paired changes, missing usage, and input guards use explicit artifact fixtures."""

import json
from copy import deepcopy

import pytest

from app.evaluation.comparison import compare_analysis_runs
from app.evaluation.comparison_report import write_comparison_report
from app.evaluation.comparison_schema import load_analysis_run
from app.services.demo_cache import get_demo_color_result

DIGEST = "a" * 64


def metadata():
    return {
        "artifact_schema_version": 1,
        "run_id": "fixture-run",
        "dataset_id": "fixture",
        "dataset_version": "1",
        "manifest_sha256": DIGEST,
        "synthetic": True,
        "reference_kind": "synthetic",
        "split": "dev",
        "provider": "synthetic",
        "execution_mode": "evaluation",
        "selected_case_ids": ["first", "second", "third"],
        "model_requested": "synthetic",
        "sdk_version": "test",
        "source_sha256": DIGEST,
        "prompt_bundle_sha256": DIGEST,
        "inference_config": {"max_attempts": 2},
        "runtime_dependencies": {"numpy": "test"},
    }


def case(case_id, *, correct=True, failed=False, missing_usage=False):
    result = get_demo_color_result()
    result["analysis_source"] = "live"
    result["primary_season"] = "Autumn" if correct else "Winter"
    result["sub_season"] = "Soft Autumn" if correct else "True Winter"
    return {
        "case_id": case_id,
        "subject_id": case_id,
        "feature": "color",
        "split": "dev",
        "reference": {"primary_season": "Autumn", "sub_season": "Soft Autumn"},
        "input_image_sha256": DIGEST,
        "status": "error" if failed else "success",
        "final_result": None if failed else result,
        "error": {"type": "SyntheticFailure", "code": "AI_INCOMPLETE"}
        if failed
        else None,
        "latency_ms": 100,
        "run_id": "fixture-run",
        "synthetic": True,
        "trace": {
            "processed_image_sha256": DIGEST,
            "photo_quality": {},
            "calls": [
                {
                    "stage": "initial",
                    "prompt": "prompt",
                    "system_instruction": "system",
                    "prompt_sha256": DIGEST,
                    "schema_name": "ColorInferenceResponse",
                    "schema_sha256": DIGEST,
                    "validated_result": None if failed else result,
                    "attempts": [
                        {
                            "number": 1,
                            "latency_ms": 90,
                            "raw_text": "{}",
                            "schema_valid": not failed,
                            "usage": {} if missing_usage else {"total_token_count": 10},
                        }
                    ],
                }
            ],
        },
    }


def write_run(path, meta, rows):
    path.mkdir()
    (path / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    (path / "cases.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


@pytest.fixture
def run_pair(tmp_path):
    before = [
        case("first", failed=True, missing_usage=True),
        case("second"),
        case("third", correct=False),
    ]
    after = [case("first"), case("second", correct=False), case("third", correct=False)]
    write_run(tmp_path / "baseline", metadata(), before)
    write_run(tmp_path / "candidate", metadata(), after)
    return tmp_path / "baseline", tmp_path / "candidate"


def test_paired_changes_include_recovered_errors_regressions_and_missing_usage(
    run_pair,
):
    report = compare_analysis_runs(*run_pair)
    paired = report["paired"]
    assert paired["case_count"] == 3
    assert paired["status_transitions"] == {
        "error_to_success": 1,
        "success_to_success": 2,
    }
    assert paired["label_changes"]["primary_season"] == {
        "improved": 1,
        "regressed": 1,
        "unchanged_incorrect": 1,
    }
    assert report["baseline"]["operations"]["final_success_rate"] == pytest.approx(
        2 / 3
    )
    assert report["candidate"]["operations"]["final_success_rate"] == 1
    assert report["quality_delta"]["color"]["primary_season"]["accuracy"] == 0
    assert report["per_case"][0]["total_tokens_delta"] is None
    assert paired["complete_usage_pair_count"] == 2
    assert paired["total_tokens_delta_on_complete_pairs"] == 0


def update_metadata(directory, field, value):
    path = directory / "run.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")


def update_case(directory, index, mutate):
    path = directory / "cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows[index])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_version", "changed"),
        ("dataset_id", "other"),
        ("manifest_sha256", "b" * 64),
    ],
)
def test_changed_dataset_is_rejected(run_pair, field, value):
    update_metadata(run_pair[1], field, value)
    with pytest.raises(ValueError, match=field):
        compare_analysis_runs(*run_pair)


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra"])
def test_incomplete_duplicate_or_extra_evidence_is_rejected(run_pair, change):
    path = run_pair[1] / "cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    else:
        extra = deepcopy(rows[0])
        extra["case_id"] = "extra"
        rows.append(extra)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        compare_analysis_runs(*run_pair)


@pytest.mark.parametrize(
    "field,value",
    [
        ("subject_id", "other"),
        ("input_image_sha256", "b" * 64),
        ("reference", {"primary_season": "Winter", "sub_season": "True Winter"}),
    ],
)
def test_case_id_cannot_hide_changed_inputs_or_labels(run_pair, field, value):
    update_case(run_pair[1], 0, lambda row: row.update({field: value}))
    with pytest.raises(ValueError, match="Changed"):
        compare_analysis_runs(*run_pair)


def test_model_or_prompt_differences_are_disclosed_instead_of_blocked(run_pair):
    update_metadata(run_pair[1], "model_requested", "different-model")
    update_metadata(run_pair[1], "prompt_bundle_sha256", "b" * 64)
    report = compare_analysis_runs(*run_pair)
    assert {item["field"] for item in report["changed_settings"]} == {
        "model_requested",
        "prompt_bundle_sha256",
    }


def test_reordering_rows_does_not_change_pairing(run_pair):
    path = run_pair[1] / "cases.jsonl"
    path.write_text(
        "\n".join(reversed(path.read_text(encoding="utf-8").splitlines())) + "\n",
        encoding="utf-8",
    )
    report = compare_analysis_runs(*run_pair)
    assert report["per_case"][0]["case_id"] == "first"
    assert report["per_case"][0]["status_transition"] == "error_to_success"


def test_missing_input_proof_keeps_failed_case_with_warning(run_pair):
    update_case(run_pair[0], 0, lambda row: row.update(input_image_sha256=None))
    report = compare_analysis_runs(*run_pair)
    assert report["paired"]["case_count"] == 3
    assert any("first" in warning for warning in report["warnings"])


def test_demo_success_and_inconsistent_status_are_rejected(run_pair):
    update_case(
        run_pair[1],
        0,
        lambda row: row["final_result"].update(analysis_source="degraded_demo"),
    )
    with pytest.raises(ValueError, match="Demo"):
        compare_analysis_runs(*run_pair)


def test_outputs_do_not_overwrite_and_cached_aggregates_are_not_trusted(
    run_pair, tmp_path
):
    (run_pair[0] / "report.json").write_text('{"accuracy": 999}', encoding="utf-8")
    report = compare_analysis_runs(*run_pair)
    output = tmp_path / "comparison"
    write_comparison_report(output, report)
    assert (output / "comparison.md").is_file()
    assert (output / "cases.csv").is_file()
    with pytest.raises(FileExistsError):
        write_comparison_report(output, report)


def test_comparison_of_same_run_has_zero_deltas(run_pair):
    report = compare_analysis_runs(run_pair[0], run_pair[0])
    assert all(row["latency_ms_delta"] == 0 for row in report["per_case"])
    assert report["quality_delta"]["color"]["primary_season"]["accuracy"] == 0


def test_new_metadata_contract_rejects_old_unversioned_artifacts(run_pair):
    path = run_pair[1] / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["artifact_schema_version"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_analysis_run(run_pair[1])


def test_same_model_alias_with_changed_resolved_version_is_disclosed(run_pair):
    def set_version(row):
        row["trace"]["calls"][0]["attempts"][0]["model_version"] = (
            "new-provider-version"
        )

    update_case(run_pair[1], 0, set_version)
    report = compare_analysis_runs(*run_pair)
    assert {change["field"] for change in report["changed_settings"]} == {
        "observed_model_versions"
    }
