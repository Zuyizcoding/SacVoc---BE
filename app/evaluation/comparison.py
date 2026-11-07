"""Paired comparison: validate identity -> summarize every case -> compute deltas.

A win/loss is defined per reference field, not by confidence or success alone.
An error is an incorrect prediction, so recovering a failed case can be a win.
Latency and usage are diagnostics, never grounds for claiming better AI quality.
All numeric deltas are candidate minus baseline; undefined values stay null.
"""

from collections import Counter
from pathlib import Path

from .comparison_schema import (
    AnalysisRun,
    CaseEvidence,
    load_analysis_run,
    validate_pair,
)
from .metrics import operational_metrics
from .reporting import label_agreement_metrics


def _usage(case: CaseEvidence) -> dict:
    attempts = [attempt for call in case.trace.calls for attempt in call.attempts]
    observed = [
        a.usage["total_token_count"]
        for a in attempts
        if a.usage.get("total_token_count") is not None
    ]
    complete = len(observed) == len(attempts)
    return {
        "observed_total_tokens": sum(observed),
        "complete": complete,
        "total_tokens": sum(observed) if complete else None,
    }


def _case_diagnostics(case: CaseEvidence) -> dict:
    calls = case.trace.calls
    return {
        "status": case.status,
        "error": case.error.model_dump() if case.error else None,
        "latency_ms": case.latency_ms,
        "logical_calls": len(calls),
        "provider_attempts": sum(len(call.attempts) for call in calls),
        "repair": any(call.stage.startswith("repair") for call in calls),
        "usage": _usage(case),
    }


def _field_changes(before: CaseEvidence, after: CaseEvidence) -> dict[str, str]:
    outcomes = {}
    for field, reference in before.reference.model_dump(exclude_none=True).items():
        first = (before.final_result or {}).get(field) == reference
        second = (after.final_result or {}).get(field) == reference
        if first == second:
            outcomes[field] = "unchanged_correct" if first else "unchanged_incorrect"
        else:
            outcomes[field] = "improved" if second else "regressed"
    return outcomes


def _summary(run: AnalysisRun) -> dict:
    rows = [case.model_dump(mode="json") for case in run.cases.values()]
    return {
        "operations": operational_metrics(rows),
        "quality": label_agreement_metrics(rows),
    }


def _observed_model_versions(run: AnalysisRun) -> list[str]:
    return sorted(
        {
            attempt.model_version
            for case in run.cases.values()
            for call in case.trace.calls
            for attempt in call.attempts
            if attempt.model_version is not None
        }
    )


def _quality_deltas(baseline: dict, candidate: dict) -> dict:
    changes = {}
    for feature, dimensions in baseline.items():
        changes[feature] = {}
        for field, metrics in dimensions.items():
            changes[feature][field] = {}
            for name in ("coverage", "accuracy", "macro_f1"):
                first, second = metrics[name], candidate[feature][field][name]
                changes[feature][field][name] = (
                    second - first if first is not None and second is not None else None
                )
    return changes


def compare_analysis_runs(baseline_dir: Path, candidate_dir: Path) -> dict:
    baseline = load_analysis_run(baseline_dir)
    candidate = load_analysis_run(candidate_dir)
    warnings = validate_pair(baseline, candidate)
    rows = []
    transitions: Counter = Counter()
    field_totals: dict[str, Counter] = {}
    for case_id in baseline.metadata.selected_case_ids:
        before, after = baseline.cases[case_id], candidate.cases[case_id]
        first, second = _case_diagnostics(before), _case_diagnostics(after)
        transition = f"{before.status}_to_{after.status}"
        transitions[transition] += 1
        changes = _field_changes(before, after)
        for field, outcome in changes.items():
            field_totals.setdefault(field, Counter())[outcome] += 1
        first_tokens, second_tokens = (
            first["usage"]["total_tokens"],
            second["usage"]["total_tokens"],
        )
        rows.append(
            {
                "case_id": case_id,
                "feature": before.feature,
                "baseline": first,
                "candidate": second,
                "status_transition": transition,
                "label_changes": changes,
                "latency_ms_delta": after.latency_ms - before.latency_ms,
                "total_tokens_delta": second_tokens - first_tokens
                if first_tokens is not None and second_tokens is not None
                else None,
            }
        )
    comparable_usage = [row for row in rows if row["total_tokens_delta"] is not None]
    both_succeeded = [
        row for row in rows if row["status_transition"] == "success_to_success"
    ]
    changed_settings = []
    for field in (
        "model_requested",
        "sdk_version",
        "inference_config",
        "runtime_dependencies",
        "source_sha256",
        "prompt_bundle_sha256",
    ):
        first, second = (
            getattr(baseline.metadata, field),
            getattr(candidate.metadata, field),
        )
        if first != second:
            changed_settings.append(
                {"field": field, "baseline": first, "candidate": second}
            )
    first_summary, second_summary = _summary(baseline), _summary(candidate)
    first_versions = _observed_model_versions(baseline)
    second_versions = _observed_model_versions(candidate)
    if first_versions != second_versions:
        changed_settings.append(
            {
                "field": "observed_model_versions",
                "baseline": first_versions,
                "candidate": second_versions,
            }
        )
    return {
        "interpretation": (
            "Synthetic paired software check; no AI quality/performance conclusion."
            if baseline.metadata.synthetic
            else "Descriptive paired agreement comparison; does not establish causality or statistical significance."
        ),
        "baseline_metadata": baseline.metadata.model_dump(),
        "candidate_metadata": candidate.metadata.model_dump(),
        "warnings": warnings,
        "changed_settings": changed_settings,
        "baseline": first_summary,
        "candidate": second_summary,
        "quality_delta": _quality_deltas(
            first_summary["quality"], second_summary["quality"]
        ),
        "paired": {
            "case_count": len(rows),
            "status_transitions": dict(transitions),
            "label_changes": {
                field: dict(counts) for field, counts in field_totals.items()
            },
            "mean_latency_ms_delta_all_cases": sum(
                row["latency_ms_delta"] for row in rows
            )
            / len(rows),
            "both_succeeded_cases": len(both_succeeded),
            "mean_latency_ms_delta_both_succeeded": sum(
                row["latency_ms_delta"] for row in both_succeeded
            )
            / len(both_succeeded)
            if both_succeeded
            else None,
            "complete_usage_pair_count": len(comparable_usage),
            "total_tokens_delta_on_complete_pairs": sum(
                row["total_tokens_delta"] for row in comparable_usage
            )
            if comparable_usage
            else None,
        },
        "per_case": rows,
    }
