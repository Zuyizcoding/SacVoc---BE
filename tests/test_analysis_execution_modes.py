"""One pipeline, explicit policies: evaluation never turns failure into a demo."""

import asyncio
from copy import deepcopy

import pytest

from app.evaluation.runner import _run_case
from app.evaluation.schemas import load_dataset
from app.services.analysis.errors import AnalysisRejectedError, GeminiOverloadedError
from app.services.analysis.photo_types import PhotoQualityReport, ProcessedImage
from app.services.analysis.tracing import capture_analysis_trace, current_stage
from app.services.analysis.workflows import analyze_image


class ScriptedInference:
    """A workflow-level fake; SDK serialization is covered in transport tests."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.stages = []

    async def analyze(self, **_request):
        self.stages.append(current_stage())
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)


@pytest.fixture(autouse=True)
def processed_photo(monkeypatch):
    monkeypatch.setattr(
        "app.services.analysis.workflows.preprocess_image_for_ai",
        lambda *_args, **_kwargs: ProcessedImage(
            b"processed-jpeg", PhotoQualityReport(photo_quality_score=0.9)
        ),
    )


@pytest.mark.parametrize("feature", ["color", "body"])
@pytest.mark.parametrize(
    "failure", [TimeoutError("timeout"), GeminiOverloadedError("quota")]
)
def test_evaluation_propagates_availability_failure_while_live_returns_demo(
    feature, failure
):
    with pytest.raises(type(failure)):
        asyncio.run(
            analyze_image(
                b"photo",
                ScriptedInference([failure]),
                feature=feature,
                mode="evaluation",
            )
        )
    live = asyncio.run(
        analyze_image(
            b"photo", ScriptedInference([failure]), feature=feature, mode="live"
        )
    )
    assert live.analysis_source == "degraded_demo"


@pytest.mark.parametrize("feature", ["color", "body"])
def test_explicit_demo_does_not_call_ai(feature):
    fake = ScriptedInference([])
    result = asyncio.run(analyze_image(b"photo", fake, feature=feature, mode="demo"))
    assert result.analysis_source == "degraded_demo"
    assert fake.stages == []


def test_body_repair_and_clothing_best_effort_keep_four_call_order():
    from app.services.demo_cache import get_demo_body_result

    incomplete = {"shape": "Rectangle"}
    clothing = {"error_code": "CLOTHING_TOO_WIDE", "message": "Synthetic clothing"}
    fake = ScriptedInference([clothing, incomplete, clothing, get_demo_body_result()])
    result = asyncio.run(
        analyze_image(b"photo", fake, feature="body", mode="evaluation")
    )
    assert result.shape == "Rectangle"
    assert result.confidence == 0.45
    assert fake.stages == [
        "initial",
        "clothing_best_effort",
        "repair",
        "repair_clothing",
    ]


def test_failed_body_repair_stays_an_error():
    fake = ScriptedInference([{"shape": "Rectangle"}, {"shape": "Rectangle"}])
    with pytest.raises(AnalysisRejectedError) as caught:
        asyncio.run(analyze_image(b"photo", fake, feature="body", mode="evaluation"))
    assert caught.value.error_code == "AI_INCOMPLETE"
    assert fake.stages == ["initial", "repair"]


def test_image_read_failure_is_evidence_instead_of_dropping_case(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    case = load_dataset(root / "evals/synthetic/analysis.dataset.json").cases[0]
    row = asyncio.run(_run_case(case, ScriptedInference([]), tmp_path / "missing.png"))
    assert row["status"] == "error"
    assert row["error"]["type"] == "FileNotFoundError"
    assert row["input_image_sha256"] is None
    assert row["trace"]["calls"] == []


def test_trace_isolated_between_concurrent_tasks():
    async def capture(name):
        from app.services.analysis.tracing import trace_stage

        with capture_analysis_trace() as trace, trace_stage(name):
            trace.photo_quality["owner"] = name
            await asyncio.sleep(0)
            assert current_stage() == name
            return trace

    async def concurrent():
        return await asyncio.gather(capture("first"), capture("second"))

    first, second = asyncio.run(concurrent())
    assert first.photo_quality["owner"] == "first"
    assert second.photo_quality["owner"] == "second"


def test_preprocessing_failure_counts_as_a_failed_case_without_ai(monkeypatch):
    from pathlib import Path

    from app.evaluation.metrics import operational_metrics
    from app.utils.image_utils import preprocess_image_for_ai

    monkeypatch.setattr(
        "app.services.analysis.workflows.preprocess_image_for_ai",
        preprocess_image_for_ai,
    )
    root = Path(__file__).resolve().parents[1]
    case = load_dataset(root / "evals/synthetic/analysis.dataset.json").cases[0]
    fake = ScriptedInference([])
    row = asyncio.run(_run_case(case, fake, b"not an image"))
    assert row["error"]["code"] == "PHOTO_UNREADABLE"
    assert fake.stages == []
    metrics = operational_metrics([row])
    assert metrics["case_count"] == 1
    assert metrics["final_success_rate"] == 0
    assert metrics["first_pass_schema_valid_rate"] == 0
