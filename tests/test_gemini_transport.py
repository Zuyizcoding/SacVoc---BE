"""Exercise the real pinned SDK over fake HTTP, not a replacement parser."""

import asyncio
import base64
import json

import httpx
import pytest
from google import genai
from google.genai import types

from app.core.config import settings
from app.evaluation.schemas import SyntheticReply
from app.evaluation.synthetic import scripted_sdk
from app.schemas.body import BodyGeminiResponse
from app.schemas.color import ColorInferenceResponse
from app.services.analysis.errors import GeminiOverloadedError
from app.services.analysis.tracing import capture_analysis_trace
from app.services.gemini_client import (
    CircuitBreaker,
    GeminiClient,
    GeminiInvalidResponseError,
    GeminiSafetyBlockedError,
)


@pytest.fixture(autouse=True)
def no_real_wait(monkeypatch):
    async def sleep(_seconds):
        pass

    monkeypatch.setattr("app.services.gemini_client.asyncio.sleep", sleep)
    monkeypatch.setattr("app.services.gemini_client.random.uniform", lambda *_args: 0)


def run(client, schema=ColorInferenceResponse):
    return asyncio.run(client.analyze(b"jpeg", "user prompt", schema, "system prompt"))


def test_real_sdk_serializes_image_schema_timeout_and_single_attempt(monkeypatch):
    requests = []
    options = []
    original_constructor = genai.Client

    def handle(request):
        requests.append(request)
        return httpx.Response(429, json={"error": {"code": 429, "message": "quota"}})

    def constructor(*, api_key, http_options, vertexai):
        assert vertexai is False
        options.append(http_options)
        http_options.client_args = {"transport": httpx.MockTransport(handle)}
        return original_constructor(
            api_key=api_key, http_options=http_options, vertexai=vertexai
        )

    monkeypatch.setattr("app.services.gemini_client.genai.Client", constructor)
    client = GeminiClient()
    try:
        with pytest.raises(GeminiOverloadedError):
            run(client, BodyGeminiResponse)
    finally:
        client.close()

    assert options[0].timeout == settings.GEMINI_TIMEOUT * 1000
    assert options[0].retry_options.attempts == 1
    assert len(requests) == settings.GEMINI_MAX_RETRIES
    assert requests[0].extensions["timeout"]["read"] == settings.GEMINI_TIMEOUT
    wire = json.loads(requests[0].content)
    parts = wire["contents"][0]["parts"]
    assert parts[0]["inlineData"] == {
        "data": base64.b64encode(b"jpeg").decode(),
        "mimeType": "image/jpeg",
    }
    assert parts[1]["text"] == "user prompt"
    assert wire["systemInstruction"]["parts"][0]["text"] == "system prompt"
    config = wire["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    # The pinned SDK emits the original protobuf field name for thinking budget.
    assert (
        config["thinkingConfig"]["thinking_budget"] == settings.GEMINI_THINKING_BUDGET
    )
    properties = config["responseSchema"]["properties"]
    assert {"shape", "measurements", "styling_tips"} <= properties.keys()
    assert "shoulder_relative" in properties["measurements"]["properties"]
    assert properties["confidence"]["minimum"] == 0
    assert properties["confidence"]["maximum"] == 1
    assert wire["safetySettings"] == [
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    ]


def test_trace_retains_malformed_reply_then_valid_json_and_usage():
    client = GeminiClient(
        scripted_sdk(
            [
                SyntheticReply(raw_text="not-json"),
                SyntheticReply(payload={"sub_season": "Soft Autumn"}),
            ]
        )
    )
    try:
        with capture_analysis_trace() as trace:
            assert run(client)["sub_season"] == "Soft Autumn"
    finally:
        client.close()
    first, second = trace.calls[0].attempts
    assert first.raw_text == "not-json"
    assert first.error_type == "GeminiInvalidResponseError"
    assert first.retry_delay_seconds == 0.75
    assert not first.schema_valid
    assert second.schema_valid
    assert second.usage["total_token_count"] == 15
    assert second.model_version == "synthetic-not-a-real-model"
    assert trace.calls[0].validated_result["sub_season"] == "Soft Autumn"
    assert not client._breaker.is_open


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_api_errors_are_not_retried_or_counted(monkeypatch, status):
    client = GeminiClient(scripted_sdk([SyntheticReply(http_status=status)]))
    try:
        with capture_analysis_trace() as trace, pytest.raises(Exception):
            run(client)
    finally:
        client.close()
    assert len(trace.calls[0].attempts) == 1
    assert trace.calls[0].attempts[0].http_status == status
    assert not client._breaker.is_open


@pytest.mark.parametrize("finish_reason", ["SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"])
def test_blocked_output_never_retries_or_opens_breaker(finish_reason):
    client = GeminiClient(
        scripted_sdk([SyntheticReply(raw_text="", finish_reason=finish_reason)])
    )
    try:
        with capture_analysis_trace() as trace, pytest.raises(GeminiSafetyBlockedError):
            run(client)
    finally:
        client.close()
    assert len(trace.calls[0].attempts) == 1
    assert not client._breaker.is_open


@pytest.mark.parametrize(
    "raw",
    ["", "[]", "null", '{"sub_season": null}', '{"confidence": 2}', '{"sub_season":'],
)
def test_invalid_replies_exhaust_attempts_without_opening_breaker(raw):
    client = GeminiClient(scripted_sdk([SyntheticReply(raw_text=raw)] * 2))
    try:
        with pytest.raises(GeminiInvalidResponseError):
            run(client)
    finally:
        client.close()
    assert not client._breaker.is_open


@pytest.mark.parametrize(
    "failure", [httpx.ReadTimeout("timeout"), httpx.ConnectError("offline")]
)
def test_httpx_transport_errors_retry_with_existing_budget(monkeypatch, failure):
    client = GeminiClient(scripted_sdk([]))
    calls = []

    async def fail(*_args):
        calls.append(None)
        raise failure

    monkeypatch.setattr(client, "_generate_once", fail)
    try:
        with capture_analysis_trace() as trace, pytest.raises(GeminiOverloadedError):
            run(client)
    finally:
        client.close()
    assert len(calls) == 2
    assert client._breaker.is_open
    assert trace.calls[0].attempts[0].retry_delay_seconds == 1


def test_breaker_resets_after_cooldown_and_success(monkeypatch):
    now = [1000]
    monkeypatch.setattr("app.services.gemini_client.time.time", lambda: now[0])
    breaker = CircuitBreaker(threshold=2, reset_after=120)
    breaker.record_error()
    breaker.record_error()
    assert breaker.is_open
    now[0] += 121
    assert not breaker.is_open
    breaker.record_error()
    assert not breaker.is_open
    breaker.record_success()
    breaker.record_error()
    assert not breaker.is_open


def test_prompt_feedback_block_is_not_a_formatting_error():
    response = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason="SAFETY"
        )
    )
    with pytest.raises(GeminiSafetyBlockedError):
        GeminiClient._read_result(response, ColorInferenceResponse)


def test_unspecified_prompt_feedback_does_not_reject_a_valid_reply():
    response = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason="BLOCKED_REASON_UNSPECIFIED"
        ),
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[types.Part.from_text(text='{"sub_season":"Soft Autumn"}')]
                )
            )
        ],
    )
    assert (
        GeminiClient._read_result(response, ColorInferenceResponse)["sub_season"]
        == "Soft Autumn"
    )


def test_missing_key_fails_before_sdk_construction(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClient()


def test_trace_context_restores_after_exception():
    from app.services.analysis.tracing import current_stage, current_trace, trace_stage

    assert current_trace() is None
    with pytest.raises(RuntimeError):
        with capture_analysis_trace(), trace_stage("repair"):
            assert current_stage() == "repair"
            raise RuntimeError("test")
    assert current_trace() is None
    assert current_stage() == "initial"
