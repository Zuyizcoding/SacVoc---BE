"""Số lượt gọi, backoff và circuit breaker là hành vi cần giữ khi refactor."""

import asyncio
import json

import pytest
from google.api_core import exceptions as google_errors

from app.core.config import settings
from app.schemas.color import ColorInferenceResponse
from app.services.analysis.errors import GeminiOverloadedError
from app.services.gemini_client import GeminiClient, GeminiInvalidResponseError


class ScriptedModel:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = 0

    def generate_content(self, *args, **kwargs):
        self.calls += 1
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return type("Response", (), {"text": json.dumps(result)})()


def run_client(monkeypatch, results, *, initial_errors=0):
    monkeypatch.setattr(settings, "GEMINI_MAX_RETRIES", 2)
    client = GeminiClient()
    model = ScriptedModel(results)
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(client, "_build_model", lambda *args: model)
    monkeypatch.setattr("app.services.gemini_client.asyncio.sleep", sleep)
    monkeypatch.setattr("app.services.gemini_client.random.uniform", lambda *args: 0)
    for _ in range(initial_errors):
        client._breaker.record_error()
    return client, model, delays


def analyze(client):
    return asyncio.run(
        client.analyze(b"jpeg", "prompt", ColorInferenceResponse, "system")
    )


@pytest.mark.parametrize(
    "failure,delay",
    [
        (google_errors.ResourceExhausted("quota"), 1.5),
        (google_errors.ServiceUnavailable("busy"), 1.0),
        (google_errors.InternalServerError("server"), 1.0),
        (TimeoutError("timeout"), 1.0),
        ({}, 0.75),
    ],
)
def test_retry_then_success_preserves_delay_and_resets_breaker(
    monkeypatch, failure, delay
):
    client, model, delays = run_client(
        monkeypatch, [failure, {"sub_season": "Soft Autumn"}]
    )
    assert analyze(client)["sub_season"] == "Soft Autumn"
    assert model.calls == 2
    assert delays == [delay]
    assert not client._breaker.is_open


def test_busy_error_stops_when_breaker_opens_mid_loop(monkeypatch):
    client, model, delays = run_client(
        monkeypatch, [google_errors.ServiceUnavailable("busy")], initial_errors=1
    )
    with pytest.raises(GeminiOverloadedError):
        analyze(client)
    assert model.calls == 1
    assert delays == []


def test_quota_keeps_remaining_attempt_even_when_breaker_opens(monkeypatch):
    client, model, _ = run_client(
        monkeypatch,
        [google_errors.ResourceExhausted("quota"), {"sub_season": "Soft Autumn"}],
        initial_errors=1,
    )
    assert analyze(client)["sub_season"] == "Soft Autumn"
    assert model.calls == 2


def test_exhausted_quota_opens_breaker_and_next_request_fails_fast(monkeypatch):
    client, model, _ = run_client(
        monkeypatch, [google_errors.ResourceExhausted("quota")] * 2
    )
    with pytest.raises(GeminiOverloadedError):
        analyze(client)
    with pytest.raises(GeminiOverloadedError):
        analyze(client)
    assert model.calls == 2


def test_invalid_output_uses_attempts_without_opening_breaker(monkeypatch):
    client, model, delays = run_client(monkeypatch, [{}, {}])
    with pytest.raises(GeminiInvalidResponseError):
        analyze(client)
    assert model.calls == 2
    assert delays == [0.75]
    assert not client._breaker.is_open
