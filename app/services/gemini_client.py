"""The application's single Gemini boundary: request, validate, retry, return.

Intuition: transport failures and invalid model output are different problems.
Transport failures count toward a circuit breaker (temporary fail-fast). Invalid
JSON is retried but never opens it. Workflows decide whether to repair missing
business details or present a demo; this module never invents analysis results.
"""

import asyncio
import json
import logging
import random
import time

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.services.analysis.errors import GeminiOverloadedError
from app.services.analysis.tracing import (
    AttemptTrace,
    ModelCallTrace,
    content_hash,
    current_stage,
    current_trace,
)
from app.services.gemini_schema import build_generation_config, parse_gemini_json

logger = logging.getLogger(__name__)
AI_OVERLOADED_DETAIL = "Há»‡ thá»‘ng AI Ä‘ang quÃ¡ táº£i hoáº·c Ä‘Ã£ cháº¡m giá»›i háº¡n free tier. Vui lÃ²ng thá»­ láº¡i sau khoáº£ng 30-60 giÃ¢y."
AI_INVALID_RESPONSE_DETAIL = "AI pháº£n há»“i chÆ°a Ä‘áº§y Ä‘á»§ hoáº·c sai Ä‘á»‹nh dáº¡ng. Vui lÃ²ng thá»­ láº¡i sau vÃ i giÃ¢y."
BUSY_STATUS_CODES = {504, 503, 502, 408}
RETRYABLE_STATUS_CODES = {429, 500, *BUSY_STATUS_CODES}


class GeminiInvalidResponseError(ValueError):
    """A reply cannot satisfy the inference schema; does not trip the breaker."""


class GeminiSafetyBlockedError(Exception):
    """The provider blocked input/output; retrying is inappropriate."""


class CircuitBreaker:
    """Remember recent infrastructure failures across requests in one process."""

    def __init__(self, threshold: int = 3, reset_after: int = 120):
        self._errors = 0
        self._last_error_time = 0.0
        self._threshold = threshold
        self._reset_after = reset_after

    def record_error(self) -> None:
        now = time.time()
        if now - self._last_error_time > self._reset_after:
            self._errors = 0
        self._errors += 1
        self._last_error_time = now

    def record_success(self) -> None:
        self._errors = 0

    @property
    def is_open(self) -> bool:
        if self._errors >= self._threshold:
            if time.time() - self._last_error_time > self._reset_after:
                self._errors = 0
                return False
            return True
        return False


def _http_status(exc: Exception) -> int | None:
    return exc.code if isinstance(exc, errors.APIError) else None


def _is_retryable(exc: Exception) -> bool:
    return _http_status(exc) in RETRYABLE_STATUS_CODES or isinstance(
        exc, (TimeoutError, ConnectionError, httpx.TransportError)
    )


class GeminiClient:
    """Use one SDK client per process; tests may inject a mock HTTP SDK client.

    GEMINI_MAX_RETRIES is a historical name: total attempts per logical call,
    not additional retries. SDK retries are disabled to avoid multiplying that
    budget. SDK timeout uses milliseconds; application settings use seconds.
    """

    def __init__(self, sdk_client: genai.Client | None = None) -> None:
        if sdk_client is None and not settings.GEMINI_API_KEY.strip():
            raise ValueError(
                "Configure GEMINI_API_KEY before using the Gemini provider"
            )
        self._sdk = sdk_client or genai.Client(
            vertexai=False,
            api_key=settings.GEMINI_API_KEY,
            http_options=types.HttpOptions(
                api_version="v1beta",
                timeout=settings.GEMINI_TIMEOUT * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self._breaker = CircuitBreaker(
            threshold=settings.GEMINI_MAX_RETRIES, reset_after=120
        )

    def close(self) -> None:
        self._sdk.close()

    async def _generate_once(
        self, config: types.GenerateContentConfig, image_bytes: bytes, prompt: str
    ) -> types.GenerateContentResponse:
        # A sync SDK call in a thread leaves FastAPI's event loop responsive and
        # avoids binding a shared async SDK client to one particular event loop.
        return await asyncio.to_thread(
            self._sdk.models.generate_content,
            model=settings.GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=config,
        )

    @staticmethod
    def _read_result(
        response: types.GenerateContentResponse, schema: type[BaseModel]
    ) -> dict:
        feedback = getattr(response, "prompt_feedback", None)
        if feedback and feedback.block_reason not in {
            None,
            "BLOCKED_REASON_UNSPECIFIED",
        }:
            raise GeminiSafetyBlockedError("Gemini blocked the input")
        for candidate in getattr(response, "candidates", None) or []:
            if candidate.finish_reason in {
                "SAFETY",
                "BLOCKLIST",
                "PROHIBITED_CONTENT",
                "SPII",
                "IMAGE_SAFETY",
            }:
                raise GeminiSafetyBlockedError("Gemini blocked the output")
        try:
            payload = parse_gemini_json(response.text or "")
            return schema.model_validate(payload).model_dump()
        except (ValueError, ValidationError) as exc:
            raise GeminiInvalidResponseError(AI_INVALID_RESPONSE_DETAIL) from exc

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        if isinstance(exc, GeminiInvalidResponseError):
            return min(0.75 * attempt, 1.5) + random.uniform(0.0, 0.25)
        if _http_status(exc) == 429:
            return min(1.5 * attempt, 4.0) + random.uniform(0.0, 0.5)
        limit = 8.0 if _http_status(exc) in {500, *BUSY_STATUS_CODES} else 6.0
        return min(2.0 ** (attempt - 1), limit) + random.uniform(0.0, 0.5)

    async def analyze(
        self,
        image_bytes: bytes,
        prompt: str,
        response_schema: type[BaseModel],
        system_instruction: str,
    ) -> dict:
        call = self._start_trace(prompt, system_instruction, response_schema)
        if self._breaker.is_open:
            raise GeminiOverloadedError(AI_OVERLOADED_DETAIL)
        config = build_generation_config(response_schema, system_instruction)
        max_attempts = settings.GEMINI_MAX_RETRIES
        for attempt_number in range(1, max_attempts + 1):
            attempt = AttemptTrace(number=attempt_number)
            if call is not None:
                call.attempts.append(attempt)
            started = time.perf_counter()
            try:
                response = await self._generate_once(config, image_bytes, prompt)
                self._record_response(attempt, response)
                result = self._read_result(response, response_schema)
                attempt.schema_valid = True
                if call is not None:
                    call.validated_result = result
                self._breaker.record_success()
                return result
            except Exception as exc:
                attempt.error_type = type(exc).__name__
                attempt.http_status = _http_status(exc) or attempt.http_status
                invalid = isinstance(exc, GeminiInvalidResponseError)
                retryable = _is_retryable(exc)
                if not invalid and not retryable:
                    raise
                if retryable:
                    self._breaker.record_error()
                logger.warning(
                    "Gemini attempt %d/%d failed (%s)",
                    attempt_number,
                    max_attempts,
                    type(exc).__name__,
                )
                # Busy responses can stop mid-loop; quota and transport failures
                # retain their existing opportunity to use the remaining attempt.
                if _http_status(exc) in BUSY_STATUS_CODES and self._breaker.is_open:
                    raise GeminiOverloadedError(AI_OVERLOADED_DETAIL) from exc
                if attempt_number == max_attempts:
                    if invalid:
                        raise
                    raise GeminiOverloadedError(AI_OVERLOADED_DETAIL) from exc
                attempt.retry_delay_seconds = self._retry_delay(exc, attempt_number)
            finally:
                # Provider + validation duration; backoff is measured separately.
                attempt.latency_ms = round((time.perf_counter() - started) * 1000, 3)
            await asyncio.sleep(attempt.retry_delay_seconds)
        raise RuntimeError("Gemini call failed without an attempt")

    @staticmethod
    def _start_trace(
        prompt: str, system_instruction: str, schema: type[BaseModel]
    ) -> ModelCallTrace | None:
        trace = current_trace()
        if trace is None:
            return None
        call = ModelCallTrace(
            stage=current_stage(),
            prompt=prompt,
            system_instruction=system_instruction,
            prompt_sha256=content_hash(system_instruction + "\n" + prompt),
            schema_name=schema.__name__,
            schema_sha256=content_hash(
                json.dumps(schema.model_json_schema(), sort_keys=True)
            ),
        )
        trace.calls.append(call)
        return call

    @staticmethod
    def _record_response(
        attempt: AttemptTrace, response: types.GenerateContentResponse
    ) -> None:
        attempt.http_status = 200
        attempt.raw_text = response.text
        attempt.model_version = getattr(response, "model_version", None)
        candidates = getattr(response, "candidates", None) or []
        if candidates and candidates[0].finish_reason:
            attempt.finish_reason = candidates[0].finish_reason.value
        feedback = getattr(response, "prompt_feedback", None)
        if feedback and feedback.block_reason:
            attempt.prompt_block_reason = feedback.block_reason.value
        usage = getattr(response, "usage_metadata", None)
        if usage:
            attempt.usage = {
                name: getattr(usage, name, None)
                for name in (
                    "prompt_token_count",
                    "candidates_token_count",
                    "thoughts_token_count",
                    "total_token_count",
                )
                if getattr(usage, name, None) is not None
            }


_gemini_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    """Share connection pooling and breaker state within one application process."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


def close_gemini_client() -> None:
    global _gemini_client
    if _gemini_client is not None:
        _gemini_client.close()
        _gemini_client = None
