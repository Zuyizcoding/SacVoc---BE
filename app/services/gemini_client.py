"""Gemini client: gọi một lần, đọc kết quả, phân loại lỗi rồi thử lại."""

import asyncio
import logging
import random
import time
from typing import Type

import google.generativeai as genai
from google.api_core import exceptions as gapi_exceptions
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.services.analysis.errors import GeminiOverloadedError
from app.services.gemini_schema import build_gemini_model, parse_gemini_json

logger = logging.getLogger(__name__)
AI_OVERLOADED_DETAIL = "Hệ thống AI đang quá tải hoặc đã chạm giới hạn free tier. Vui lòng thử lại sau khoảng 30-60 giây."
AI_INVALID_RESPONSE_DETAIL = (
    "AI phản hồi chưa đầy đủ hoặc sai định dạng. Vui lòng thử lại sau vài giây."
)


class GeminiInvalidResponseError(ValueError):
    """Structured output thiếu field hoặc sai định dạng; không mở breaker."""


class CircuitBreaker:
    """Fail-fast khi đủ threshold lỗi; reset sau reset_after giây yên lặng.

    GeminiClient truyền ngưỡng cấu hình vào, thay vì luôn dùng mặc định của lớp.
    """

    def __init__(self, threshold: int = 3, reset_after: int = 120):
        self._errors: int = 0
        self._last_error_time: float = 0.0
        self._threshold = threshold
        self._reset_after = reset_after  # giây

    def record_error(self) -> None:
        now = time.time()
        # Reset counter nếu đã qua khoảng reset
        if now - self._last_error_time > self._reset_after:
            self._errors = 0
        self._errors += 1
        self._last_error_time = now

    def record_success(self) -> None:
        self._errors = 0

    @property
    def is_open(self) -> bool:
        """True = circuit mở → không gọi Gemini (fail-fast)."""
        if self._errors >= self._threshold:
            # Tự reset sau khoảng reset
            if time.time() - self._last_error_time > self._reset_after:
                self._errors = 0
                return False
            return True
        return False


QUOTA_ERRORS = (gapi_exceptions.ResourceExhausted, gapi_exceptions.TooManyRequests)
BUSY_ERRORS = (
    gapi_exceptions.DeadlineExceeded,
    gapi_exceptions.ServiceUnavailable,
    gapi_exceptions.BadGateway,
    gapi_exceptions.GatewayTimeout,
)
RETRYABLE_ERRORS = (
    TimeoutError,
    ConnectionError,
    *QUOTA_ERRORS,
    *BUSY_ERRORS,
    gapi_exceptions.InternalServerError,
)


class GeminiClient:
    def __init__(self) -> None:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self._breaker = CircuitBreaker(
            threshold=settings.GEMINI_MAX_RETRIES, reset_after=120
        )

    def _build_model(self, response_schema: Type[BaseModel], system_instruction: str):
        return build_gemini_model(response_schema, system_instruction)

    async def _generate_once(self, model, image_bytes: bytes, prompt: str):
        # SDK đồng bộ chạy ở thread để không chặn event loop của FastAPI.
        return await asyncio.to_thread(
            model.generate_content,
            [{"mime_type": "image/jpeg", "data": image_bytes}, prompt],
            request_options={"timeout": settings.GEMINI_TIMEOUT},
        )

    @staticmethod
    def _read_result(response, schema: Type[BaseModel]) -> dict:
        try:
            return schema.model_validate(parse_gemini_json(response.text)).model_dump()
        except (ValueError, ValidationError) as exc:
            raise GeminiInvalidResponseError(AI_INVALID_RESPONSE_DETAIL) from exc

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        if isinstance(exc, GeminiInvalidResponseError):
            return min(0.75 * attempt, 1.5) + random.uniform(0.0, 0.25)
        if isinstance(exc, QUOTA_ERRORS):
            return min(1.5 * attempt, 4.0) + random.uniform(0.0, 0.5)
        limit = (
            8.0
            if isinstance(exc, (*BUSY_ERRORS, gapi_exceptions.InternalServerError))
            else 6
        )
        return min(2.0 ** (attempt - 1), limit) + random.uniform(0.0, 0.5)

    async def analyze(
        self,
        image_bytes: bytes,
        prompt: str,
        response_schema: Type[BaseModel],
        system_instruction: str,
    ) -> dict:
        if self._breaker.is_open:
            raise GeminiOverloadedError(AI_OVERLOADED_DETAIL)
        model = self._build_model(response_schema, system_instruction)
        # Tên biến môi trường cũ được giữ; giá trị là tổng số lần gọi SDK.
        max_attempts = settings.GEMINI_MAX_RETRIES
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._generate_once(model, image_bytes, prompt)
                result = self._read_result(response, response_schema)
                self._breaker.record_success()
                return result
            except Exception as exc:
                invalid = isinstance(exc, GeminiInvalidResponseError)
                retryable = isinstance(exc, RETRYABLE_ERRORS)
                if not invalid and not retryable:
                    raise
                if retryable:
                    self._breaker.record_error()
                logger.warning("Gemini attempt %d/%d: %s", attempt, max_attempts, exc)
                # Chỉ nhóm upstream busy dừng giữa vòng khi breaker vừa mở.
                if isinstance(exc, BUSY_ERRORS) and self._breaker.is_open:
                    raise GeminiOverloadedError(AI_OVERLOADED_DETAIL) from exc
                if attempt < max_attempts:
                    await asyncio.sleep(self._retry_delay(exc, attempt))
                    continue
                if invalid:
                    raise
                raise GeminiOverloadedError(AI_OVERLOADED_DETAIL) from exc
        raise RuntimeError("Gemini call failed with unknown error")


_gemini_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    """Một client trong mỗi process để circuit breaker giữ trạng thái qua request."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client
