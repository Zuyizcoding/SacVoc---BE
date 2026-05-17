"""
Gemini API Client — wrapper dùng chung cho cả F1 (Color) và F2 (Body).

Tuân thủ:
  - backend-python #2: Dependency Injection — inject qua Depends()
  - backend-python #4: API key từ Settings (pydantic-settings)
    - Tính năng #10: retry 3 lần, timeout theo GEMINI_TIMEOUT
  - Tính năng #11: response_schema ép Gemini trả đúng JSON
  - Tính năng #12: Safety Settings BLOCK_NONE cho ảnh body
  - Tính năng #13: Circuit Breaker — >3 lỗi trong 2 phút → fallback

LƯU Ý QUAN TRỌNG:
  - KHÔNG đặt temperature < 1.0 (thinking model giảm hiệu suất)
  - PHẢI dùng response_schema thay vì regex parse JSON
"""
import json
import re
import time
import asyncio
import random
import logging
import inspect
from copy import deepcopy
from typing import Type

import google.generativeai as genai
from fastapi import HTTPException
from google.api_core.exceptions import (
    DeadlineExceeded,
    InternalServerError,
    ResourceExhausted,
    ServiceUnavailable,
    TooManyRequests,
)
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)


AI_OVERLOADED_DETAIL = (
    "Hệ thống AI đang quá tải hoặc hết lượt sử dụng. Vui lòng thử lại sau!"
)


def _raise_ai_overloaded_http() -> None:
    raise HTTPException(status_code=429, detail=AI_OVERLOADED_DETAIL)


def _deref_json_schema_refs(schema: object, defs: dict[str, object]) -> object:
    """Resolve internal Pydantic `$ref` that point to `#/$defs/...`.

    google-generativeai expects a simplified schema object and doesn't understand
    `$ref`/`$defs`. We inline referenced definitions to keep the schema self-contained.
    """

    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.split("/")[-1]
            if name in defs:
                return _deref_json_schema_refs(defs[name], defs)

        return {k: _deref_json_schema_refs(v, defs) for k, v in schema.items() if k != "$defs"}

    if isinstance(schema, list):
        return [_deref_json_schema_refs(item, defs) for item in schema]

    return schema


def _sanitize_schema_for_genai(schema: object, *, _in_properties_map: bool = False) -> object:
    """Strip JSONSchema keywords unsupported by google-generativeai Schema.

    The SDK's `Schema` proto doesn't accept constraints like `maximum`, `minimum`,
    `pattern`, etc. Keeping only a small subset avoids `Unknown field for Schema`.
    """

    allowed_keys = {
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "description",
        "format",
        "nullable",
    }

    if isinstance(schema, dict):
        # When we're inside a `properties` object, keys are *field names* and must
        # be preserved verbatim.
        if _in_properties_map:
            return {k: _sanitize_schema_for_genai(v) for k, v in schema.items()}

        sanitized: dict = {}
        for k, v in schema.items():
            if k not in allowed_keys:
                continue

            if k == "properties" and isinstance(v, dict):
                sanitized[k] = _sanitize_schema_for_genai(v, _in_properties_map=True)
            else:
                sanitized[k] = _sanitize_schema_for_genai(v)

        # Defensive: if required mentions fields not present in properties,
        # drop required to avoid INVALID_ARGUMENT from the API.
        try:
            req = sanitized.get("required")
            props = sanitized.get("properties")
            if isinstance(req, list) and isinstance(props, dict):
                if any((not isinstance(name, str)) or (name not in props) for name in req):
                    sanitized.pop("required", None)
        except Exception:
            pass

        return sanitized

    if isinstance(schema, list):
        return [_sanitize_schema_for_genai(item) for item in schema]

    return schema


def _pydantic_model_to_genai_response_schema(model: Type[BaseModel]) -> dict:
    """Convert a Pydantic model into a genai-compatible response_schema dict."""

    raw = deepcopy(model.model_json_schema())
    defs = raw.get("$defs") or {}
    deref = _deref_json_schema_refs(raw, defs)
    sanitized = _sanitize_schema_for_genai(deref)
    assert isinstance(sanitized, dict)
    return sanitized


# ──────────────────────────────────────────────
# Circuit Breaker (Tính năng #13)
# ──────────────────────────────────────────────
class CircuitBreaker:
    """Sau >3 lỗi liên tiếp trong 2 phút → fail-fast.

    Mục tiêu: tránh spam gọi Gemini khi upstream đang lỗi/quá tải.
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


# ──────────────────────────────────────────────
# JSON Parser Fallback
# ──────────────────────────────────────────────
def parse_gemini_json(text: str) -> dict:
    """Trích xuất JSON từ Gemini response.

    Khi dùng response_schema, Gemini LUÔN trả JSON hợp lệ.
    Hàm này là fallback cưỡng bức cho edge case cực hiếm.
    """
    # Happy path — response_schema đảm bảo JSON hợp lệ
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback 1: strip markdown code block ```json...```
    match = re.search(r"```json\s*([\s\S]*?)```", text)
    if match:
        return json.loads(match.group(1))

    # Fallback 2: tìm {...} bất kỳ
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Gemini không trả về JSON hợp lệ")


# ──────────────────────────────────────────────
# Gemini Client chính
# ──────────────────────────────────────────────
class GeminiClient:
    """Wrapper gọi Gemini 2.5 Flash Vision API.

    Sử dụng:
        client = GeminiClient()
        result = await client.analyze(image_bytes, prompt, ColorGeminiResponse)
    """

    def __init__(self) -> None:
        # Cấu hình API key từ Settings (backend-python #4)
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self._breaker = CircuitBreaker(
            threshold=settings.GEMINI_MAX_RETRIES,
            reset_after=120,
        )

    def _build_model(
        self,
        response_schema: Type[BaseModel],
        system_instruction: str,
    ) -> genai.GenerativeModel:
        """Tạo model instance với response_schema + safety settings.

        Tính năng #11: response_schema ép JSON — không bao giờ JSONDecodeError.
        Tính năng #12: BLOCK_NONE cho ảnh toàn thân.
        """
        # google-generativeai SDK versions differ. Some builds don't expose
        # genai.types.ThinkingConfig / GenerationConfig.thinking_config.
        gen_config_kwargs: dict = {
            "response_mime_type": "application/json",
            # KHÔNG đặt temperature — giữ mặc định 1.0
            # thinking models giảm hiệu suất ở temperature < 1.0
        }

        # Structured output schema:
        # - Pydantic emits JSONSchema with keywords like `maximum`/`minimum`.
        # - google-generativeai Schema rejects those keywords.
        # => sanitize to a minimal, SDK-compatible schema; fallback if unsupported.
        try:
            gen_config_kwargs["response_schema"] = _pydantic_model_to_genai_response_schema(
                response_schema
            )
        except Exception as exc:
            logger.warning("response_schema sanitization failed; continuing without it: %s", str(exc))

        try:
            gen_config_sig = inspect.signature(genai.GenerationConfig)
            supports_thinking_config = "thinking_config" in gen_config_sig.parameters
        except Exception:
            supports_thinking_config = False

        if supports_thinking_config and hasattr(getattr(genai, "types", None), "ThinkingConfig"):
            try:
                gen_config_kwargs["thinking_config"] = genai.types.ThinkingConfig(
                    thinking_budget=settings.GEMINI_THINKING_BUDGET,
                )
            except Exception as exc:
                logger.warning("ThinkingConfig unsupported by current SDK: %s", str(exc))

        try:
            generation_config = genai.GenerationConfig(**gen_config_kwargs)
        except Exception as exc:
            # As a last resort, disable response_schema.
            logger.warning("GenerationConfig rejected response_schema; disabling structured schema: %s", str(exc))
            gen_config_kwargs.pop("response_schema", None)
            generation_config = genai.GenerationConfig(**gen_config_kwargs)

        return genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            # --- System Instruction: tách role ra khỏi user prompt ---
            system_instruction=system_instruction,
            # --- Generation Config ---
            generation_config=generation_config,
            # --- Safety Settings (Tính năng #12) ---
            safety_settings={
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
                    HarmBlockThreshold.BLOCK_NONE
                ),
                HarmCategory.HARM_CATEGORY_HARASSMENT: (
                    HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
            },
        )

    async def analyze(
        self,
        image_bytes: bytes,
        prompt: str,
        response_schema: Type[BaseModel],
        system_instruction: str,
    ) -> dict:
        """Gửi ảnh + prompt đến Gemini, trả về dict đã validate.

        Args:
            image_bytes: Ảnh đã compress (≤1MB JPEG).
            prompt: User prompt (không chứa role — role nằm trong system_instruction).
            response_schema: Pydantic model ép output JSON.
            system_instruction: Persona + format rules (chạy 1 lần).

        Returns:
            dict chứa kết quả phân tích đã validate.

        Raises:
            HTTPException: 429 khi Gemini bị quota/overload hoặc circuit breaker mở.
            ValueError: Khi Gemini không trả JSON hợp lệ.
        """
        # --- Circuit Breaker check (Tính năng #13) ---
        if self._breaker.is_open:
            logger.warning("⚡ Circuit breaker OPEN — fail-fast 429")
            _raise_ai_overloaded_http()

        model = self._build_model(response_schema, system_instruction)

        # --- Chuẩn bị image part ---
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_bytes,
        }

        # --- Retry logic (Tính năng #10) ---
        last_exception: Exception | None = None
        for attempt in range(1, settings.GEMINI_MAX_RETRIES + 1):
            try:
                logger.info(
                    "🤖 Gemini call attempt %d/%d",
                    attempt,
                    settings.GEMINI_MAX_RETRIES,
                )
                response = await asyncio.to_thread(
                    model.generate_content,
                    [image_part, prompt],
                    request_options={"timeout": settings.GEMINI_TIMEOUT},
                )

                # Parse response
                result = parse_gemini_json(response.text)

                # Validate bằng Pydantic schema
                validated = response_schema.model_validate(result)

                # Ghi nhận thành công — reset circuit breaker
                self._breaker.record_success()
                logger.info("✅ Gemini trả kết quả thành công")

                return validated.model_dump()

            except (ResourceExhausted, TooManyRequests) as exc:
                # Quota exceeded / rate limited (HTTP 429) → không retry.
                last_exception = exc
                self._breaker.record_error()
                logger.warning(
                    "❌ Gemini quota/rate-limit (429): %s",
                    str(exc),
                )
                _raise_ai_overloaded_http()

            except InternalServerError as exc:
                # Upstream 500 — có thể transient, retry theo policy; nếu hết retry → map 429 cho UX.
                last_exception = exc
                self._breaker.record_error()
                logger.warning(
                    "❌ Gemini internal error (500) attempt %d/%d: %s",
                    attempt,
                    settings.GEMINI_MAX_RETRIES,
                    str(exc),
                )

                if attempt < settings.GEMINI_MAX_RETRIES:
                    backoff = min(2.0 ** (attempt - 1), 8.0) + random.uniform(0.0, 0.5)
                    await asyncio.sleep(backoff)
                    continue
                _raise_ai_overloaded_http()

            except (DeadlineExceeded, ServiceUnavailable) as exc:
                last_exception = exc
                self._breaker.record_error()
                logger.warning(
                    "❌ Gemini attempt %d failed (transient): %s",
                    attempt,
                    str(exc),
                )

                # If the breaker opened due to repeated failures, fail fast.
                if self._breaker.is_open:
                    _raise_ai_overloaded_http()

                # Exponential-ish backoff with small jitter.
                if attempt < settings.GEMINI_MAX_RETRIES:
                    backoff = min(2.0 ** (attempt - 1), 8.0) + random.uniform(0.0, 0.5)
                    await asyncio.sleep(backoff)
                continue

            except Exception as exc:
                last_exception = exc
                self._breaker.record_error()
                logger.warning(
                    "❌ Gemini attempt %d failed: %s", attempt, str(exc)
                )

                if self._breaker.is_open:
                    _raise_ai_overloaded_http()

        # Hết retry — raise exception cuối cùng
        if self._breaker.is_open:
            _raise_ai_overloaded_http()

        if isinstance(last_exception, (InternalServerError, ResourceExhausted, TooManyRequests)):
            _raise_ai_overloaded_http()
        raise last_exception  # type: ignore[misc]


# ──────────────────────────────────────────────
# Dependency Injection (backend-python #2)
# ──────────────────────────────────────────────
# Singleton instance — inject vào endpoint qua Depends()
_gemini_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    """FastAPI Dependency — inject GeminiClient vào router.

    Sử dụng trong endpoint:
        @router.post("/color")
        async def analyze_color(
            client: GeminiClient = Depends(get_gemini_client),
        ):
    """
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client
