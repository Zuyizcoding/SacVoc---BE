"""
Gemini API Client — wrapper dùng chung cho cả F1 (Color) và F2 (Body).

Tuân thủ:
  - backend-python #2: Dependency Injection — inject qua Depends()
  - backend-python #4: API key từ Settings (pydantic-settings)
  - Tính năng #10: retry 3 lần, timeout 15s
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
import logging
from typing import Type

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Circuit Breaker (Tính năng #13)
# ──────────────────────────────────────────────
class CircuitBreaker:
    """Sau >3 lỗi liên tiếp trong 2 phút → chuyển sang demo cache.
    Tránh loading vô tận lúc demo khi Gemini bị sự cố.
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
        """True = circuit mở → dùng demo cache thay vì gọi Gemini."""
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
        return genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            # --- System Instruction: tách role ra khỏi user prompt ---
            system_instruction=system_instruction,
            # --- Generation Config ---
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema,   # Tính năng #11
                thinking_config=genai.types.ThinkingConfig(
                    thinking_budget=settings.GEMINI_THINKING_BUDGET,  # 1024 token
                ),
                # KHÔNG đặt temperature — giữ mặc định 1.0
                # thinking models giảm hiệu suất ở temperature < 1.0
            ),
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
            RuntimeError: Khi circuit breaker mở (>3 lỗi trong 2 phút).
            ValueError: Khi Gemini không trả JSON hợp lệ.
        """
        # --- Circuit Breaker check (Tính năng #13) ---
        if self._breaker.is_open:
            logger.warning("⚡ Circuit breaker OPEN — trả demo cache")
            raise RuntimeError("CIRCUIT_BREAKER_OPEN")

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
                response = model.generate_content(
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

            except Exception as exc:
                last_exception = exc
                self._breaker.record_error()
                logger.warning(
                    "❌ Gemini attempt %d failed: %s", attempt, str(exc)
                )

        # Hết retry — raise exception cuối cùng
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
