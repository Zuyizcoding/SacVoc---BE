"""Đọc cấu hình môi trường bằng Pydantic Settings."""

from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Đọc biến môi trường từ file .env."""

    # --- Gemini ---
    # Offline evaluation/schema tools can import settings without credentials.
    # GeminiClient validates the key before constructing a real SDK transport.
    GEMINI_API_KEY: str = Field(default="", repr=False)
    GEMINI_MODEL: str = "gemini-2.5-flash"
    # Safe default cho free tier: fail nhanh hơn nhưng vẫn đủ thời gian cho vision.
    GEMINI_TIMEOUT: int = Field(default=30, gt=0)
    # Historical env name: total attempts per call, not additional retries.
    GEMINI_MAX_RETRIES: int = Field(default=2, ge=1)
    GEMINI_THINKING_BUDGET: int = 1024  # token cho suy luận nội bộ

    # --- App ---
    APP_NAME: str = "Sắc Vóc API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # --- Database ---
    DATABASE_URL: str = "sqlite:///./sac_voc.db"

    # --- F3 Recommendation ---
    F3_QUERY_VECTORS_PATH: str = "data/query_vectors.json"
    F3_RUNTIME_TEXT_ENCODER_MODEL: str = "fashion-clip"

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> Any:
        """Cho phép giá trị DEBUG phổ biến ngoài true/false."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {
                "1",
                "true",
                "t",
                "yes",
                "y",
                "on",
                "debug",
                "development",
                "dev",
            }:
                return True
            if normalized in {
                "0",
                "false",
                "f",
                "no",
                "n",
                "off",
                "release",
                "prod",
                "production",
            }:
                return False
        return value

    model_config = {"env_file": ".env", "extra": "ignore"}


# Singleton — import settings ở bất kỳ đâu
settings = Settings()
