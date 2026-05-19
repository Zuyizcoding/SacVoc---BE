"""
Cấu hình ứng dụng — load từ .env.
Tuân thủ: backend-python #4 — pydantic-settings, không hard-code key.
"""
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Đọc biến môi trường từ file .env."""

    # --- Gemini ---
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TIMEOUT: int = 60            # giây (vision requests thường chậm hơn)
    GEMINI_MAX_RETRIES: int = 3
    GEMINI_THINKING_BUDGET: int = 1024  # token cho suy luận nội bộ

    # --- App ---
    APP_NAME: str = "Sắc Vóc API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # --- Database ---
    DATABASE_URL: str = "sqlite:///./sac_voc.db"

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> Any:
        """Cho phép giá trị DEBUG phổ biến ngoài true/false."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {
                "1", "true", "t", "yes", "y", "on",
                "debug", "development", "dev",
            }:
                return True
            if normalized in {
                "0", "false", "f", "no", "n", "off",
                "release", "prod", "production",
            }:
                return False
        return value

    model_config = {"env_file": ".env", "extra": "ignore"}


# Singleton — import settings ở bất kỳ đâu
settings = Settings()
