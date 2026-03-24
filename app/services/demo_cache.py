"""
Demo Cache System (Tính năng #14).

Lưu 5 bộ JSON kết quả cứng — khi Circuit Breaker mở,
service layer đọc cache này thay vì gọi Gemini.

Tuân thủ: architecture-standard #4 — không để hệ thống "chết" khi service con lỗi.
"""
import json
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Thư mục chứa JSON kết quả mẫu
CACHE_DIR = Path(__file__).parent.parent / "assets" / "demo_cache"


def get_demo_color_result() -> dict:
    """Trả kết quả Color mẫu từ demo cache.

    File mẫu: assets/demo_cache/color_soft_autumn.json
    """
    cache_files = list(CACHE_DIR.glob("color_*.json"))
    if not cache_files:
        logger.error("❌ Không tìm thấy file demo cache color")
        return _fallback_color()

    chosen = random.choice(cache_files)
    logger.info("📦 Dùng demo cache: %s", chosen.name)
    with open(chosen, "r", encoding="utf-8") as f:
        return json.load(f)


def get_demo_body_result() -> dict:
    """Trả kết quả Body mẫu từ demo cache."""
    cache_files = list(CACHE_DIR.glob("body_*.json"))
    if not cache_files:
        logger.error("❌ Không tìm thấy file demo cache body")
        return _fallback_body()

    chosen = random.choice(cache_files)
    logger.info("📦 Dùng demo cache: %s", chosen.name)
    with open(chosen, "r", encoding="utf-8") as f:
        return json.load(f)


def _fallback_color() -> dict:
    """Kết quả mặc định cứng — dùng khi không có file cache."""
    return {
        "sub_season": "Soft Autumn",
        "primary_season": "Autumn",
        "confidence": 0.80,
        "secondary_season": "True Autumn",
        "explanation": "Đây là kết quả mẫu. Vui lòng thử lại sau.",
        "palette": ["#8B4513", "#D2691E", "#A0522D", "#CD853F", "#DEB887", "#F4A460"],
        "makeup": {
            "lip": ["Terracotta", "Brick Red", "Nude Warm"],
            "eye": ["Bronze", "Brown", "Copper"],
            "hair": ["Chestnut", "Auburn", "Dark Caramel"],
        },
        "advice": "Đây là kết quả mẫu — hệ thống AI tạm thời không khả dụng.",
    }


def _fallback_body() -> dict:
    """Kết quả mặc định cứng — dùng khi không có file cache."""
    return {
        "shape": "Hourglass",
        "confidence": 0.80,
        "measurements": {
            "shoulder_relative": "balanced",
            "hip_relative": "balanced",
            "waist_definition": "defined",
        },
        "styling_tips": [
            "Chọn đồ ôm vừa vặn để tôn đường cong",
            "Váy chữ A luôn phù hợp",
            "Thắt lưng giúp nhấn eo",
        ],
        "avoid": [
            "Đồ quá rộng che mất đường cong",
            "Váy bút chì quá chật",
        ],
        "explanation": "Đây là kết quả mẫu — hệ thống AI tạm thời không khả dụng.",
    }
