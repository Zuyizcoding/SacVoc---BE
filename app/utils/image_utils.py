"""
Tiện ích xử lý ảnh trước khi gửi Gemini.

Tuân thủ:
  - Tính năng #8: validate ≥400×400px, compress ≤1MB, convert JPEG
  - KHÔNG import cv2 — dùng Pillow thay thế (nhẹ hơn ~600MB RAM)
"""
import io
import base64

from PIL import Image
from fastapi import HTTPException


# ──────────────────────────────────────────────
# Validate
# ──────────────────────────────────────────────
def validate_image(file_bytes: bytes) -> Image.Image:
    """Kiểm tra ảnh hợp lệ, kích thước ≥400×400px.

    Raises:
        HTTPException 422: Ảnh không hợp lệ hoặc quá nhỏ.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="File không phải ảnh hợp lệ. Vui lòng chọn file JPEG/PNG.",
        )

    width, height = img.size
    if width < 400 or height < 400:
        raise HTTPException(
            status_code=422,
            detail=f"Ảnh quá nhỏ ({width}×{height}). Yêu cầu tối thiểu 400×400px.",
        )

    return img


# ──────────────────────────────────────────────
# Compress
# ──────────────────────────────────────────────
def compress_image(file_bytes: bytes, max_kb: int = 1000) -> bytes:
    """Compress ảnh về ≤ max_kb KB, convert sang JPEG.

    Gemini nhận tốt nhất ≤5MB nhưng nhỏ hơn = nhanh hơn.
    JPEG nhỏ hơn ~60% so với PNG cùng chất lượng.

    Returns:
        bytes: Ảnh JPEG đã compress.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    output = io.BytesIO()
    quality = 85

    while quality > 20:
        output.seek(0)
        output.truncate()
        img.save(output, format="JPEG", quality=quality)
        if output.tell() <= max_kb * 1024:
            break
        quality -= 10

    return output.getvalue()


# ──────────────────────────────────────────────
# Base64 (cho Gemini inline data)
# ──────────────────────────────────────────────
def to_base64(file_bytes: bytes) -> str:
    """Chuyển bytes sang base64 string để gửi Gemini inline."""
    return base64.b64encode(file_bytes).decode("utf-8")
