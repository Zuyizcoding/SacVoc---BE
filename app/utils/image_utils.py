"""Image utilities for analysis requests.

Quick wins implemented:
- Soft gate instead of strict 400x400 rejection.
- Auto upscale + square padding before AI call.
- Return quality flags so caller can lower confidence and add a friendly warning.
"""

import base64
import io

from fastapi import HTTPException
from PIL import Image, ImageFilter, ImageStat


def validate_image(file_bytes: bytes) -> Image.Image:
    """Validate image format and hard floor resolution.

    We no longer block by 400x400. We only reject extremely tiny images.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Không đọc được ảnh. Vui lòng chọn ảnh rõ hơn hoặc thử ảnh khác.",
        )

    width, height = img.size
    short_side = min(width, height)
    area = width * height

    if short_side < 140 or area < 30000:
        raise HTTPException(
            status_code=422,
            detail=(
                "Ảnh quá nhỏ để hệ thống nhận diện chính xác. "
                "Vui lòng chọn ảnh rõ hơn, ưu tiên thấy rõ khuôn mặt/cơ thể."
            ),
        )

    return img


def _estimate_detail_score(img: Image.Image) -> float:
    """Estimate visual detail using edge intensity (cv2-free)."""
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(edges).mean[0])


def _normalize_for_ai(
    img: Image.Image,
    *,
    target_short_side: int = 512,
    max_side: int = 1600,
    pad_to_square: bool = True,
) -> Image.Image:
    """Resize with aspect ratio + optional square padding."""
    img = img.convert("RGB")
    width, height = img.size
    short_side = min(width, height)
    long_side = max(width, height)

    scale = 1.0
    if short_side < target_short_side:
        scale = max(scale, target_short_side / float(short_side))
    if long_side * scale > max_side:
        scale = min(scale, max_side / float(long_side))

    if abs(scale - 1.0) > 1e-6:
        resized = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        img = img.resize(resized, Image.Resampling.LANCZOS)

    if not pad_to_square:
        return img

    w2, h2 = img.size
    canvas_size = max(w2, h2)
    if w2 == h2:
        return img

    canvas = Image.new("RGB", (canvas_size, canvas_size), color=(245, 245, 245))
    left = (canvas_size - w2) // 2
    top = (canvas_size - h2) // 2
    canvas.paste(img, (left, top))
    return canvas


def _compress_pil_image(img: Image.Image, max_kb: int = 1000) -> bytes:
    """Compress PIL image into JPEG <= max_kb."""
    img = img.convert("RGB")
    output = io.BytesIO()
    quality = 85

    while quality > 20:
        output.seek(0)
        output.truncate()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        if output.tell() <= max_kb * 1024:
            break
        quality -= 10

    return output.getvalue()


def preprocess_image_for_ai(file_bytes: bytes, max_kb: int = 1000) -> tuple[bytes, dict]:
    """Preprocess uploaded image for AI and return quality flags."""
    img = validate_image(file_bytes)

    width, height = img.size
    short_side = min(width, height)
    area = width * height
    detail_score = _estimate_detail_score(img)

    flags = {
        "orig_width": width,
        "orig_height": height,
        "soft_low_resolution": short_side < 400 or area < 160000,
        "soft_low_detail": detail_score < 8.0,
        "detail_score": round(detail_score, 2),
    }

    normalized = _normalize_for_ai(
        img,
        target_short_side=512,
        max_side=1600,
        pad_to_square=True,
    )
    compressed = _compress_pil_image(normalized, max_kb=max_kb)

    return compressed, flags


def compress_image(file_bytes: bytes, max_kb: int = 1000) -> bytes:
    """Backward-compatible wrapper."""
    compressed, _ = preprocess_image_for_ai(file_bytes, max_kb=max_kb)
    return compressed


def to_base64(file_bytes: bytes) -> str:
    """Convert raw bytes to base64 string."""
    return base64.b64encode(file_bytes).decode("utf-8")
