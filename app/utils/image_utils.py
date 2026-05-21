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


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def validate_image(file_bytes: bytes) -> Image.Image:
    """Validate image format and hard floor resolution.

    We no longer block by 400x400. We only reject extremely tiny images.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
    except Exception:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "PHOTO_UNREADABLE",
                "message": "Không đọc được ảnh. Vui lòng chọn ảnh rõ hơn hoặc thử ảnh khác.",
            },
        )

    width, height = img.size
    short_side = min(width, height)
    area = width * height

    if short_side < 140 or area < 30000:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "PHOTO_TOO_SMALL",
                "message": (
                    "Ảnh quá nhỏ để hệ thống nhận diện chính xác. "
                    "Vui lòng chọn ảnh rõ hơn, ưu tiên thấy rõ khuôn mặt/cơ thể."
                ),
            },
        )

    return img


def _estimate_detail_score(img: Image.Image) -> float:
    """Estimate visual detail using edge intensity (cv2-free)."""
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(edges).mean[0])


def _estimate_region_detail(
    img: Image.Image,
    *,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> float:
    width, height = img.size
    crop = img.crop(
        (
            int(width * left_ratio),
            int(height * top_ratio),
            int(width * right_ratio),
            int(height * bottom_ratio),
        )
    )
    return _estimate_detail_score(crop)


def _estimate_brightness_and_contrast(img: Image.Image) -> tuple[float, float]:
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    mean_brightness = float(stat.mean[0])
    stddev = float(stat.stddev[0]) if stat.stddev else 0.0
    return mean_brightness, stddev


def _estimate_region_brightness_and_contrast(
    img: Image.Image,
    *,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> tuple[float, float]:
    width, height = img.size
    crop = img.crop(
        (
            int(width * left_ratio),
            int(height * top_ratio),
            int(width * right_ratio),
            int(height * bottom_ratio),
        )
    )
    return _estimate_brightness_and_contrast(crop)


def _estimate_color_cast(img: Image.Image) -> float:
    rgb = img.convert("RGB")
    stat = ImageStat.Stat(rgb)
    r_mean, g_mean, b_mean = [float(value) for value in stat.mean[:3]]
    return max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))


def _is_far_subject_for_color(
    *,
    portrait_ratio: float,
    detail_score: float,
    face_hint_detail: float,
    upper_center_detail: float,
) -> bool:
    """Heuristic for rejecting F1 photos where the subject is likely too small.

    Close-up selfies can still have low edge/detail scores under soft lighting,
    so we only block when the upper-face regions are both low-detail and do not
    stand out relative to the whole frame.
    """
    if portrait_ratio <= 1.15:
        return False
    if detail_score >= 6.5 or face_hint_detail >= 9.0 or upper_center_detail >= 8.5:
        return False

    baseline = max(detail_score, 0.1)
    face_focus_ratio = face_hint_detail / baseline
    upper_center_focus_ratio = upper_center_detail / baseline

    return face_focus_ratio < 1.2 and upper_center_focus_ratio < 1.15


def _is_soft_bad_lighting(
    *,
    global_brightness: float,
    global_contrast: float,
    face_brightness: float,
    face_contrast: float,
    face_neck_brightness: float,
    face_neck_contrast: float,
) -> bool:
    """Prefer face/neck exposure over whole-frame brightness.

    White backgrounds and clothing can make passport-style photos look globally
    over-bright even when the face exposure is actually usable. We therefore use
    the face/neck crop as the primary signal, and only fall back to global stats
    for clearly extreme cases.
    """
    if face_neck_brightness < 72:
        return True
    if face_neck_contrast < 24:
        return True
    if face_neck_brightness > 208 and face_neck_contrast < 40:
        return True
    if face_brightness > 215 and face_contrast < 36:
        return True
    if global_brightness < 60:
        return True
    if global_contrast < 22:
        return True
    if global_brightness > 225 and face_neck_brightness > 210:
        return True
    return False


def _score_resolution_quality(short_side: int, area: int) -> float:
    short_score = _clamp01((short_side - 180.0) / 340.0)
    area_score = _clamp01((area - 50_000.0) / 250_000.0)
    return round(0.6 * short_score + 0.4 * area_score, 4)


def _score_detail_quality(
    detail_score: float,
    face_hint_detail: float,
    upper_center_detail: float,
) -> float:
    global_score = _clamp01((detail_score - 4.5) / 5.5)
    face_score = _clamp01((face_hint_detail - 5.5) / 6.5)
    upper_score = _clamp01((upper_center_detail - 5.0) / 6.5)
    return round(0.3 * global_score + 0.4 * face_score + 0.3 * upper_score, 4)


def _score_lighting_quality(
    *,
    global_brightness: float,
    global_contrast: float,
    face_neck_brightness: float,
    face_neck_contrast: float,
) -> float:
    if 95 <= face_neck_brightness <= 185:
        brightness_score = 1.0
    elif 80 <= face_neck_brightness <= 200:
        brightness_score = 0.85
    elif 72 <= face_neck_brightness <= 208:
        brightness_score = 0.65
    else:
        brightness_score = 0.4

    if face_neck_contrast >= 55:
        contrast_score = 1.0
    elif face_neck_contrast >= 42:
        contrast_score = 0.82
    elif face_neck_contrast >= 28:
        contrast_score = 0.65
    else:
        contrast_score = 0.4

    score = 0.65 * brightness_score + 0.35 * contrast_score
    if global_brightness > 225 and face_neck_brightness > 210:
        score = min(score, 0.3)
    if global_contrast < 22:
        score = min(score, 0.35)
    return round(score, 4)


def _score_color_quality(color_cast: float) -> float:
    if color_cast <= 12:
        return 1.0
    if color_cast <= 20:
        return 0.9
    if color_cast <= 30:
        return 0.78
    if color_cast <= 40:
        return 0.6
    return 0.35


def _score_framing_quality(
    *,
    portrait_ratio: float,
    face_focus_ratio: float,
    upper_center_focus_ratio: float,
    far_subject_for_color: bool,
) -> float:
    focus_score = (
        _clamp01((face_focus_ratio - 0.9) / 1.0)
        + _clamp01((upper_center_focus_ratio - 0.9) / 1.0)
    ) / 2.0
    portrait_score = 1.0 if 1.05 <= portrait_ratio <= 1.7 else 0.85
    score = 0.75 * focus_score + 0.25 * portrait_score
    if far_subject_for_color:
        score = min(score, 0.35)
    return round(score, 4)


def _photo_quality_band(score: float) -> str:
    if score >= 0.85:
        return "excellent"
    if score >= 0.72:
        return "good"
    if score >= 0.58:
        return "usable"
    return "poor"


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
    face_hint_detail = _estimate_region_detail(
        img,
        left_ratio=0.30,
        top_ratio=0.05,
        right_ratio=0.70,
        bottom_ratio=0.35,
    )
    upper_center_detail = _estimate_region_detail(
        img,
        left_ratio=0.25,
        top_ratio=0.0,
        right_ratio=0.75,
        bottom_ratio=0.45,
    )
    brightness, contrast = _estimate_brightness_and_contrast(img)
    face_brightness, face_contrast = _estimate_region_brightness_and_contrast(
        img,
        left_ratio=0.28,
        top_ratio=0.12,
        right_ratio=0.72,
        bottom_ratio=0.58,
    )
    face_neck_brightness, face_neck_contrast = _estimate_region_brightness_and_contrast(
        img,
        left_ratio=0.26,
        top_ratio=0.10,
        right_ratio=0.74,
        bottom_ratio=0.72,
    )
    color_cast = _estimate_color_cast(img)
    portrait_ratio = height / float(width) if width else 1.0
    face_focus_ratio = face_hint_detail / max(detail_score, 0.1)
    upper_center_focus_ratio = upper_center_detail / max(detail_score, 0.1)
    far_subject_for_color = _is_far_subject_for_color(
        portrait_ratio=portrait_ratio,
        detail_score=detail_score,
        face_hint_detail=face_hint_detail,
        upper_center_detail=upper_center_detail,
    )
    resolution_quality_score = _score_resolution_quality(short_side, area)
    detail_quality_score = _score_detail_quality(
        detail_score,
        face_hint_detail,
        upper_center_detail,
    )
    lighting_quality_score = _score_lighting_quality(
        global_brightness=brightness,
        global_contrast=contrast,
        face_neck_brightness=face_neck_brightness,
        face_neck_contrast=face_neck_contrast,
    )
    color_quality_score = _score_color_quality(color_cast)
    framing_quality_score = _score_framing_quality(
        portrait_ratio=portrait_ratio,
        face_focus_ratio=face_focus_ratio,
        upper_center_focus_ratio=upper_center_focus_ratio,
        far_subject_for_color=far_subject_for_color,
    )
    photo_quality_score = round(
        0.12 * resolution_quality_score
        + 0.30 * detail_quality_score
        + 0.24 * lighting_quality_score
        + 0.12 * color_quality_score
        + 0.22 * framing_quality_score,
        4,
    )
    photo_quality_band = _photo_quality_band(photo_quality_score)

    flags = {
        "orig_width": width,
        "orig_height": height,
        "soft_low_resolution": short_side < 400 or area < 160000,
        "soft_low_detail": detail_score < 8.0,
        "soft_bad_lighting": _is_soft_bad_lighting(
            global_brightness=brightness,
            global_contrast=contrast,
            face_brightness=face_brightness,
            face_contrast=face_contrast,
            face_neck_brightness=face_neck_brightness,
            face_neck_contrast=face_neck_contrast,
        ),
        "soft_color_cast": color_cast > 35,
        "soft_far_subject_for_color": far_subject_for_color,
        "detail_score": round(detail_score, 2),
        "face_hint_detail_score": round(face_hint_detail, 2),
        "upper_center_detail_score": round(upper_center_detail, 2),
        "face_focus_ratio": round(face_focus_ratio, 2),
        "upper_center_focus_ratio": round(upper_center_focus_ratio, 2),
        "brightness_score": round(brightness, 2),
        "contrast_score": round(contrast, 2),
        "face_brightness_score": round(face_brightness, 2),
        "face_contrast_score": round(face_contrast, 2),
        "face_neck_brightness_score": round(face_neck_brightness, 2),
        "face_neck_contrast_score": round(face_neck_contrast, 2),
        "color_cast_score": round(color_cast, 2),
        "portrait_ratio": round(portrait_ratio, 2),
        "resolution_quality_score": resolution_quality_score,
        "detail_quality_score": detail_quality_score,
        "lighting_quality_score": lighting_quality_score,
        "color_quality_score": round(color_quality_score, 4),
        "framing_quality_score": framing_quality_score,
        "photo_quality_score": photo_quality_score,
        "photo_quality_band": photo_quality_band,
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
