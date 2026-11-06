"""Dữ liệu trung gian có tên rõ, không cần đoán khóa trong dict."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PhotoQualityReport:
    """Chỉ số heuristic của ảnh; không phải xác suất AI phân tích đúng."""

    orig_width: int = 0
    orig_height: int = 0
    soft_low_resolution: bool = False
    soft_low_detail: bool = False
    soft_bad_lighting: bool = False
    soft_color_cast: bool = False
    soft_far_subject_for_color: bool = False
    detail_score: float = 0.0
    face_hint_detail_score: float = 0.0
    upper_center_detail_score: float = 0.0
    face_focus_ratio: float = 0.0
    upper_center_focus_ratio: float = 0.0
    brightness_score: float = 0.0
    contrast_score: float = 0.0
    face_brightness_score: float = 0.0
    face_contrast_score: float = 0.0
    face_neck_brightness_score: float = 0.0
    face_neck_contrast_score: float = 0.0
    color_cast_score: float = 0.0
    portrait_ratio: float = 0.0
    resolution_quality_score: float = 0.0
    detail_quality_score: float = 0.0
    lighting_quality_score: float = 0.0
    color_quality_score: float = 0.0
    framing_quality_score: float = 0.0
    photo_quality_score: float = 0.75
    photo_quality_band: str = "good"


@dataclass(frozen=True)
class ProcessedImage:
    jpeg_bytes: bytes
    quality: PhotoQualityReport
