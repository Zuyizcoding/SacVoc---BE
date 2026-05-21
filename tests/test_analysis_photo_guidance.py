from app.api.endpoints.analyze import (
    _build_body_medium_quality_hint_prompt,
    _color_close_up_guidance_message,
    _should_bias_body_best_effort,
    _should_return_color_close_up_guidance,
)
from app.utils.image_utils import (
    _is_far_subject_for_color,
    _is_soft_bad_lighting,
    _photo_quality_band,
    _score_color_quality,
    _score_detail_quality,
    _score_framing_quality,
    _score_lighting_quality,
    _score_resolution_quality,
)


def test_color_guidance_triggers_for_far_subject_photo() -> None:
    quality_flags = {
        "soft_far_subject_for_color": True,
    }

    assert _should_return_color_close_up_guidance(quality_flags) is True
    assert "selfie gần hơn" in _color_close_up_guidance_message(quality_flags)


def test_color_guidance_does_not_trigger_for_close_up_photo() -> None:
    quality_flags = {
        "soft_far_subject_for_color": False,
    }

    assert _should_return_color_close_up_guidance(quality_flags) is False


def test_far_subject_heuristic_triggers_for_small_subject_photo() -> None:
    assert (
        _is_far_subject_for_color(
            portrait_ratio=1.33,
            detail_score=5.1,
            face_hint_detail=5.6,
            upper_center_detail=5.5,
        )
        is True
    )


def test_far_subject_heuristic_skips_soft_but_close_selfie() -> None:
    assert (
        _is_far_subject_for_color(
            portrait_ratio=1.33,
            detail_score=5.39,
            face_hint_detail=6.66,
            upper_center_detail=6.10,
        )
        is False
    )


def test_soft_bad_lighting_ignores_white_background_when_face_region_is_ok() -> None:
    assert (
        _is_soft_bad_lighting(
            global_brightness=198.42,
            global_contrast=68.12,
            face_brightness=138.67,
            face_contrast=58.90,
            face_neck_brightness=144.16,
            face_neck_contrast=59.74,
        )
        is False
    )


def test_soft_bad_lighting_triggers_for_overexposed_face_region() -> None:
    assert (
        _is_soft_bad_lighting(
            global_brightness=214.0,
            global_contrast=34.0,
            face_brightness=218.0,
            face_contrast=28.0,
            face_neck_brightness=212.0,
            face_neck_contrast=32.0,
        )
        is True
    )


def test_body_best_effort_bias_for_medium_quality_full_body() -> None:
    quality_flags = {
        "soft_low_detail": True,
        "soft_low_resolution": False,
        "soft_bad_lighting": False,
        "soft_color_cast": False,
        "detail_score": 5.33,
    }

    assert _should_bias_body_best_effort(quality_flags) is True
    prompt = _build_body_medium_quality_hint_prompt(quality_flags)
    assert "best-effort" in prompt
    assert "CLOTHING_TOO_WIDE" in prompt


def test_weighted_photo_quality_score_is_high_for_clear_face_and_balanced_light() -> None:
    photo_quality_score = (
        0.12 * _score_resolution_quality(787, 787 * 989)
        + 0.30 * _score_detail_quality(8.05, 19.97, 16.95)
        + 0.24
        * _score_lighting_quality(
            global_brightness=198.42,
            global_contrast=68.12,
            face_neck_brightness=144.16,
            face_neck_contrast=59.74,
        )
        + 0.12 * _score_color_quality(21.49)
        + 0.22
        * _score_framing_quality(
            portrait_ratio=1.26,
            face_focus_ratio=2.48,
            upper_center_focus_ratio=2.11,
            far_subject_for_color=False,
        )
    )

    assert photo_quality_score > 0.85
    assert _photo_quality_band(photo_quality_score) == "excellent"


def test_weighted_photo_quality_score_drops_for_soft_far_subject_photo() -> None:
    photo_quality_score = (
        0.12 * _score_resolution_quality(320, 320 * 480)
        + 0.30 * _score_detail_quality(5.1, 5.6, 5.5)
        + 0.24
        * _score_lighting_quality(
            global_brightness=170.0,
            global_contrast=30.0,
            face_neck_brightness=168.0,
            face_neck_contrast=28.0,
        )
        + 0.12 * _score_color_quality(18.0)
        + 0.22
        * _score_framing_quality(
            portrait_ratio=1.33,
            face_focus_ratio=1.1,
            upper_center_focus_ratio=1.08,
            far_subject_for_color=True,
        )
    )

    assert photo_quality_score < 0.58
    assert _photo_quality_band(photo_quality_score) == "poor"
