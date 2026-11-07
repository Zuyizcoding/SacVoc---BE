"""F1/F2 pipeline: prepare image -> infer/repair -> quality policy -> API model.

The reusable analyze_image() has no database dependency. Live API wrappers save
its result; evaluation calls it directly with demo fallback disabled. This keeps
measurement on the real business pipeline without touching user history.
"""

import logging
from dataclasses import asdict
from typing import Literal

from sqlalchemy.orm import Session

from app.core.prompts import get_body_prompts, get_color_prompts
from app.schemas.body import BodyApiResponse, BodyGeminiResponse
from app.schemas.color import ColorApiResponse, ColorInferenceResponse
from app.services.gemini_client import (
    GeminiClient,
    GeminiInvalidResponseError,
    GeminiSafetyBlockedError,
)
from app.utils.image_utils import preprocess_image_for_ai

from .errors import (
    AnalysisError,
    AnalysisRejectedError,
    AnalysisUnavailableError,
    GeminiOverloadedError,
)
from .photo_quality import (
    _apply_clothing_soft_fail_adjustments,
    _apply_soft_quality_adjustments,
    _photo_quality_score_value,
    needs_body_quality_hint,
    needs_color_quality_hint,
)
from .photo_types import ProcessedImage
from .prompts import (
    _build_body_best_effort_prompt,
    _build_body_completion_prompt,
    _build_body_medium_quality_hint_prompt,
    _build_color_completion_prompt,
    _build_color_quality_hint_prompt,
)
from .results import (
    _build_demo_body_payload,
    _build_demo_color_payload,
    _color_missing_fields,
    _finalize_color_result,
    merge_nonempty_result,
)
from .tracing import content_hash, current_trace, trace_stage

logger = logging.getLogger(__name__)


def _reject_if_error(result: dict, fallback: str) -> None:
    if result.get("error_code") is not None:
        code = str(result["error_code"]).strip().upper() or "ANALYSIS_FAILED"
        raise AnalysisRejectedError(result.get("message", fallback), code)


async def _infer_color(image: ProcessedImage, client: GeminiClient) -> dict:
    system, prompt = get_color_prompts()
    if needs_color_quality_hint(image.quality) or image.quality.soft_low_detail:
        prompt += "\n\n" + _build_color_quality_hint_prompt(image.quality)
    result = await client.analyze(
        image_bytes=image.jpeg_bytes,
        prompt=prompt,
        response_schema=ColorInferenceResponse,
        system_instruction=system,
    )
    _reject_if_error(
        result, "Không thể phân tích ảnh. Vui lòng thử lại với ảnh rõ hơn."
    )
    missing = _color_missing_fields(result)
    if missing:
        with trace_stage("repair"):
            repaired = await client.analyze(
                image_bytes=image.jpeg_bytes,
                prompt=_build_color_completion_prompt(missing, result, image.quality),
                response_schema=ColorInferenceResponse,
                system_instruction=system,
            )
        if repaired.get("error_code") is None:
            result = merge_nonempty_result(result, repaired)
    if "sub_season" in _color_missing_fields(result):
        raise AnalysisUnavailableError(
            "AI phản hồi chưa đủ dữ liệu để kết luận mùa sắc đáng tin cậy. Vui lòng thử lại sau vài giây hoặc dùng ảnh selfie rõ hơn.",
            "AI_INCOMPLETE",
        )
    return _finalize_color_result(result, image.quality)


async def _infer_body_pass(
    image: ProcessedImage, client: GeminiClient, prompt: str, *, repair: bool = False
) -> dict:
    """Each pass allows one best-effort call when clothing hides body proportions."""
    system, _ = get_body_prompts()
    result = await client.analyze(
        image_bytes=image.jpeg_bytes,
        prompt=prompt,
        response_schema=BodyGeminiResponse,
        system_instruction=system,
    )
    if str(result.get("error_code") or "").strip().upper() == "CLOTHING_TOO_WIDE":
        with trace_stage("repair_clothing" if repair else "clothing_best_effort"):
            result = await client.analyze(
                image_bytes=image.jpeg_bytes,
                prompt=_build_body_best_effort_prompt(),
                response_schema=BodyGeminiResponse,
                system_instruction=system,
            )
        if result.get("error_code") is not None:
            fallback = (
                "Không thể phân tích dáng người chính xác từ ảnh hiện tại."
                if repair
                else "Không thể phân tích dáng người chính xác do trang phục quá rộng."
            )
            raise AnalysisRejectedError(
                result.get("message", fallback), "CLOTHING_TOO_WIDE"
            )
        result = _apply_clothing_soft_fail_adjustments(result)
    _reject_if_error(
        result, "Không thể phân tích dáng người chính xác từ ảnh hiện tại."
    )
    return result


async def _infer_body(image: ProcessedImage, client: GeminiClient) -> dict:
    _, prompt = get_body_prompts()
    if needs_body_quality_hint(image.quality):
        prompt += "\n\n" + _build_body_medium_quality_hint_prompt(image.quality)
    result = await _infer_body_pass(image, client, prompt)
    parsed = BodyGeminiResponse.model_validate(result)
    missing = parsed.missing_detail_fields()
    if missing:
        with trace_stage("repair"):
            repaired = await _infer_body_pass(
                image,
                client,
                _build_body_completion_prompt(missing, result),
                repair=True,
            )
        result = merge_nonempty_result(result, repaired)
        parsed = BodyGeminiResponse.model_validate(result)
    if parsed.missing_detail_fields():
        raise AnalysisRejectedError(
            "Ảnh chưa đủ rõ để đưa ra gợi ý dáng người đáng tin cậy. Vui lòng dùng ảnh toàn thân, đứng thẳng và mặc đồ ôm sát hơn.",
            "AI_INCOMPLETE",
        )
    payload = _apply_soft_quality_adjustments(parsed.model_dump(), image.quality)
    if payload.get("confidence") is None:
        payload["confidence"] = round(
            min(0.78, 0.38 + 0.32 * _photo_quality_score_value(image.quality)), 4
        )
    return payload


async def _infer_with_fallback(
    image: ProcessedImage,
    client: GeminiClient,
    *,
    feature: Literal["F1", "F2"],
    allow_demo: bool = True,
) -> dict:
    infer = _infer_color if feature == "F1" else _infer_body
    demo = _build_demo_color_payload if feature == "F1" else _build_demo_body_payload
    try:
        result = await infer(image, client)
        result["analysis_source"] = "live"
        return result
    except (GeminiOverloadedError, TimeoutError):
        if not allow_demo:
            raise
        logger.warning("%s dùng demo do Gemini quá tải hoặc timeout", feature)
        return demo(image.quality)
    except AnalysisError:
        raise
    except (GeminiInvalidResponseError, GeminiSafetyBlockedError) as exc:
        # A blocked reply cannot provide a usable analysis. Keep the existing
        # public incomplete-response contract while retaining its cause in trace.
        subject = "màu sắc" if feature == "F1" else "dáng người"
        photo = "selfie" if feature == "F1" else "toàn thân"
        raise AnalysisUnavailableError(
            f"AI phản hồi chưa đầy đủ cho phân tích {subject}. Vui lòng thử lại sau vài giây hoặc dùng ảnh {photo} rõ hơn.",
            "AI_INCOMPLETE",
        ) from exc
    except ValueError as exc:
        subject = "màu sắc" if feature == "F1" else "dáng người"
        raise AnalysisUnavailableError(
            f"AI trả về kết quả chưa nhất quán cho phân tích {subject}. Vui lòng thử lại sau vài giây.",
            "AI_INVALID_RESULT",
        ) from exc


async def _analyze_payload(
    image_bytes: bytes,
    client: GeminiClient,
    *,
    feature: Literal["color", "body"],
    mode: Literal["live", "demo", "evaluation"] = "live",
) -> dict:
    """Run the business pipeline without persistence.

    live: actual AI with the existing availability fallback.
    demo: explicit canned output; no AI call.
    evaluation: actual AI, including repair and quality rules, but never a demo.

    A synthetic provider may be injected for tests. It still passes through the
    same schema/repair pipeline and must be identified as synthetic by the runner.
    """
    if feature not in {"color", "body"} or mode not in {"live", "demo", "evaluation"}:
        raise ValueError("Unknown analysis feature or execution mode")
    image = preprocess_image_for_ai(image_bytes, max_kb=1000)
    trace = current_trace()
    if trace is not None:
        trace.photo_quality = asdict(image.quality)
        trace.processed_image_sha256 = content_hash(image.jpeg_bytes)
    logger.info("%s photo quality=%s", feature, asdict(image.quality))
    if mode == "demo":
        demo = (
            _build_demo_color_payload
            if feature == "color"
            else _build_demo_body_payload
        )
        result = demo(image.quality)
    else:
        result = await _infer_with_fallback(
            image,
            client,
            feature="F1" if feature == "color" else "F2",
            allow_demo=mode == "live",
        )
    return result


async def analyze_image(
    image_bytes: bytes,
    client: GeminiClient,
    *,
    feature: Literal["color", "body"],
    mode: Literal["live", "demo", "evaluation"] = "live",
) -> ColorApiResponse | BodyApiResponse:
    """Return a validated result without importing or opening the database."""
    result = await _analyze_payload(image_bytes, client, feature=feature, mode=mode)
    schema = ColorApiResponse if feature == "color" else BodyApiResponse
    return schema.model_validate(result)


async def analyze_color_image(
    image_bytes: bytes, device_id: str, db: Session, client: GeminiClient
) -> ColorApiResponse:
    # Persistence belongs only to the live API wrapper. Preserve the exact payload
    # saved historically, including inference fields omitted from the API schema.
    from app.db.crud import save_color_result

    result = await _analyze_payload(image_bytes, client, feature="color")
    save_color_result(
        db=db,
        device_id=device_id,
        sub_season=result["sub_season"],
        primary_season=result["primary_season"],
        confidence=result["confidence"],
        full_result=result,
    )
    return ColorApiResponse.model_validate(result)


async def analyze_body_image(
    image_bytes: bytes, device_id: str, db: Session, client: GeminiClient
) -> BodyApiResponse:
    from app.db.crud import save_body_result

    result = await _analyze_payload(image_bytes, client, feature="body")
    save_body_result(
        db=db,
        device_id=device_id,
        shape=result["shape"],
        confidence=float(result["confidence"]),
        full_result=result,
    )
    return BodyApiResponse.model_validate(result)
