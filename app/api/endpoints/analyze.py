"""
Router phÃ¢n tÃ­ch áº£nh â€” Color (F1) + Body (F2).

TuÃ¢n thá»§:
  - backend-python #1: Router riÃªng biá»‡t cho /analyze
  - backend-python #2: Depends() inject GeminiClient + DB Session
  - TÃ­nh nÄƒng #9: try/finally xÃ³a áº£nh táº¡m â€” KHÃ”NG lÆ°u áº£nh ngÆ°á»i dÃ¹ng
  - TÃ­nh nÄƒng #34: HTTP 503/422/429/400 â†’ thÃ´ng bÃ¡o tiáº¿ng Viá»‡t rÃµ rÃ ng
  - TÃ­nh nÄƒng #15: F1+F2â†’F3 combined flow â€” Ä‘á»c káº¿t quáº£ tá»« DB

Luá»“ng xá»­ lÃ½:
  1. Nháº­n áº£nh upload (UploadFile)
  2. Validate kÃ­ch thÆ°á»›c (â‰¥400Ã—400px)
  3. Compress (â‰¤1MB JPEG)
  4. Gá»­i Gemini qua GeminiClient
  5. Override palette tá»« season_data.py (F1)
  6. LÆ°u káº¿t quáº£ vÃ o ANALYSIS_HISTORY
  7. finally: XÃ“A áº£nh táº¡m (dÃ¹ thÃ nh cÃ´ng hay tháº¥t báº¡i)
"""
import os
import logging
import tempfile
import json

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Header
from google.api_core.exceptions import DeadlineExceeded
from sqlalchemy.orm import Session

from app.schemas.base import ApiResponse
from app.schemas.color import ColorApiResponse, ColorGeminiResponse
from app.schemas.body import BodyApiResponse, BodyGeminiResponse
from app.core.prompts import get_body_prompts, get_color_prompts
from app.services.gemini_client import GeminiClient, get_gemini_client
from app.schemas.history import HistoryApiResponse
from app.utils.image_utils import preprocess_image_for_ai
from app.db.database import get_db
from app.db.crud import get_analysis, save_color_result, save_body_result
from app.data.season_data import (
    canonicalize_sub_season,
    get_season_entry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analyze", tags=["Analysis"])


def _canonicalize_color_result(result: dict) -> dict:
    """Chuáº©n hÃ³a F1 báº±ng báº£ng season_data trÆ°á»›c khi lÆ°u/tráº£ vá»."""
    sub_season_raw = str(result.get("sub_season") or "").strip()
    canonical_sub_season = canonicalize_sub_season(sub_season_raw)
    if canonical_sub_season is None:
        raise ValueError(f"Invalid sub-season from Gemini: {sub_season_raw}")

    entry = get_season_entry(canonical_sub_season)
    result["sub_season"] = canonical_sub_season
    result["primary_season"] = entry["primary"]
    result["palette"] = entry["palette"]
    result["makeup"] = entry["makeup"]
    return result


def _merge_body_result(base: dict, patch: dict) -> dict:
    """Merge partial F2 payloads, Æ°u tiÃªn dá»¯ liá»‡u má»›i khi field cÃ³ giÃ¡ trá»‹."""
    merged = dict(base)
    for key, value in patch.items():
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value
    return merged


def _build_body_completion_prompt(missing_fields: list[str], partial_result: dict) -> str:
    """Prompt ngáº¯n Ä‘á»ƒ yÃªu cáº§u Gemini chá»‰ hoÃ n thiá»‡n field F2 cÃ²n thiáº¿u."""
    missing_text = ", ".join(missing_fields)
    return (
        "<task>\n"
        "Báº¡n vá»«a phÃ¢n tÃ­ch body shape tá»« áº£nh nÃ y nhÆ°ng cÃ²n thiáº¿u má»™t sá»‘ field báº¯t buá»™c.\n"
        f"HÃ£y phÃ¢n tÃ­ch láº¡i cÃ¹ng áº£nh vÃ  tráº£ JSON hoÃ n chá»‰nh, táº­p trung bá»• sung: {missing_text}.\n"
        "Náº¿u áº£nh khÃ´ng Ä‘á»§ rÃµ Ä‘á»ƒ káº¿t luáº­n cháº¯c cháº¯n, Æ°u tiÃªn váº«n tráº£ káº¿t quáº£ vá»›i confidence tháº¥p,\n"
        "chá»‰ tráº£ JSON lá»—i CLOTHING_TOO_WIDE khi trang phá»¥c che dÃ¡ng quÃ¡ náº·ng vÃ  khÃ´ng thá»ƒ suy luáº­n tá»· lá»‡ cÆ¡ báº£n.\n"
        "nhÆ°ng khÃ´ng Ä‘Æ°á»£c bá»‹a máº¹o máº·c Ä‘á»“ chung chung.\n"
        f"Partial result trÆ°á»›c Ä‘Ã³ Ä‘á»ƒ tham chiáº¿u: {json.dumps(partial_result, ensure_ascii=False)}\n"
        "</task>"
    )


def _build_body_best_effort_prompt() -> str:
    """Prompt fail-safe: force best-effort analysis for borderline wide clothing."""
    return (
        "<task>\n"
        "Ảnh này có thể mặc đồ hơi rộng, nhưng hãy ưu tiên phân tích best-effort.\n"
        "Yêu cầu:\n"
        "1) VẪN trả đầy đủ JSON body shape (shape, confidence, measurements, styling_tips, avoid, explanation).\n"
        "2) Đặt confidence thấp (0.25-0.45) nếu độ chắc chắn không cao.\n"
        "3) Trong explanation phải nêu rõ là trang phục hơi rộng nên kết quả chỉ tham khảo.\n"
        "4) CHỈ trả lỗi CLOTHING_TOO_WIDE khi bị che dáng cực nặng: không nhìn thấy tương quan vai/eo/hông một cách tối thiểu.\n"
        "5) Không bịa dữ liệu, không dùng câu chung chung.\n"
        "</task>"
    )


def _apply_clothing_soft_fail_adjustments(result: dict) -> dict:
    """Cap confidence and append warning when analysis is forced from wide clothing."""
    patched = dict(result)
    warning = (
        "Trang phục trong ảnh hơi rộng nên hệ thống phân tích theo hướng tham khảo. "
        "Bạn có thể dùng ảnh mặc đồ ôm sát hơn để tăng độ chính xác."
    )

    confidence_raw = patched.get("confidence")
    if isinstance(confidence_raw, (int, float)):
        patched["confidence"] = min(float(confidence_raw), 0.45)
    else:
        patched["confidence"] = 0.4

    explanation = str(patched.get("explanation") or "").strip()
    if explanation:
        if "Lưu ý:" not in explanation:
            patched["explanation"] = f"{explanation} Lưu ý: {warning}"
    else:
        patched["explanation"] = f"Lưu ý: {warning}"

    return patched


def _soft_quality_warning_message() -> str:
    return (
        "Ảnh hiện tại hơi nhỏ hoặc thiếu chi tiết, kết quả chỉ nên dùng để tham khảo. "
        "Bạn có thể chụp ảnh sáng và rõ hơn để tăng độ chính xác."
    )


def _apply_soft_quality_adjustments(result: dict, quality_flags: dict) -> dict:
    """Lower confidence and add a friendly warning when image quality is suboptimal."""
    low_resolution = bool(quality_flags.get("soft_low_resolution"))
    low_detail = bool(quality_flags.get("soft_low_detail"))
    if not (low_resolution or low_detail):
        return result

    warning = _soft_quality_warning_message()
    patched = dict(result)

    confidence_raw = patched.get("confidence")
    if isinstance(confidence_raw, (int, float)):
        patched["confidence"] = min(float(confidence_raw), 0.55)

    explanation = str(patched.get("explanation") or "").strip()
    if explanation:
        patched["explanation"] = f"{explanation} Lưu ý: {warning}"

    advice = patched.get("advice")
    if isinstance(advice, str) and advice.strip():
        patched["advice"] = f"{advice.strip()} Lưu ý: {warning}"

    return patched


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# POST /api/v1/analyze/color (TÃ­nh nÄƒng #16, #17, #18, #19)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@router.post(
    "/color",
    response_model=ApiResponse[ColorApiResponse],
    summary="F1 â€” PhÃ¢n tÃ­ch mÃ u sáº¯c cÃ¡ nhÃ¢n",
    description="Nháº­n áº£nh selfie, tráº£ vá» 12-season + palette + makeup guide.",
)
async def analyze_color(
    file: UploadFile = File(..., description="áº¢nh selfie JPEG/PNG"),
    x_device_id: str = Header(
        ..., description="SHA256 device ID (TÃ­nh nÄƒng #3)"
    ),
    client: GeminiClient = Depends(get_gemini_client),
    db: Session = Depends(get_db),
) -> ApiResponse[ColorApiResponse]:
    """PhÃ¢n tÃ­ch mÃ u sáº¯c cÃ¡ nhÃ¢n tá»« áº£nh selfie.

    Flow:
        Upload â†’ Validate â†’ Compress â†’ Gemini â†’ Override palette â†’ Response
    """
    tmp_path: str | None = None

    try:
        # --- BÆ°á»›c 1: Äá»c file upload ---
        file_bytes = await file.read()
        logger.info(
            "ðŸ“¸ Color analysis request | device=%s | size=%d bytes",
            x_device_id[:8],
            len(file_bytes),
        )

        compressed, quality_flags = preprocess_image_for_ai(
            file_bytes,
            max_kb=1000,
        )

        # --- BÆ°á»›c 4: LÆ°u táº¡m (Ä‘á»ƒ log/debug náº¿u cáº§n) ---
        with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False
        ) as tmp:
            tmp.write(compressed)
            tmp_path = tmp.name

        # --- BÆ°á»›c 5: Gá»i Gemini (TÃ­nh nÄƒng #10, #11, #12) ---
        system_instruction_f1, prompt_f1 = get_color_prompts()
        result = await client.analyze(
            image_bytes=compressed,
            prompt=prompt_f1,
            response_schema=ColorGeminiResponse,
            system_instruction=system_instruction_f1,
        )

        # If Gemini returns a structured error, propagate as an API-level error.
        if result.get("error_code") is not None:
            return ApiResponse(
                success=False,
                data=None,
                error=result.get(
                    "message",
                    "KhÃ´ng thá»ƒ phÃ¢n tÃ­ch áº£nh. Vui lÃ²ng thá»­ láº¡i vá»›i áº£nh rÃµ hÆ¡n.",
                ),
                error_code=422,
            )

        # --- BÆ°á»›c 6: Canonicalize F1 tá»« season_data.py (TÃ­nh nÄƒng #20) ---
        result = _canonicalize_color_result(result)
        result = _apply_soft_quality_adjustments(result, quality_flags)

        # --- BÆ°á»›c 7: LÆ°u vÃ o ANALYSIS_HISTORY ---
        save_color_result(
            db=db,
            device_id=x_device_id,
            sub_season=result["sub_season"],
            primary_season=result["primary_season"],
            confidence=result["confidence"],
            full_result=result,
        )

        return ApiResponse(
            success=True,
            data=ColorApiResponse.model_validate(result),
        )

    # â”€â”€â”€ Error Handling (TÃ­nh nÄƒng #34) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    except HTTPException:
        raise  # 422 tá»« validate_image â€” giá»¯ nguyÃªn

    except TimeoutError:
        logger.error("â±ï¸ Gemini timeout cho F1")
        raise HTTPException(
            status_code=503,
            detail="AI Ä‘ang báº­n. Vui lÃ²ng thá»­ láº¡i sau 5 giÃ¢y.",
        )

    except DeadlineExceeded:
        logger.error("â±ï¸ Gemini timeout (DeadlineExceeded) cho F1")
        raise HTTPException(
            status_code=503,
            detail="AI Ä‘ang báº­n. Vui lÃ²ng thá»­ láº¡i sau 5 giÃ¢y.",
        )

    except ValueError as exc:
        logger.error("ðŸ”´ Gemini JSON parse error: %s", str(exc))
        raise HTTPException(
            status_code=400,
            detail="KhÃ´ng phÃ¢n tÃ­ch Ä‘Æ°á»£c áº£nh. Vui lÃ²ng chá»¥p áº£nh rÃµ hÆ¡n.",
        )

    except Exception as exc:
        logger.exception("ðŸ’¥ Lá»—i khÃ´ng xÃ¡c Ä‘á»‹nh trong F1: %s", str(exc))
        raise HTTPException(
            status_code=503,
            detail="Há»‡ thá»‘ng gáº·p sá»± cá»‘. Vui lÃ²ng thá»­ láº¡i sau.",
        )

    # â”€â”€â”€ Photo Cleanup (TÃ­nh nÄƒng #9) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info("ðŸ—‘ï¸ ÄÃ£ xÃ³a áº£nh táº¡m: %s", tmp_path)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# POST /api/v1/analyze/body (TÃ­nh nÄƒng #23, #24, #25, #26)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@router.post(
    "/body",
    response_model=ApiResponse[BodyApiResponse],
    summary="F2 â€” PhÃ¢n tÃ­ch dÃ¡ng ngÆ°á»i",
    description="Nháº­n áº£nh toÃ n thÃ¢n, tráº£ vá» body shape + styling tips.",
)
async def analyze_body(
    file: UploadFile = File(..., description="áº¢nh toÃ n thÃ¢n JPEG/PNG"),
    x_device_id: str = Header(
        ..., description="SHA256 device ID (TÃ­nh nÄƒng #3)"
    ),
    client: GeminiClient = Depends(get_gemini_client),
    db: Session = Depends(get_db),
) -> ApiResponse[BodyApiResponse]:
    """PhÃ¢n tÃ­ch dÃ¡ng ngÆ°á»i tá»« áº£nh toÃ n thÃ¢n.

    Flow tÆ°Æ¡ng tá»± F1 â€” copy pattern try/finally.
    """
    tmp_path: str | None = None

    try:
        file_bytes = await file.read()
        logger.info(
            "ðŸ“¸ Body analysis request | device=%s | size=%d bytes",
            x_device_id[:8],
            len(file_bytes),
        )

        compressed, quality_flags = preprocess_image_for_ai(
            file_bytes,
            max_kb=1000,
        )

        with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False
        ) as tmp:
            tmp.write(compressed)
            tmp_path = tmp.name

        system_instruction_f2, prompt_f2 = get_body_prompts()

        result = await client.analyze(
            image_bytes=compressed,
            prompt=prompt_f2,
            response_schema=BodyGeminiResponse,
            system_instruction=system_instruction_f2,
        )

        # For wide-clothing edge cases, run one best-effort retry before hard fail.
        if result.get("error_code") is not None:
            if str(result.get("error_code")).strip().upper() == "CLOTHING_TOO_WIDE":
                logger.info(
                    "F2 CLOTHING_TOO_WIDE on first pass -> run best-effort retry | device=%s",
                    x_device_id[:8],
                )
                retry_prompt = _build_body_best_effort_prompt()
                retry_result = await client.analyze(
                    image_bytes=compressed,
                    prompt=retry_prompt,
                    response_schema=BodyGeminiResponse,
                    system_instruction=system_instruction_f2,
                )
                if retry_result.get("error_code") is None:
                    result = _apply_clothing_soft_fail_adjustments(retry_result)
                else:
                    return ApiResponse(
                        success=False,
                        data=None,
                        error=retry_result.get(
                            "message",
                            "KhÃ´ng thá»ƒ phÃ¢n tÃ­ch dÃ¡ng ngÆ°á»i chÃ­nh xÃ¡c do trang phá»¥c quÃ¡ rá»™ng.",
                        ),
                        error_code=422,
                    )
            else:
                return ApiResponse(
                    success=False,
                    data=None,
                    error=result.get(
                        "message",
                        "KhÃ´ng thá»ƒ phÃ¢n tÃ­ch dÃ¡ng ngÆ°á»i chÃ­nh xÃ¡c tá»« áº£nh hiá»‡n táº¡i.",
                    ),
                    error_code=422,
                )

        parsed_result = BodyGeminiResponse.model_validate(result)
        missing_fields = parsed_result.missing_detail_fields()
        if missing_fields:
            logger.warning(
                "F2 thiáº¿u field grounded sau lÆ°á»£t Ä‘áº§u: %s | device=%s",
                missing_fields,
                x_device_id[:8],
            )
            repair_prompt = _build_body_completion_prompt(missing_fields, result)
            repaired = await client.analyze(
                image_bytes=compressed,
                prompt=repair_prompt,
                response_schema=BodyGeminiResponse,
                system_instruction=system_instruction_f2,
            )

            if repaired.get("error_code") is not None:
                if str(repaired.get("error_code")).strip().upper() == "CLOTHING_TOO_WIDE":
                    logger.info(
                        "F2 CLOTHING_TOO_WIDE on repair pass -> run best-effort retry | device=%s",
                        x_device_id[:8],
                    )
                    retry_prompt = _build_body_best_effort_prompt()
                    retry_result = await client.analyze(
                        image_bytes=compressed,
                        prompt=retry_prompt,
                        response_schema=BodyGeminiResponse,
                        system_instruction=system_instruction_f2,
                    )
                    if retry_result.get("error_code") is None:
                        repaired = _apply_clothing_soft_fail_adjustments(retry_result)
                    else:
                        return ApiResponse(
                            success=False,
                            data=None,
                            error=retry_result.get(
                                "message",
                                "KhÃ´ng thá»ƒ phÃ¢n tÃ­ch dÃ¡ng ngÆ°á»i chÃ­nh xÃ¡c tá»« áº£nh hiá»‡n táº¡i.",
                            ),
                            error_code=422,
                        )
                else:
                    return ApiResponse(
                        success=False,
                        data=None,
                        error=repaired.get(
                            "message",
                            "KhÃ´ng thá»ƒ phÃ¢n tÃ­ch dÃ¡ng ngÆ°á»i chÃ­nh xÃ¡c tá»« áº£nh hiá»‡n táº¡i.",
                        ),
                        error_code=422,
                    )

            result = _merge_body_result(result, repaired)
            parsed_result = BodyGeminiResponse.model_validate(result)
            missing_fields = parsed_result.missing_detail_fields()

        if missing_fields:
            logger.warning(
                "F2 váº«n thiáº¿u field sau repair pass: %s | device=%s",
                missing_fields,
                x_device_id[:8],
            )
            return ApiResponse(
                success=False,
                data=None,
                error=(
                    "áº¢nh chÆ°a Ä‘á»§ rÃµ Ä‘á»ƒ Ä‘Æ°a ra gá»£i Ã½ dÃ¡ng ngÆ°á»i Ä‘Ã¡ng tin cáº­y. "
                    "Vui lÃ²ng dÃ¹ng áº£nh toÃ n thÃ¢n, Ä‘á»©ng tháº³ng vÃ  máº·c Ä‘á»“ Ã´m sÃ¡t hÆ¡n."
                ),
                error_code=422,
            )

        response_payload = parsed_result.model_dump()
        response_payload = _apply_soft_quality_adjustments(
            response_payload,
            quality_flags,
        )
        if response_payload.get("confidence") is None:
            response_payload["confidence"] = 0.4

        save_body_result(
            db=db,
            device_id=x_device_id,
            shape=parsed_result.shape,
            confidence=float(response_payload["confidence"]),
            full_result=response_payload,
        )

        return ApiResponse(
            success=True,
            data=BodyApiResponse.model_validate(response_payload),
        )

    except HTTPException:
        raise

    except TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="AI Ä‘ang báº­n. Vui lÃ²ng thá»­ láº¡i sau 5 giÃ¢y.",
        )

    except DeadlineExceeded:
        raise HTTPException(
            status_code=503,
            detail="AI Ä‘ang báº­n. Vui lÃ²ng thá»­ láº¡i sau 5 giÃ¢y.",
        )

    except ValueError as exc:
        logger.error("ðŸ”´ Body JSON parse error: %s", str(exc))
        raise HTTPException(
            status_code=400,
            detail="KhÃ´ng phÃ¢n tÃ­ch Ä‘Æ°á»£c áº£nh. Chá»¥p áº£nh toÃ n thÃ¢n rÃµ hÆ¡n nhÃ©!",
        )

    except Exception as exc:
        logger.exception("ðŸ’¥ Lá»—i khÃ´ng xÃ¡c Ä‘á»‹nh trong F2: %s", str(exc))
        raise HTTPException(
            status_code=503,
            detail="Há»‡ thá»‘ng gáº·p sá»± cá»‘. Vui lÃ²ng thá»­ láº¡i sau.",
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info("ðŸ—‘ï¸ ÄÃ£ xÃ³a áº£nh táº¡m: %s", tmp_path)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# GET /api/v1/analyze/history (SHOULD â€” Xem láº¡i káº¿t quáº£)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@router.get(
    "/history",
    response_model=ApiResponse[HistoryApiResponse],
    summary="Xem láº¡i káº¿t quáº£ phÃ¢n tÃ­ch F1 + F2",
    description="Äá»c káº¿t quáº£ phÃ¢n tÃ­ch Ä‘Ã£ lÆ°u trong AnalysisHistory.",
)
async def get_analysis_history(
    x_device_id: str = Header(
        ..., description="SHA256 device ID (TÃ­nh nÄƒng #3)"
    ),
    db: Session = Depends(get_db),
) -> ApiResponse[HistoryApiResponse]:
    """Tráº£ vá» lá»‹ch sá»­ F1 + F2 náº¿u Ä‘Ã£ phÃ¢n tÃ­ch.

    SHOULD: MÃ n "Xem láº¡i káº¿t quáº£" Ä‘á»c *_result_json tá»« AnalysisHistory.
    """
    history = get_analysis(db, x_device_id)

    if history is None:
        return ApiResponse(success=True, data=HistoryApiResponse())

    color: ColorApiResponse | None = None
    body: BodyApiResponse | None = None

    if history.color_result_json:
        color_json = json.loads(history.color_result_json)
        color = ColorApiResponse.model_validate(color_json)

    if history.body_result_json:
        body_json = json.loads(history.body_result_json)
        body = BodyApiResponse.model_validate(body_json)

    return ApiResponse(
        success=True,
        data=HistoryApiResponse(color=color, body=body),
    )

