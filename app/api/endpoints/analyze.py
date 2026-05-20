"""
Router phân tích ảnh — Color (F1) + Body (F2).

Tuân thủ:
  - backend-python #1: Router riêng biệt cho /analyze
  - backend-python #2: Depends() inject GeminiClient + DB Session
  - Tính năng #9: try/finally xóa ảnh tạm — KHÔNG lưu ảnh người dùng
  - Tính năng #34: HTTP 503/422/429/400 → thông báo tiếng Việt rõ ràng
  - Tính năng #15: F1+F2→F3 combined flow — đọc kết quả từ DB

Luồng xử lý:
  1. Nhận ảnh upload (UploadFile)
  2. Validate kích thước (≥400×400px)
  3. Compress (≤1MB JPEG)
  4. Gửi Gemini qua GeminiClient
  5. Override palette từ season_data.py (F1)
  6. Lưu kết quả vào ANALYSIS_HISTORY
  7. finally: XÓA ảnh tạm (dù thành công hay thất bại)
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
from app.utils.image_utils import validate_image, compress_image
from app.db.database import get_db
from app.db.crud import get_analysis, save_color_result, save_body_result
from app.data.season_data import (
    canonicalize_sub_season,
    get_season_entry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analyze", tags=["Analysis"])


def _canonicalize_color_result(result: dict) -> dict:
    """Chuẩn hóa F1 bằng bảng season_data trước khi lưu/trả về."""
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
    """Merge partial F2 payloads, ưu tiên dữ liệu mới khi field có giá trị."""
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
    """Prompt ngắn để yêu cầu Gemini chỉ hoàn thiện field F2 còn thiếu."""
    missing_text = ", ".join(missing_fields)
    return (
        "<task>\n"
        "Bạn vừa phân tích body shape từ ảnh này nhưng còn thiếu một số field bắt buộc.\n"
        f"Hãy phân tích lại cùng ảnh và trả JSON hoàn chỉnh, tập trung bổ sung: {missing_text}.\n"
        "Nếu ảnh không đủ rõ để kết luận chắc chắn, hãy trả JSON lỗi CLOTHING_TOO_WIDE hoặc giữ confidence thấp,\n"
        "nhưng không được bịa mẹo mặc đồ chung chung.\n"
        f"Partial result trước đó để tham chiếu: {json.dumps(partial_result, ensure_ascii=False)}\n"
        "</task>"
    )


# ──────────────────────────────────────────────
# POST /api/v1/analyze/color (Tính năng #16, #17, #18, #19)
# ──────────────────────────────────────────────
@router.post(
    "/color",
    response_model=ApiResponse[ColorApiResponse],
    summary="F1 — Phân tích màu sắc cá nhân",
    description="Nhận ảnh selfie, trả về 12-season + palette + makeup guide.",
)
async def analyze_color(
    file: UploadFile = File(..., description="Ảnh selfie JPEG/PNG"),
    x_device_id: str = Header(
        ..., description="SHA256 device ID (Tính năng #3)"
    ),
    client: GeminiClient = Depends(get_gemini_client),
    db: Session = Depends(get_db),
) -> ApiResponse[ColorApiResponse]:
    """Phân tích màu sắc cá nhân từ ảnh selfie.

    Flow:
        Upload → Validate → Compress → Gemini → Override palette → Response
    """
    tmp_path: str | None = None

    try:
        # --- Bước 1: Đọc file upload ---
        file_bytes = await file.read()
        logger.info(
            "📸 Color analysis request | device=%s | size=%d bytes",
            x_device_id[:8],
            len(file_bytes),
        )

        # --- Bước 2: Validate ảnh (Tính năng #8) ---
        validate_image(file_bytes)

        # --- Bước 3: Compress ≤1MB (Tính năng #8) ---
        compressed = compress_image(file_bytes, max_kb=1000)

        # --- Bước 4: Lưu tạm (để log/debug nếu cần) ---
        with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False
        ) as tmp:
            tmp.write(compressed)
            tmp_path = tmp.name

        # --- Bước 5: Gọi Gemini (Tính năng #10, #11, #12) ---
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
                    "Không thể phân tích ảnh. Vui lòng thử lại với ảnh rõ hơn.",
                ),
                error_code=422,
            )

        # --- Bước 6: Canonicalize F1 từ season_data.py (Tính năng #20) ---
        result = _canonicalize_color_result(result)

        # --- Bước 7: Lưu vào ANALYSIS_HISTORY ---
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

    # ─── Error Handling (Tính năng #34) ────────────────
    except HTTPException:
        raise  # 422 từ validate_image — giữ nguyên

    except TimeoutError:
        logger.error("⏱️ Gemini timeout cho F1")
        raise HTTPException(
            status_code=503,
            detail="AI đang bận. Vui lòng thử lại sau 5 giây.",
        )

    except DeadlineExceeded:
        logger.error("⏱️ Gemini timeout (DeadlineExceeded) cho F1")
        raise HTTPException(
            status_code=503,
            detail="AI đang bận. Vui lòng thử lại sau 5 giây.",
        )

    except ValueError as exc:
        logger.error("🔴 Gemini JSON parse error: %s", str(exc))
        raise HTTPException(
            status_code=400,
            detail="Không phân tích được ảnh. Vui lòng chụp ảnh rõ hơn.",
        )

    except Exception as exc:
        logger.exception("💥 Lỗi không xác định trong F1: %s", str(exc))
        raise HTTPException(
            status_code=503,
            detail="Hệ thống gặp sự cố. Vui lòng thử lại sau.",
        )

    # ─── Photo Cleanup (Tính năng #9) ─────────────────
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info("🗑️ Đã xóa ảnh tạm: %s", tmp_path)


# ──────────────────────────────────────────────
# POST /api/v1/analyze/body (Tính năng #23, #24, #25, #26)
# ──────────────────────────────────────────────
@router.post(
    "/body",
    response_model=ApiResponse[BodyApiResponse],
    summary="F2 — Phân tích dáng người",
    description="Nhận ảnh toàn thân, trả về body shape + styling tips.",
)
async def analyze_body(
    file: UploadFile = File(..., description="Ảnh toàn thân JPEG/PNG"),
    x_device_id: str = Header(
        ..., description="SHA256 device ID (Tính năng #3)"
    ),
    client: GeminiClient = Depends(get_gemini_client),
    db: Session = Depends(get_db),
) -> ApiResponse[BodyApiResponse]:
    """Phân tích dáng người từ ảnh toàn thân.

    Flow tương tự F1 — copy pattern try/finally.
    """
    tmp_path: str | None = None

    try:
        file_bytes = await file.read()
        logger.info(
            "📸 Body analysis request | device=%s | size=%d bytes",
            x_device_id[:8],
            len(file_bytes),
        )

        validate_image(file_bytes)
        compressed = compress_image(file_bytes, max_kb=1000)

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

        # If Gemini returns a structured error, propagate as an API-level error
        # (frontend already expects `success=false` with `error` message).
        if result.get("error_code") is not None:
            return ApiResponse(
                success=False,
                data=None,
                error=result.get(
                    "message",
                    "Không thể phân tích dáng người chính xác do trang phục quá rộng.",
                ),
                error_code=422,
            )

        parsed_result = BodyGeminiResponse.model_validate(result)
        missing_fields = parsed_result.missing_detail_fields()
        if missing_fields:
            logger.warning(
                "F2 thiếu field grounded sau lượt đầu: %s | device=%s",
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
                return ApiResponse(
                    success=False,
                    data=None,
                    error=repaired.get(
                        "message",
                        "Không thể phân tích dáng người chính xác từ ảnh hiện tại.",
                    ),
                    error_code=422,
                )

            result = _merge_body_result(result, repaired)
            parsed_result = BodyGeminiResponse.model_validate(result)
            missing_fields = parsed_result.missing_detail_fields()

        if missing_fields:
            logger.warning(
                "F2 vẫn thiếu field sau repair pass: %s | device=%s",
                missing_fields,
                x_device_id[:8],
            )
            return ApiResponse(
                success=False,
                data=None,
                error=(
                    "Ảnh chưa đủ rõ để đưa ra gợi ý dáng người đáng tin cậy. "
                    "Vui lòng dùng ảnh toàn thân, đứng thẳng và mặc đồ ôm sát hơn."
                ),
                error_code=422,
            )

        response_payload = parsed_result.model_dump()
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
            detail="AI đang bận. Vui lòng thử lại sau 5 giây.",
        )

    except DeadlineExceeded:
        raise HTTPException(
            status_code=503,
            detail="AI đang bận. Vui lòng thử lại sau 5 giây.",
        )

    except ValueError as exc:
        logger.error("🔴 Body JSON parse error: %s", str(exc))
        raise HTTPException(
            status_code=400,
            detail="Không phân tích được ảnh. Chụp ảnh toàn thân rõ hơn nhé!",
        )

    except Exception as exc:
        logger.exception("💥 Lỗi không xác định trong F2: %s", str(exc))
        raise HTTPException(
            status_code=503,
            detail="Hệ thống gặp sự cố. Vui lòng thử lại sau.",
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info("🗑️ Đã xóa ảnh tạm: %s", tmp_path)


# ──────────────────────────────────────────────
# GET /api/v1/analyze/history (SHOULD — Xem lại kết quả)
# ──────────────────────────────────────────────
@router.get(
    "/history",
    response_model=ApiResponse[HistoryApiResponse],
    summary="Xem lại kết quả phân tích F1 + F2",
    description="Đọc kết quả phân tích đã lưu trong AnalysisHistory.",
)
async def get_analysis_history(
    x_device_id: str = Header(
        ..., description="SHA256 device ID (Tính năng #3)"
    ),
    db: Session = Depends(get_db),
) -> ApiResponse[HistoryApiResponse]:
    """Trả về lịch sử F1 + F2 nếu đã phân tích.

    SHOULD: Màn "Xem lại kết quả" đọc *_result_json từ AnalysisHistory.
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
