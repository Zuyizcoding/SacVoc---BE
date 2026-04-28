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
from sqlalchemy.orm import Session

from app.schemas.base import ApiResponse
from app.schemas.color import ColorApiResponse, ColorGeminiResponse
from app.schemas.body import BodyApiResponse, BodyGeminiResponse
from app.services.gemini_client import GeminiClient, get_gemini_client
from app.schemas.history import HistoryApiResponse
from app.services.demo_cache import get_demo_color_result, get_demo_body_result
from app.utils.image_utils import validate_image, compress_image
from app.db.database import get_db
from app.db.crud import get_analysis, save_color_result, save_body_result
from app.data.season_data import get_palette_override

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analyze", tags=["Analysis"])


# ──────────────────────────────────────────────
# Prompts — sẽ chuyển sang file riêng ở Sprint 2
# ──────────────────────────────────────────────
# TODO: Chuyển sang src/backend/app/core/prompts.py (xem ENHANCED_SHOULD_IMPLEMENTATION)
SYSTEM_INSTRUCTION_F1 = (
    "Bạn là chuyên gia phân tích màu sắc cá nhân (Personal Color Analysis). "
    "Luôn trả lời bằng JSON hợp lệ theo schema được cung cấp. "
    "Khi không chắc chắn, hạ confidence xuống thay vì đoán."
)

PROMPT_F1 = """
<task>
Phân tích seasonal color type của người trong ảnh này.
Phân tích TUẦN TỰ: undertone da → độ sáng → màu tóc → màu mắt → kết luận sub-season.
</task>
"""

SYSTEM_INSTRUCTION_F2 = (
    "Bạn là chuyên gia tư vấn phong cách thời trang. "
    "Luôn trả lời bằng JSON hợp lệ theo schema được cung cấp. "
    "Styling tips phải bằng tiếng Việt tự nhiên, cụ thể và cá nhân hóa."
)

PROMPT_F2 = """
<task>
Phân tích body shape của người trong ảnh này.
Tập trung vào tỷ lệ: vai, eo, hông.
Đưa ra mẹo mặc đồ cá nhân hóa dựa trên đặc điểm cụ thể quan sát được.
</task>
"""


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
        try:
            result = await client.analyze(
                image_bytes=compressed,
                prompt=PROMPT_F1,
                response_schema=ColorGeminiResponse,
                system_instruction=SYSTEM_INSTRUCTION_F1,
            )
        except RuntimeError as exc:
            if "CIRCUIT_BREAKER_OPEN" in str(exc):
                # Tính năng #13 + #14: fallback demo cache
                logger.warning("⚡ Dùng demo cache cho F1")
                result = get_demo_color_result()
            else:
                raise

        # --- Bước 6: Override palette từ season_data.py (Tính năng #20) ---
        result["palette"] = get_palette_override(result["sub_season"])

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

        try:
            result = await client.analyze(
                image_bytes=compressed,
                prompt=PROMPT_F2,
                response_schema=BodyGeminiResponse,
                system_instruction=SYSTEM_INSTRUCTION_F2,
            )
        except RuntimeError as exc:
            if "CIRCUIT_BREAKER_OPEN" in str(exc):
                logger.warning("⚡ Dùng demo cache cho F2")
                result = get_demo_body_result()
            else:
                raise

        save_body_result(
            db=db,
            device_id=x_device_id,
            shape=result["shape"],
            confidence=result["confidence"],
            full_result=result,
        )

        return ApiResponse(
            success=True,
            data=BodyApiResponse.model_validate(result),
        )

    except HTTPException:
        raise

    except TimeoutError:
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
