"""
F3 Router — Outfit Recommendation API.

Tuân thủ:
  - backend-python #1: router riêng trong api/endpoints/
  - backend-python #2: Depends(get_db) inject DB Session
  - backend-python #3: Pydantic schema cho response, type hints đầy đủ
  - Tính năng #15: F1+F2→F3 combined flow — kiểm tra đã phân tích chưa
  - Tính năng #31: GET /outfits/recommend → top-6
  - Tính năng #32: GET /outfits/{id} → chi tiết outfit
  - Tính năng #33: shopee_url trong response → nút "Mua trên Shopee"
  - Tính năng #34: Error handling — 400/404/503

FLOW:
  Flutter gửi GET /api/v1/outfits/recommend
    → Header: X-Device-ID (SHA256)
    → Query param: occasion (optional, default "daily")
    → Server đọc AnalysisHistory theo device_id
    → Nếu thiếu F1 hoặc F2 → 400 + message yêu cầu phân tích
    → Nếu đủ → gọi outfit_service.recommend_outfits()
    → Trả ApiResponse[OutfitRecommendResponse]
"""
import json

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.crud import get_analysis, has_completed_f1_f2
from app.db.models import Outfit
from app.schemas.base import ApiResponse
from app.schemas.outfit import (
    OutfitDetailResponse,
    OutfitItem,
    OutfitRecommendResponse,
)
from app.services.outfit_service import RecommendationNotReadyError, recommend_outfits


router = APIRouter()


def _extract_photo_quality_score(result_json: str | None) -> float | None:
    if not result_json:
        return None
    try:
        payload = json.loads(result_json)
    except Exception:
        return None
    raw = payload.get("photo_quality_score")
    if not isinstance(raw, (int, float)):
        return None
    return max(0.0, min(float(raw), 1.0))


# ═══════════════════════════════════════════════
# GET /outfits/recommend — Tính năng #31
# ═══════════════════════════════════════════════

@router.get(
    "/outfits/recommend",
    response_model=ApiResponse[OutfitRecommendResponse],
    summary="Gợi ý 6 outfit phù hợp dựa trên F1 Color + F2 Body",
    responses={
        400: {"description": "Chưa hoàn thành F1 hoặc F2"},
        503: {"description": "Lỗi hệ thống / DB trống"},
    },
)
async def get_outfit_recommendations(
    x_device_id: str = Header(
        ...,
        alias="X-Device-ID",
        description="SHA256 device ID từ device_info_plus — Tính năng #3",
        min_length=64,
        max_length=64,
    ),
    occasion: str = Query(
        default="daily",
        description="daily | work | party | date — mặc định daily",
        regex="^(daily|work|party|date)$",
    ),
    gender: str = Query(
        default="male",
        description="Male-only mode: chỉ hỗ trợ giá trị 'male'.",
        regex="^male$",
    ),
    db: Session = Depends(get_db),
) -> ApiResponse[OutfitRecommendResponse]:
    """Gợi ý outfit dựa trên kết quả phân tích F1 + F2.

    Tính năng #15 — Combined Flow:
      1. Đọc AnalysisHistory theo device_id
      2. Kiểm tra đã có F1 (color) VÀ F2 (body) chưa
      3. Nếu thiếu → 400 Bad Request + message rõ ràng
      4. Nếu đủ → gọi outfit_service.recommend_outfits()
    """
    # ─── Bước 1: Kiểm tra F1 + F2 (Tính năng #15) ───
    has_f1, has_f2 = has_completed_f1_f2(db, x_device_id)

    if not has_f1 and not has_f2:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_analysis",
                "message": "Bạn chưa phân tích màu sắc (F1) và dáng người (F2). "
                           "Vui lòng hoàn thành cả hai trước khi xem gợi ý trang phục.",
                "missing": ["color", "body"],
            },
        )

    if not has_f1:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_color_analysis",
                "message": "Bạn chưa phân tích màu sắc cá nhân (F1). "
                           "Vui lòng chụp ảnh selfie để xác định mùa sắc trước.",
                "missing": ["color"],
            },
        )

    if not has_f2:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_body_analysis",
                "message": "Bạn chưa phân tích dáng người (F2). "
                           "Vui lòng chụp ảnh toàn thân để xác định dáng trước.",
                "missing": ["body"],
            },
        )

    # ─── Bước 2: Đọc kết quả F1+F2 từ DB ───
    history = get_analysis(db, x_device_id)

    # ─── Bước 3: Gọi Outfit Service ───
    try:
        # Male-only patch: always lock gender filter to male.
        effective_gender = "male"

        result = recommend_outfits(
            db=db,
            color_sub_season=history.color_sub_season,
            body_shape=history.body_shape,
            occasion=occasion,
            gender=effective_gender,
            top_k=6,
            color_confidence=history.color_confidence,
            body_confidence=history.body_confidence,
            color_photo_quality_score=_extract_photo_quality_score(history.color_result_json),
            body_photo_quality_score=_extract_photo_quality_score(history.body_result_json),
        )

        return ApiResponse(success=True, data=result)

    except RecommendationNotReadyError as exc:
        return ApiResponse(
            success=False,
            data=None,
            error=str(exc),
            error_code="RECOMMENDATION_NOT_READY",
        )

    except Exception as e:
        # Tính năng #34: Error handling — log + trả 503
        # import logging; logging.exception("F3 outfit recommend error")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "outfit_service_error",
                "message": f"Lỗi hệ thống khi gợi ý trang phục. Vui lòng thử lại. ({e})",
            },
        )


# ═══════════════════════════════════════════════
# GET /outfits/{id} — Tính năng #32
# ═══════════════════════════════════════════════

@router.get(
    "/outfits/{outfit_id}",
    response_model=ApiResponse[OutfitDetailResponse],
    summary="Chi tiết 1 outfit",
    responses={404: {"description": "Outfit không tồn tại"}},
)
async def get_outfit_detail(
    outfit_id: int,
    db: Session = Depends(get_db),
) -> ApiResponse[OutfitDetailResponse]:
    """Chi tiết outfit — ảnh lớn + tên + category + giá + link Shopee.

    Tính năng #32: MUST — hiển thị thông tin static từ seed data.
    SHOULD: Thêm `explanation` từ Gemini real-time (file ENHANCED_SHOULD).
    """
    outfit = db.query(Outfit).filter(Outfit.id == outfit_id).first()

    if outfit is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "outfit_not_found",
                "message": f"Không tìm thấy trang phục với ID {outfit_id}.",
            },
        )

    detail = OutfitDetailResponse(
        id=outfit.id,
        name=outfit.name,
        category=outfit.category,
        image_url=outfit.image_url,
        shopee_url=outfit.shopee_url,
        price_range=outfit.price_range,
        explanation=None,  # MUST: None. SHOULD: Gemini real-time (xem ENHANCED_SHOULD)
    )

    return ApiResponse(success=True, data=detail)

