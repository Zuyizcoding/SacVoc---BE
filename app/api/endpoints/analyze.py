"""HTTP adapter F1/F2: nhận request và chuyển lỗi nghiệp vụ sang API cũ."""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.crud import get_analysis
from app.db.database import get_db
from app.schemas.base import ApiResponse
from app.schemas.body import BodyApiResponse
from app.schemas.color import ColorApiResponse
from app.schemas.history import HistoryApiResponse
from app.services.analysis.errors import (
    AnalysisRejectedError,
    AnalysisUnavailableError,
    PhotoValidationError,
)
from app.services.analysis.workflows import analyze_body_image, analyze_color_image
from app.services.gemini_client import GeminiClient, get_gemini_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/analyze", tags=["Analysis"])
Result = TypeVar("Result", ColorApiResponse, BodyApiResponse)


async def _respond(
    file: UploadFile,
    operation: Callable[[bytes], Awaitable[Result]],
    subject: str,
) -> ApiResponse[Result]:
    try:
        return ApiResponse(success=True, data=await operation(await file.read()))
    except AnalysisRejectedError as exc:
        return ApiResponse(success=False, error=exc.message, error_code=exc.error_code)
    except (PhotoValidationError, AnalysisUnavailableError) as exc:
        status = 422 if isinstance(exc, PhotoValidationError) else 503
        raise HTTPException(
            status, detail={"error_code": exc.error_code, "message": exc.message}
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            503,
            detail={
                "error_code": "AI_INVALID_RESULT",
                "message": f"AI trả về kết quả chưa nhất quán cho phân tích {subject}. "
                "Vui lòng thử lại sau vài giây.",
            },
        ) from exc
    except Exception as exc:
        logger.exception("Phân tích ảnh thất bại")
        raise HTTPException(
            503,
            detail={
                "error_code": "ANALYSIS_FAILED",
                "message": "Hệ thống gặp sự cố. Vui lòng thử lại sau.",
            },
        ) from exc


@router.post(
    "/color",
    response_model=ApiResponse[ColorApiResponse],
    summary="F1 - Phân tích màu sắc cá nhân",
)
async def analyze_color(
    file: UploadFile = File(..., description="Ảnh selfie JPEG/PNG"),
    x_device_id: str = Header(..., description="SHA256 device ID"),
    client: GeminiClient = Depends(get_gemini_client),
    db: Session = Depends(get_db),
) -> ApiResponse[ColorApiResponse]:
    return await _respond(
        file,
        lambda image_bytes: analyze_color_image(image_bytes, x_device_id, db, client),
        "màu sắc",
    )


@router.post(
    "/body",
    response_model=ApiResponse[BodyApiResponse],
    summary="F2 - Phân tích dáng người",
)
async def analyze_body(
    file: UploadFile = File(..., description="Ảnh toàn thân JPEG/PNG"),
    x_device_id: str = Header(..., description="SHA256 device ID"),
    client: GeminiClient = Depends(get_gemini_client),
    db: Session = Depends(get_db),
) -> ApiResponse[BodyApiResponse]:
    return await _respond(
        file,
        lambda image_bytes: analyze_body_image(image_bytes, x_device_id, db, client),
        "dáng người",
    )


@router.get(
    "/history",
    response_model=ApiResponse[HistoryApiResponse],
    summary="Xem lại kết quả phân tích F1 + F2",
)
async def get_analysis_history(
    x_device_id: str = Header(..., description="SHA256 device ID"),
    db: Session = Depends(get_db),
) -> ApiResponse[HistoryApiResponse]:
    history = get_analysis(db, x_device_id)
    if history is None:
        return ApiResponse(success=True, data=HistoryApiResponse())
    color = (
        ColorApiResponse.model_validate(json.loads(history.color_result_json))
        if history.color_result_json
        else None
    )
    body = (
        BodyApiResponse.model_validate(json.loads(history.body_result_json))
        if history.body_result_json
        else None
    )
    return ApiResponse(success=True, data=HistoryApiResponse(color=color, body=body))
