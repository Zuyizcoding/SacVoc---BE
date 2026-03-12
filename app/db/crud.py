"""
CRUD operations cho AnalysisHistory.

Tuân thủ:
  - backend-python #2: nhận Session qua tham số (inject từ Depends)
  - backend-python #3: Type Hints đầy đủ

Tính năng #15: F3 đọc sub_season + shape từ AnalysisHistory
SHOULD: Màn "Xem lại kết quả" đọc *_result_json từ đây
"""
import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import AnalysisHistory


def get_analysis(db: Session, device_id: str) -> AnalysisHistory | None:
    """Lấy lịch sử phân tích theo device_id.

    Returns:
        AnalysisHistory hoặc None nếu device chưa từng phân tích.
    """
    return db.query(AnalysisHistory).filter(
        AnalysisHistory.device_id == device_id
    ).first()


def save_color_result(
    db: Session,
    device_id: str,
    sub_season: str,
    primary_season: str,
    confidence: float,
    full_result: dict,
) -> AnalysisHistory:
    """Lưu/cập nhật kết quả F1 Color Analysis.

    Nếu device_id đã tồn tại → UPDATE (phân tích lại).
    Nếu chưa có → INSERT mới.

    Args:
        db: SQLAlchemy Session (từ Depends(get_db))
        device_id: SHA256 hash từ device_info_plus
        sub_season: VD "Soft Autumn"
        primary_season: VD "Autumn"
        confidence: 0.0 → 1.0
        full_result: Toàn bộ dict response — serialize thành JSON text

    Returns:
        AnalysisHistory đã lưu
    """
    history = get_analysis(db, device_id)

    if history is None:
        # Thiết bị mới — tạo row
        history = AnalysisHistory(device_id=device_id)
        db.add(history)

    # Cập nhật fields F1
    history.color_sub_season = sub_season
    history.color_primary_season = primary_season
    history.color_confidence = confidence
    history.color_result_json = json.dumps(full_result, ensure_ascii=False)
    history.color_analyzed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(history)
    return history


def save_body_result(
    db: Session,
    device_id: str,
    shape: str,
    confidence: float,
    full_result: dict,
) -> AnalysisHistory:
    """Lưu/cập nhật kết quả F2 Body Analysis.

    Tương tự save_color_result — UPDATE nếu đã có, INSERT nếu chưa.
    """
    history = get_analysis(db, device_id)

    if history is None:
        history = AnalysisHistory(device_id=device_id)
        db.add(history)

    history.body_shape = shape
    history.body_confidence = confidence
    history.body_result_json = json.dumps(full_result, ensure_ascii=False)
    history.body_analyzed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(history)
    return history


def has_completed_f1_f2(db: Session, device_id: str) -> tuple[bool, bool]:
    """Kiểm tra device đã hoàn thành F1 và F2 chưa.

    Tính năng #15: Nếu thiếu F1 hoặc F2 → F3 trả lỗi yêu cầu phân tích trước.

    Returns:
        (has_f1, has_f2) — tuple 2 bool
    """
    history = get_analysis(db, device_id)
    if history is None:
        return (False, False)

    has_f1 = history.color_sub_season is not None
    has_f2 = history.body_shape is not None
    return (has_f1, has_f2)
