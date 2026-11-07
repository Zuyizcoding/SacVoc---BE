"""Đọc và kiểm tra catalog từ SQLite; không tải model AI."""

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import Outfit

from .catalog_policy import MALE_GENDER_ALIASES, unique_image_outfits
from .types import SUPPORTED_EMBEDDING_BACKENDS, RecommendationNotReadyError


def _load_outfit_vectors(
    db: Session,
    occasion: str | None = None,
    gender: str | None = None,
) -> tuple[list[Outfit], np.ndarray]:
    query = db.query(Outfit).filter(Outfit.vector_blob.isnot(None))

    if occasion:
        query = query.filter(Outfit.occasion == occasion)

    if (gender or "").lower() == "male":
        query = query.filter(func.lower(Outfit.gender).in_(MALE_GENDER_ALIASES))

    outfits = query.all()
    if not outfits:
        return ([], np.empty((0, 512), dtype=np.float32))

    outfits = unique_image_outfits(outfits)

    vectors: list[np.ndarray] = []
    for outfit in outfits:
        vec = np.frombuffer(outfit.vector_blob, dtype=np.float32)
        if vec.shape != (512,):
            raise RecommendationNotReadyError(
                f"Outfit ID {outfit.id} có embedding shape không hợp lệ: {vec.shape}"
            )
        vectors.append(vec)

    return (outfits, np.stack(vectors))


def _load_outfit_embedding_meta(outfits: list[Outfit]) -> dict[str, str]:
    if not outfits:
        return {}

    meta_rows = {
        (
            str(outfit.embedding_backend or "").strip(),
            str(outfit.embedding_model or "").strip(),
            str(outfit.embedding_version or "").strip(),
        )
        for outfit in outfits
    }
    if any(
        not backend or not model or not version for backend, model, version in meta_rows
    ):
        raise RecommendationNotReadyError(
            "Outfit vectors đang thiếu metadata encoder/model/version. "
            "Hãy encode lại outfit vectors thật trước khi dùng F3."
        )
    if len(meta_rows) != 1:
        raise RecommendationNotReadyError(
            "Catalog outfit đang trộn nhiều loại embedding backend/model khác nhau. "
            "Hãy re-encode toàn bộ catalog bằng một encoder thống nhất."
        )

    backend, model, version = next(iter(meta_rows))
    if backend not in SUPPORTED_EMBEDDING_BACKENDS:
        raise RecommendationNotReadyError(
            "Catalog outfit hiện không dùng embedding thật được hỗ trợ. "
            "Dummy vectors không được phép dùng cho recommendation accuracy."
        )
    return {"backend": backend, "model": model, "version": version}
