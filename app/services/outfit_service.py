"""Điều phối F3: tạo query → lấy vector/catalog → xếp hạng → trả kết quả."""

import json
import logging

from sqlalchemy.orm import Session

from app.schemas.outfit import OutfitItem, OutfitRecommendResponse
from app.services.recommendation.catalog import (
    _load_outfit_embedding_meta,
    _load_outfit_vectors,
)
from app.services.recommendation.embedding_store import (
    _assert_query_vector_consistency,
    _resolve_query_vector,
    _resolve_query_vectors_path,
)
from app.services.recommendation.query_builder import build_outfit_query
from app.services.recommendation.ranking import (
    _cosine_similarity_search,
    _rank_candidates,
    _select_diverse_top_k,
)
from app.services.recommendation.types import (
    RecommendationNotReadyError as RecommendationNotReadyError,
)

logger = logging.getLogger(__name__)


def recommend_outfits(
    db: Session,
    color_sub_season: str,
    body_shape: str,
    occasion: str = "daily",
    gender: str = "male",
    top_k: int = 6,
    text_encoder=None,
    color_confidence: float | None = None,
    body_confidence: float | None = None,
    color_photo_quality_score: float | None = None,
    body_photo_quality_score: float | None = None,
) -> OutfitRecommendResponse:
    query_text = build_outfit_query(color_sub_season, body_shape, occasion)
    query_bundle = _resolve_query_vector(query_text, text_encoder=text_encoder)

    outfits, outfit_matrix = _load_outfit_vectors(db, occasion=occasion, gender=gender)
    if len(outfits) == 0:
        raise RecommendationNotReadyError(
            "Dữ liệu outfit chưa sẵn sàng cho gợi ý. "
            "Hãy seed outfit và encode vector_blob thật trước khi dùng F3."
        )

    outfit_meta = _load_outfit_embedding_meta(outfits)
    _assert_query_vector_consistency(query_bundle, outfit_meta)

    raw_scores = _cosine_similarity_search(query_bundle.vector, outfit_matrix)
    ranked = _rank_candidates(
        outfits,
        raw_scores,
        color_sub_season=color_sub_season,
        body_shape=body_shape,
        occasion=occasion,
        color_confidence=color_confidence,
        body_confidence=body_confidence,
        color_photo_quality_score=color_photo_quality_score,
        body_photo_quality_score=body_photo_quality_score,
    )
    selected = _select_diverse_top_k(ranked, top_k)

    logger.info(
        "F3 recommend diagnostics=%s",
        json.dumps(
            {
                "query": query_text,
                "query_source": query_bundle.source,
                "query_backend": query_bundle.backend,
                "query_model": query_bundle.model,
                "color_confidence": color_confidence,
                "body_confidence": body_confidence,
                "color_photo_quality_score": color_photo_quality_score,
                "body_photo_quality_score": body_photo_quality_score,
                "outfit_backend": outfit_meta["backend"],
                "outfit_model": outfit_meta["model"],
                "candidate_count": len(ranked),
                "selected": [
                    {
                        "id": candidate.outfit.id,
                        "score": round(candidate.final_score, 4),
                        "retrieval_score": round(candidate.retrieval_score, 4),
                        "reasons": list(candidate.reason_tags),
                    }
                    for candidate in selected
                ],
            },
            ensure_ascii=False,
        ),
    )

    outfit_items = [
        OutfitItem(
            id=candidate.outfit.id,
            name=candidate.outfit.name,
            category=candidate.outfit.category,
            image_url=candidate.outfit.image_url,
            shopee_url=candidate.outfit.shopee_url,
            price_range=candidate.outfit.price_range,
            score=round(candidate.final_score, 4),
            retrieval_score=round(candidate.retrieval_score, 4),
            reason_tags=list(candidate.reason_tags) or None,
        )
        for candidate in selected
    ]

    return OutfitRecommendResponse(
        query_used=query_text,
        query_source=query_bundle.source,
        outfits=outfit_items,
    )


def get_recommendation_readiness(
    db: Session,
    *,
    occasion: str = "daily",
    gender: str = "male",
) -> dict[str, object]:
    checks: dict[str, object] = {
        "catalog_seeded": False,
        "vector_blob_available": False,
        "embedding_meta_valid": False,
        "query_vector_available": False,
        "query_vector_consistent": False,
        "query_vector_source": None,
        "outfit_count": 0,
        "query_vectors_path": str(_resolve_query_vectors_path()),
        "errors": [],
    }
    errors: list[str] = []

    try:
        outfits, _ = _load_outfit_vectors(db, occasion=occasion, gender=gender)
        checks["outfit_count"] = len(outfits)
        checks["catalog_seeded"] = len(outfits) > 0
        if outfits:
            checks["vector_blob_available"] = True
            meta = _load_outfit_embedding_meta(outfits)
            checks["embedding_meta_valid"] = True
            sample_query = build_outfit_query("Soft Autumn", "Rectangle", occasion)
            query_bundle = _resolve_query_vector(sample_query)
            checks["query_vector_available"] = True
            checks["query_vector_source"] = query_bundle.source
            _assert_query_vector_consistency(query_bundle, meta)
            checks["query_vector_consistent"] = True
    except RecommendationNotReadyError as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"Unexpected readiness error: {exc}")

    checks["errors"] = errors
    checks["ready"] = (
        bool(checks["catalog_seeded"])
        and bool(checks["vector_blob_available"])
        and bool(checks["embedding_meta_valid"])
        and bool(checks["query_vector_available"])
        and bool(checks["query_vector_consistent"])
        and not errors
    )
    return checks
