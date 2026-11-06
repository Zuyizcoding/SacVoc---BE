import numpy as np
import pytest

from app.db.models import Outfit
from app.services.recommendation.catalog import _load_outfit_embedding_meta
from app.services.recommendation.embedding_store import _assert_query_vector_consistency
from app.services.recommendation.query_builder import build_outfit_query
from app.services.recommendation.ranking import (
    _rank_candidates,
    _season_alignment_score,
    _select_diverse_top_k,
)
from app.services.recommendation.types import (
    QueryVectorBundle,
    RecommendationNotReadyError,
)


def _make_outfit(
    *,
    outfit_id: int,
    name: str,
    category: str = "top",
    image_url: str = "https://example.com/a.jpg",
    season_bias: str = "Autumn",
    body_shape_bias: str = "Rectangle",
    fit: str = "regular",
    formality: str = "casual",
    color_family: str = "earth",
    embedding_backend: str = "fashion_clip",
    embedding_model: str = "fashion-clip",
    embedding_version: str = "2026-05-mens-v1",
) -> Outfit:
    return Outfit(
        id=outfit_id,
        name=name,
        category=category,
        image_url=image_url,
        shopee_url="https://example.com/item",
        price_range="200k - 500k",
        occasion="daily",
        gender="male",
        season_bias=season_bias,
        body_shape_bias=body_shape_bias,
        fit=fit,
        formality=formality,
        color_family=color_family,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_version=embedding_version,
        vector_blob=np.ones(512, dtype=np.float32).tobytes(),
    )


def test_build_outfit_query_is_menswear_focused() -> None:
    query = build_outfit_query("Soft Autumn", "Rectangle", "daily")
    assert "men's outfit" in query
    assert "Rectangle" in query
    assert "streetwear" in query


def test_load_outfit_embedding_meta_rejects_dummy_vectors() -> None:
    outfit = _make_outfit(
        outfit_id=1,
        name="Dummy",
        embedding_backend="dummy",
        embedding_model="sha256-seeded-rng",
        embedding_version="2026-05-dummy",
    )
    with pytest.raises(RecommendationNotReadyError):
        _load_outfit_embedding_meta([outfit])


def test_assert_query_vector_consistency_rejects_model_mismatch() -> None:
    bundle = QueryVectorBundle(
        vector=np.ones(512, dtype=np.float32),
        source="precomputed_query_vector",
        backend="hf_clip",
        model="openai/clip-vit-base-patch32",
        version="2026-05-mens-v1",
        query_signature="abc",
    )
    with pytest.raises(RecommendationNotReadyError):
        _assert_query_vector_consistency(
            bundle,
            {
                "backend": "fashion_clip",
                "model": "fashion-clip",
                "version": "2026-05-mens-v1",
            },
        )


def test_diversity_selection_skips_duplicate_images() -> None:
    outfits = [
        _make_outfit(outfit_id=1, name="A", category="top", image_url="same.jpg"),
        _make_outfit(outfit_id=2, name="B", category="top", image_url="same.jpg"),
        _make_outfit(outfit_id=3, name="C", category="bottom", image_url="bottom.jpg"),
    ]
    ranked = _rank_candidates(
        outfits,
        np.array([0.95, 0.92, 0.88], dtype=np.float32),
        color_sub_season="Soft Autumn",
        body_shape="Rectangle",
        occasion="daily",
    )
    selected = _select_diverse_top_k(ranked, top_k=3)
    assert len(selected) == 2
    assert {candidate.outfit.id for candidate in selected} == {1, 3}


def test_season_alignment_score_preserves_primary_season_bonus_for_explicit_subseasons() -> (
    None
):
    outfit = _make_outfit(outfit_id=1, name="A", season_bias="Soft Autumn|True Autumn")
    score, reasons = _season_alignment_score(outfit, "Deep Autumn")
    assert score == pytest.approx(0.82)
    assert reasons == ["primary_season"]


def test_season_alignment_score_returns_exact_match_for_explicit_subseason() -> None:
    outfit = _make_outfit(outfit_id=1, name="A", season_bias="Soft Autumn|True Autumn")
    score, reasons = _season_alignment_score(outfit, "Soft Autumn")
    assert score == pytest.approx(1.0)
    assert reasons == ["exact_season"]


def test_season_alignment_score_returns_mismatch_for_cross_season_bias() -> None:
    outfit = _make_outfit(outfit_id=1, name="A", season_bias="True Winter|Deep Winter")
    score, reasons = _season_alignment_score(outfit, "Soft Autumn")
    assert score == pytest.approx(0.15)
    assert reasons == []


def test_rank_candidates_prefers_same_primary_season_with_explicit_subseason_biases() -> (
    None
):
    outfits = [
        _make_outfit(
            outfit_id=1, name="Same family", season_bias="Soft Autumn|True Autumn"
        ),
        _make_outfit(
            outfit_id=2, name="Different family", season_bias="True Winter|Deep Winter"
        ),
    ]
    raw_scores = np.array([0.9, 0.9], dtype=np.float32)
    ranked = _rank_candidates(
        outfits,
        raw_scores,
        color_sub_season="Deep Autumn",
        body_shape="Rectangle",
        occasion="daily",
    )
    assert ranked[0].outfit.id == 1
    assert ranked[0].final_score > ranked[1].final_score
    assert "primary_season" in ranked[0].reason_tags


def test_rank_candidates_softens_rule_bias_when_analysis_confidence_is_low() -> None:
    outfits = [_make_outfit(outfit_id=1, name="A", category="top", image_url="a.jpg")]
    raw_scores = np.array([0.9], dtype=np.float32)
    high_conf = _rank_candidates(
        outfits,
        raw_scores,
        color_sub_season="Soft Autumn",
        body_shape="Rectangle",
        occasion="daily",
        color_confidence=0.9,
        body_confidence=0.9,
    )[0]
    low_conf = _rank_candidates(
        outfits,
        raw_scores,
        color_sub_season="Soft Autumn",
        body_shape="Rectangle",
        occasion="daily",
        color_confidence=0.35,
        body_confidence=0.35,
    )[0]
    assert low_conf.final_score < high_conf.final_score
    assert "low_color_confidence" in low_conf.reason_tags
    assert "low_body_confidence" in low_conf.reason_tags


def test_rank_candidates_uses_photo_quality_score_in_downstream_weighting() -> None:
    outfits = [_make_outfit(outfit_id=1, name="A", category="top", image_url="a.jpg")]
    raw_scores = np.array([0.9], dtype=np.float32)
    high_quality = _rank_candidates(
        outfits,
        raw_scores,
        color_sub_season="Soft Autumn",
        body_shape="Rectangle",
        occasion="daily",
        color_confidence=0.8,
        body_confidence=0.8,
        color_photo_quality_score=0.9,
        body_photo_quality_score=0.88,
    )[0]
    low_quality = _rank_candidates(
        outfits,
        raw_scores,
        color_sub_season="Soft Autumn",
        body_shape="Rectangle",
        occasion="daily",
        color_confidence=0.8,
        body_confidence=0.8,
        color_photo_quality_score=0.42,
        body_photo_quality_score=0.46,
    )[0]
    assert low_quality.final_score < high_quality.final_score
    assert "low_color_photo_quality" in low_quality.reason_tags
    assert "low_body_photo_quality" in low_quality.reason_tags
