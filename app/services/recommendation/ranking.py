"""Chấm điểm và chọn đa dạng; bảo toàn trọng số đang dùng cho demo."""

from collections import Counter
from collections.abc import Sequence

import numpy as np

from app.data.body_shapes import canonicalize_recommendation_shape

from .query_builder import (
    BODY_SHAPE_COMPATIBILITY,
    SEASON_COLOR_FAMILY,
    _parse_bias_tags,
    _primary_season,
)
from .types import RankableOutfit, RankedCandidate

# Trọng số heuristic hiện tại; chưa phải xác suất đúng hoặc trọng số đã học.
SEMANTIC_WEIGHT = 0.56
SEASON_WEIGHT = 0.16
BODY_SHAPE_WEIGHT = 0.14
COLOR_FAMILY_WEIGHT = 0.06
FIT_WEIGHT = 0.04
FORMALITY_WEIGHT = 0.04


def _confidence_signal(
    confidence: float | None,
    *,
    photo_quality_score: float | None = None,
    fallback: float = 0.75,
) -> float:
    if confidence is None:
        signal = fallback
    else:
        signal = float(confidence)

    signal = max(0.35, min(signal, 1.0))
    if photo_quality_score is None:
        return signal

    quality_signal = max(0.25, min(float(photo_quality_score), 1.0))
    return max(0.3, min(0.7 * signal + 0.3 * quality_signal, 1.0))


def _cosine_similarity_search(
    query_vector: np.ndarray, outfit_matrix: np.ndarray
) -> np.ndarray:
    if outfit_matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    return outfit_matrix @ query_vector


def _season_alignment_score(
    outfit: RankableOutfit, color_sub_season: str
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    bias = _parse_bias_tags(outfit.season_bias)
    if not bias:
        return (0.4, reasons)

    sub_season = color_sub_season.casefold()
    primary = _primary_season(color_sub_season).casefold()
    if sub_season and sub_season in bias:
        reasons.append("exact_season")
        return (1.0, reasons)
    bias_primary_seasons = {
        _primary_season(tag).casefold() for tag in bias if _primary_season(tag)
    }
    if primary and (primary in bias or primary in bias_primary_seasons):
        reasons.append("primary_season")
        return (0.82, reasons)
    return (0.15, reasons)


def _body_shape_alignment_score(
    outfit: RankableOutfit, body_shape: str
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    bias = _parse_bias_tags(outfit.body_shape_bias)
    if not bias:
        return (0.4, reasons)

    canonical = canonicalize_recommendation_shape(body_shape)
    key = canonical.casefold()
    if key in bias:
        reasons.append("exact_shape")
        return (1.0, reasons)
    compatible = BODY_SHAPE_COMPATIBILITY.get(canonical, set())
    if bias.intersection({item.casefold() for item in compatible}):
        reasons.append("compatible_shape")
        return (0.72, reasons)
    return (0.15, reasons)


def _color_family_bonus(
    outfit: RankableOutfit, color_sub_season: str
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    family = str(outfit.color_family or "").strip().casefold()
    if not family:
        return (0.4, reasons)
    primary = _primary_season(color_sub_season)
    preferred = {item.casefold() for item in SEASON_COLOR_FAMILY.get(primary, set())}
    if family in preferred:
        reasons.append("color_family_match")
        return (1.0, reasons)
    return (0.25, reasons)


def _fit_bonus(outfit: RankableOutfit, body_shape: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    fit = str(outfit.fit or "").strip().casefold()
    canonical = canonicalize_recommendation_shape(body_shape)
    if not fit:
        return (0.4, reasons)
    preferred_fit = {
        "Trapezoid": {"tailored", "regular"},
        "Rectangle": {"regular", "relaxed"},
        "Inverted Triangle": {"regular", "relaxed"},
        "Triangle": {"tailored", "regular"},
        "Oval": {"regular", "tailored"},
    }
    if fit in preferred_fit.get(canonical, set()):
        reasons.append("fit_match")
        return (1.0, reasons)
    return (0.3, reasons)


def _formality_bonus(outfit: RankableOutfit, occasion: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    formality = str(outfit.formality or "").strip().casefold()
    if not formality:
        return (0.4, reasons)
    target = {
        "daily": {"casual", "smart_casual"},
        "work": {"business_casual", "formal"},
        "party": {"smart_casual", "formal"},
        "date": {"smart_casual", "business_casual"},
    }
    if formality in target.get(occasion, set()):
        reasons.append("formality_match")
        return (1.0, reasons)
    return (0.2, reasons)


def _rank_candidates(
    outfits: Sequence[RankableOutfit],
    raw_scores: np.ndarray,
    *,
    color_sub_season: str,
    body_shape: str,
    occasion: str,
    color_confidence: float | None = None,
    body_confidence: float | None = None,
    color_photo_quality_score: float | None = None,
    body_photo_quality_score: float | None = None,
) -> list[RankedCandidate]:
    ranked: list[RankedCandidate] = []
    color_signal = _confidence_signal(
        color_confidence,
        photo_quality_score=color_photo_quality_score,
    )
    body_signal = _confidence_signal(
        body_confidence,
        photo_quality_score=body_photo_quality_score,
    )

    for index, outfit in enumerate(outfits):
        retrieval_score = float(raw_scores[index])
        semantic_score = max(0.0, min(1.0, (retrieval_score + 1.0) / 2.0))
        season_score, season_reasons = _season_alignment_score(outfit, color_sub_season)
        shape_score, shape_reasons = _body_shape_alignment_score(outfit, body_shape)
        color_score, color_reasons = _color_family_bonus(outfit, color_sub_season)
        fit_score, fit_reasons = _fit_bonus(outfit, body_shape)
        formality_score, formality_reasons = _formality_bonus(outfit, occasion)

        final_score = (
            SEMANTIC_WEIGHT * semantic_score
            + SEASON_WEIGHT * season_score * (0.4 + 0.6 * color_signal)
            + BODY_SHAPE_WEIGHT * shape_score * (0.4 + 0.6 * body_signal)
            + COLOR_FAMILY_WEIGHT * color_score * (0.5 + 0.5 * color_signal)
            + FIT_WEIGHT * fit_score * (0.5 + 0.5 * body_signal)
            + FORMALITY_WEIGHT * formality_score
        )
        final_score = max(0.0, min(1.0, final_score))
        reason_tags = tuple(
            dict.fromkeys(
                season_reasons
                + shape_reasons
                + color_reasons
                + fit_reasons
                + formality_reasons
                + (["low_color_confidence"] if color_signal < 0.55 else [])
                + (["low_body_confidence"] if body_signal < 0.55 else [])
                + (
                    ["low_color_photo_quality"]
                    if (color_photo_quality_score or 1.0) < 0.58
                    else []
                )
                + (
                    ["low_body_photo_quality"]
                    if (body_photo_quality_score or 1.0) < 0.58
                    else []
                )
            )
        )
        ranked.append(
            RankedCandidate(
                outfit=outfit,
                final_score=final_score,
                retrieval_score=retrieval_score,
                reason_tags=reason_tags,
            )
        )

    ranked.sort(
        key=lambda candidate: (candidate.final_score, candidate.retrieval_score),
        reverse=True,
    )
    return ranked


def _select_diverse_top_k(
    candidates: list[RankedCandidate], top_k: int
) -> list[RankedCandidate]:
    selected: list[RankedCandidate] = []
    category_counts: Counter[str] = Counter()
    seen_images: set[str] = set()

    for candidate in candidates:
        category = candidate.outfit.category
        if candidate.outfit.image_url in seen_images:
            continue
        if category_counts[category] >= 2:
            continue
        selected.append(candidate)
        category_counts[category] += 1
        seen_images.add(candidate.outfit.image_url)
        if len(selected) >= top_k:
            return selected

    for candidate in candidates:
        if candidate in selected:
            continue
        if candidate.outfit.image_url in seen_images:
            continue
        selected.append(candidate)
        seen_images.add(candidate.outfit.image_url)
        if len(selected) >= top_k:
            return selected
    return selected
