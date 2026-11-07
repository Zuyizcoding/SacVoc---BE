"""Kiểu dữ liệu và metadata dùng chung của F3."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Retain the committed bundle's identity for baseline verification.
EMBEDDING_SCHEMA_VERSION = "2026-05-mens-v1"
# Correct projected-feature handling changes vector geometry, not templates.
PROJECTED_EMBEDDING_VERSION = "2026-11-projected-v2"

SUPPORTED_EMBEDDING_BACKENDS = {"fashion_clip", "hf_clip"}


class RecommendationNotReadyError(RuntimeError):
    """Raised when the recommendation stack is not safe to use."""


@dataclass(frozen=True)
class QueryVectorBundle:
    vector: np.ndarray
    source: str
    backend: str | None = None
    model: str | None = None
    version: str | None = None
    query_signature: str | None = None


class RankableOutfit(Protocol):
    """Fields read by ranking and response rendering, independent of storage.

    A Protocol describes an object's available attributes. Both a SQLAlchemy row
    and an evaluation snapshot satisfy this contract; neither must inherit it.
    Ranking can therefore reuse identical rules without opening a database.
    """

    id: int
    name: str
    category: str
    image_url: str
    shopee_url: str
    price_range: str
    season_bias: str | None
    body_shape_bias: str | None
    color_family: str | None
    fit: str | None
    formality: str | None


@dataclass(frozen=True)
class RankedCandidate:
    outfit: RankableOutfit
    final_score: float
    retrieval_score: float
    reason_tags: tuple[str, ...]
