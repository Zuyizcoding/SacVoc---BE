"""Kiểu dữ liệu và metadata dùng chung của F3."""

from dataclasses import dataclass

import numpy as np

from app.db.models import Outfit

EMBEDDING_SCHEMA_VERSION = "2026-05-mens-v1"

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


@dataclass(frozen=True)
class RankedCandidate:
    outfit: Outfit
    final_score: float
    retrieval_score: float
    reason_tags: tuple[str, ...]
