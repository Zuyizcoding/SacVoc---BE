"""Numerically verify a shared text/image embedding space.

Intuition: equal dimensions and model names do not imply equal coordinates.
An image vector can only be compared with the saved query vectors if the same
encoder reproduces those queries. Test ALL 240 templates, not a hand-picked one.

Schema: each query has cosine similarity and maximum coordinate error.
Pipeline: validate committed bundle -> re-encode text -> normalize -> compare.
This module never rewrites queries. A failed report must stop image encoding.
"""

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from app.services.recommendation.query_builder import (
    _all_query_templates,
    _current_query_signature,
)
from app.services.recommendation.types import EMBEDDING_SCHEMA_VERSION

MIN_QUERY_COSINE = 0.99999
MAX_COORDINATE_ERROR = 0.0001
IDENTITY = {
    "backend": "fashion_clip",
    "model": "fashion-clip",
    "version": EMBEDDING_SCHEMA_VERSION,
}


class TextEncoder(Protocol):
    def encode_text(self, texts: list[str]) -> np.ndarray: ...


def normalized_vectors(values: np.ndarray, count: int) -> np.ndarray:
    """Use the legacy encoder's float32 L2 convention; reject invalid output."""
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.shape != (count, 512) or not np.isfinite(vectors).all():
        raise ValueError("Expected finite vectors with shape (count, 512)")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Zero embeddings are not valid")
    return vectors / norms


def read_reference_queries(
    vectors_path: Path, metadata_path: Path, *, identity: dict[str, str] | None = None
) -> dict:
    """Validate identity, signature, exact template coverage, and unit norms."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_identity = IDENTITY if identity is None else identity
    if any(metadata.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("Committed query metadata does not declare FashionCLIP")
    if (
        metadata.get("query_count") != 240
        or metadata.get("query_signature") != _current_query_signature()
    ):
        raise ValueError("Committed query count/signature does not match templates")
    queries = json.loads(vectors_path.read_text(encoding="utf-8"))
    templates = _all_query_templates()
    if len(templates) != 240 or set(queries) != set(templates):
        raise ValueError("Committed query keys must match all 240 templates")
    raw = np.asarray([queries[text] for text in templates], dtype=np.float32)
    normalized_vectors(raw, 240)
    if not np.allclose(np.linalg.norm(raw, axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Committed queries must already be unit vectors")
    return queries


def verify_compatibility(encoder: TextEncoder, reference: dict) -> dict:
    """Compare coordinates as well as direction, allowing only rounding noise."""
    queries = _all_query_templates()
    expected = normalized_vectors(np.array([reference[q] for q in queries]), 240)
    actual = normalized_vectors(encoder.encode_text(queries), 240)
    # Recompute lengths in float64 so a reported cosine never exceeds 1.
    expected64, actual64 = expected.astype("float64"), actual.astype("float64")
    cosine = np.clip(
        np.sum(expected64 * actual64, axis=1)
        / (np.linalg.norm(expected64, axis=1) * np.linalg.norm(actual64, axis=1)),
        -1.0,
        1.0,
    )
    errors = np.max(np.abs(expected - actual), axis=1)
    passed = (cosine >= MIN_QUERY_COSINE) & (errors <= MAX_COORDINATE_ERROR)
    return {
        "compatible": bool(passed.all()),
        "query_count": 240,
        "passed_queries": int(passed.sum()),
        "min_required_cosine": MIN_QUERY_COSINE,
        "max_allowed_coordinate_error": MAX_COORDINATE_ERROR,
        "min_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
        "max_coordinate_error": float(errors.max()),
        "queries": [
            {
                "query": query,
                "cosine": float(similarity),
                "max_coordinate_error": float(error),
                "passed": bool(ok),
            }
            for query, similarity, error, ok in zip(queries, cosine, errors, passed)
        ],
    }
