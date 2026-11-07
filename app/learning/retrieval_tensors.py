"""CPU tensor retrieval, explained through explicit shapes.

A query matrix has shape [query_count, embedding_size]; a candidate matrix has
[candidate_count, embedding_size]. Normalizing each row makes its length one.
Their matrix product Q @ C.T then contains one cosine similarity per pair.

These functions teach inference math, not model training. They return the full
score matrix even when query multiplication is batched; batching here is not a
streaming index or a solution for an arbitrarily large catalog.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as functional


@dataclass(frozen=True)
class TopKResult:
    scores: torch.Tensor
    indices: torch.Tensor


def _validate_matrix(matrix: torch.Tensor) -> None:
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Expected a non-empty [rows, embedding_size] matrix")
    if matrix.device.type != "cpu" or not matrix.is_floating_point():
        raise ValueError("Learning examples require floating-point CPU tensors")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("Tensor values must be finite")


@torch.inference_mode()
def normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    """Normalize embeddings along their feature dimension, without mutating input."""
    _validate_matrix(matrix)
    lengths = torch.linalg.vector_norm(matrix, dim=1)
    if bool((lengths <= 1e-12).any()):
        raise ValueError("Zero or near-zero vectors do not define a cosine direction")
    return functional.normalize(matrix, p=2, dim=1, eps=1e-12)


@torch.inference_mode()
def batched_cosine_scores(
    queries: torch.Tensor, candidates: torch.Tensor, *, batch_size: int = 32
) -> torch.Tensor:
    """Batch query rows; candidate order and score-matrix shape stay unchanged."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    normalized_queries = normalize_rows(queries)
    normalized_candidates = normalize_rows(candidates)
    if queries.shape[1] != candidates.shape[1] or queries.dtype != candidates.dtype:
        raise ValueError(
            "Queries and candidates need the same embedding size and dtype"
        )
    score_batches = []
    for start in range(0, len(queries), batch_size):
        query_batch = normalized_queries[start : start + batch_size]
        score_batches.append(query_batch @ normalized_candidates.T)
    return torch.cat(score_batches, dim=0)


@torch.inference_mode()
def stable_top_k(scores: torch.Tensor, *, k: int) -> TopKResult:
    """Return at most k candidates; equal scores retain original candidate order."""
    _validate_matrix(scores)
    if k <= 0:
        raise ValueError("k must be positive")
    # Stable sorting deliberately defines tie behavior; torch.topk does not
    # guarantee stable indices for tied values.
    indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    values = torch.gather(scores, dim=1, index=indices)
    return TopKResult(scores=values, indices=indices)
