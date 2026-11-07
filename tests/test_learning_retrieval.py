"""Optional CPU tensor tests: run in the separate learning environment."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.learning.retrieval_tensors import (  # noqa: E402
    batched_cosine_scores,
    normalize_rows,
    stable_top_k,
)


def test_cosine_matches_hand_calculation_and_numpy_without_gradients():
    queries = torch.tensor([[3.0, 0.0], [0.0, 4.0]], requires_grad=True)
    candidates = torch.tensor([[2.0, 0.0], [0.0, 5.0], [-2.0, 0.0]])
    original = queries.detach().clone()
    scores = batched_cosine_scores(queries, candidates, batch_size=1)
    expected = np.array([[1, 0, -1], [0, 1, 0]], dtype=np.float32)
    np.testing.assert_allclose(scores.numpy(), expected)
    torch.testing.assert_close(
        scores, batched_cosine_scores(queries, candidates, batch_size=10)
    )
    torch.testing.assert_close(queries.detach(), original)
    assert not scores.requires_grad


def test_stable_top_k_preserves_tied_catalog_order():
    scores = torch.tensor([[0.8, 0.8, 0.1], [-1.0, 0.0, -1.0]])
    result = stable_top_k(scores, k=2)
    assert result.indices.tolist() == [[0, 1], [1, 0]]
    torch.testing.assert_close(result.scores, torch.tensor([[0.8, 0.8], [0.0, -1.0]]))
    assert stable_top_k(scores, k=20).indices.shape == (2, 3)


@pytest.mark.parametrize(
    "matrix",
    [
        torch.zeros((1, 2)),
        torch.empty((0, 2)),
        torch.ones(2),
        torch.tensor([[float("nan"), 1.0]]),
        torch.tensor([[float("inf"), 1.0]]),
        torch.tensor([[1, 2]]),
    ],
)
def test_invalid_embedding_matrices_are_rejected(matrix):
    with pytest.raises(ValueError):
        normalize_rows(matrix)


@pytest.mark.parametrize("failure", ["batch", "dimension", "dtype"])
def test_incompatible_batches_are_rejected(failure):
    queries = torch.ones((2, 3))
    candidates = torch.ones((4, 3))
    if failure == "dimension":
        candidates = torch.ones((4, 2))
    elif failure == "dtype":
        candidates = candidates.double()
    with pytest.raises(ValueError):
        batched_cosine_scores(
            queries, candidates, batch_size=0 if failure == "batch" else 2
        )


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_top_k_is_rejected(k):
    with pytest.raises(ValueError):
        stable_top_k(torch.ones((1, 2)), k=k)
