"""Regression coverage for SDK output containers without installing Torch."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.recommendation import embedding_store
from app.services.recommendation.types import (
    PROJECTED_EMBEDDING_VERSION,
    QueryVectorBundle,
    RecommendationNotReadyError,
)
from app.utils import fashion_clip_compat as compat


class FakeTensor:
    """Model a projected tensor; preserve its numbers through detach/cpu."""

    def __init__(self):
        self.values = np.arange(512, dtype=np.float32).reshape(1, 512)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def forbidden_projection(*args, **kwargs):
    raise AssertionError("Projected feature API results must not be projected again")


@pytest.mark.parametrize("wrapped", [False, True])
def test_feature_values_survive_both_sdk_output_contracts(wrapped):
    tensor = FakeTensor()
    raw = SimpleNamespace(pooler_output=tensor) if wrapped else tensor
    np.testing.assert_array_equal(compat.clip_features_to_numpy(raw), tensor.values)


@pytest.mark.parametrize("raw", [None, (), {}, SimpleNamespace(pooler_output=None)])
def test_unsupported_feature_containers_fail_explicitly(raw):
    with pytest.raises(TypeError):
        compat.clip_features_to_numpy(raw)


def test_fashion_clip_wrapper_never_invokes_projection_and_patches_once():
    text, image = FakeTensor(), FakeTensor()
    model = SimpleNamespace(
        get_text_features=lambda *a, **kw: SimpleNamespace(pooler_output=text),
        get_image_features=lambda *a, **kw: SimpleNamespace(pooler_output=image),
        text_projection=forbidden_projection,
        visual_projection=forbidden_projection,
    )
    compat.patch_clip_model_feature_outputs(model)
    patched_method = model.get_text_features
    compat.patch_clip_model_feature_outputs(model)
    assert model.get_text_features is patched_method
    assert model.get_text_features() is text
    assert model.get_image_features() is image


def test_loader_keyword_mapping_preserves_modern_token(monkeypatch):
    class Loader:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return name, kwargs

    class Model(Loader):
        pass

    class Processor(Loader):
        pass

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(CLIPModel=Model, CLIPProcessor=Processor),
    )
    monkeypatch.setattr(compat, "_PATCH_APPLIED", False)
    compat.apply_transformers_compat_patch()
    original_patch = Model.from_pretrained.__func__
    compat.apply_transformers_compat_patch()
    assert Model.from_pretrained.__func__ is original_patch
    assert Model.from_pretrained("local", use_auth_token="legacy")[1] == {
        "token": "legacy"
    }
    assert Processor.from_pretrained("local", use_auth_token="legacy", token="modern")[
        1
    ] == {"token": "modern"}


def test_runtime_hf_fallback_unwraps_projected_features(monkeypatch):
    tensor = FakeTensor()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    encoder = embedding_store.RuntimeFashionClipTextEncoder.__new__(
        embedding_store.RuntimeFashionClipTextEncoder
    )
    encoder._backend = "hf_clip"
    encoder._processor = lambda **kwargs: {}
    encoder._model = SimpleNamespace(
        get_text_features=lambda **kwargs: SimpleNamespace(pooler_output=tensor),
        text_projection=forbidden_projection,
    )
    np.testing.assert_array_equal(encoder.encode_text(["query"]), tensor.values)


def test_corrected_runtime_is_versioned_and_rejects_legacy_catalog(monkeypatch):
    monkeypatch.setattr(embedding_store, "load_fashion_clip", lambda name: object())
    encoder = embedding_store.RuntimeFashionClipTextEncoder("fashion-clip")
    assert encoder.version == PROJECTED_EMBEDDING_VERSION
    bundle = QueryVectorBundle(
        vector=np.zeros(512),
        source="runtime_fashion_clip",
        backend="fashion_clip",
        model="fashion-clip",
        version=encoder.version,
    )
    with pytest.raises(RecommendationNotReadyError):
        embedding_store._assert_query_vector_consistency(
            bundle,
            {
                "backend": "fashion_clip",
                "model": "fashion-clip",
                "version": "2026-05-mens-v1",
            },
        )
