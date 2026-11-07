"""Adapt FashionCLIP to Transformers without projecting features twice.

Intuition: CLIPModel.get_text_features/get_image_features already include the
learned projection. Transformers 4 returns a tensor; Transformers 5 returns an
object whose pooler_output contains that same projected tensor. The adapter
only unwraps the container. A 512-dimensional shape cannot tell us whether a
projection has already happened, so dimension-based projection is unsafe.

Pipeline: translate the old loader keyword -> load model -> unwrap feature
outputs -> let the existing caller normalize vectors. No weights are changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor

_PATCH_APPLIED = False


def apply_transformers_compat_patch() -> None:
    """Map legacy use_auth_token to token for both CLIP loaders."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    from transformers import CLIPModel, CLIPProcessor

    def wrap_loader(original):
        def compatible_loader(cls, pretrained_model_name_or_path, *args, **kwargs):
            # Prefer an explicit modern token if both spellings were supplied.
            legacy_token = kwargs.pop("use_auth_token", None)
            if legacy_token is not None and "token" not in kwargs:
                kwargs["token"] = legacy_token
            return original(cls, pretrained_model_name_or_path, *args, **kwargs)

        return classmethod(compatible_loader)

    for target in (CLIPModel, CLIPProcessor):
        target.from_pretrained = wrap_loader(target.from_pretrained.__func__)

    _PATCH_APPLIED = True


def _clip_features_to_tensor(raw: Any) -> Tensor:
    """Unwrap projected CLIP feature API output; never apply a projection."""
    features = raw if hasattr(raw, "detach") else getattr(raw, "pooler_output", None)
    if features is None or not all(
        hasattr(features, method) for method in ("detach", "cpu", "numpy")
    ):
        raise TypeError(f"Unsupported CLIP feature output type: {type(raw)}")
    return features


def patch_clip_model_feature_outputs(model: Any) -> None:
    """Let the legacy FashionCLIP wrapper receive tensors on either SDK."""
    if getattr(model, "_fashion_clip_compat_patched", False):
        return

    original_image = model.get_image_features
    original_text = model.get_text_features

    def get_image_features(*args, **kwargs):
        return _clip_features_to_tensor(original_image(*args, **kwargs))

    def get_text_features(*args, **kwargs):
        return _clip_features_to_tensor(original_text(*args, **kwargs))

    model.get_image_features = get_image_features
    model.get_text_features = get_text_features
    model._fashion_clip_compat_patched = True


def load_fashion_clip(model_name: str = "fashion-clip"):
    """Load FashionCLIP and adapt output containers, preserving feature values."""
    apply_transformers_compat_patch()
    from fashion_clip.fashion_clip import FashionCLIP

    model = FashionCLIP(model_name)
    patch_clip_model_feature_outputs(model.model)
    return model


def clip_features_to_numpy(raw: Any) -> np.ndarray:
    """Convert projected API features to CPU NumPy, detached from gradients."""
    return _clip_features_to_tensor(raw).detach().cpu().numpy()
