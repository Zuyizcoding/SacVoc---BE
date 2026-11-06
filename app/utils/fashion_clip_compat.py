"""Compatibility helpers for fashion-clip + transformers v5+.

fashion-clip 0.2.2 expects older transformers APIs:
- CLIPModel.from_pretrained(..., use_auth_token=...) -> use `token` instead
- get_image_features / get_text_features return tensors with .detach()
  but transformers v5 returns BaseModelOutputWithPooling objects
"""

from __future__ import annotations

import numpy as np

_PATCH_APPLIED = False


def apply_transformers_compat_patch() -> None:
    """Map legacy `use_auth_token` to `token` for CLIP loaders."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    from transformers import CLIPModel, CLIPProcessor

    def _wrap_from_pretrained(cls, original):
        def compat(pretrained_model_name_or_path, *args, **kwargs):
            if "use_auth_token" in kwargs and "token" not in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return original(pretrained_model_name_or_path, *args, **kwargs)

        compat.__name__ = getattr(original, "__name__", "from_pretrained")
        return classmethod(compat)

    for target in (CLIPModel, CLIPProcessor):
        original = target.from_pretrained.__func__
        target.from_pretrained = _wrap_from_pretrained(target, original)

    _PATCH_APPLIED = True


def _clip_features_to_tensor(raw, projection_layer) -> "object":
    import torch

    if isinstance(raw, torch.Tensor):
        return raw
    if hasattr(raw, "detach"):
        return raw

    pooled = getattr(raw, "pooler_output", None)
    if pooled is None:
        raise TypeError(f"Unsupported CLIP feature output type: {type(raw)}")

    if projection_layer is not None:
        in_features = getattr(projection_layer, "in_features", None)
        if in_features is None or pooled.shape[-1] == in_features:
            pooled = projection_layer(pooled)

    return pooled


def patch_clip_model_feature_outputs(model) -> None:
    """Ensure fashion-clip encode_* paths receive plain tensors."""
    if getattr(model, "_fashion_clip_compat_patched", False):
        return

    original_image = model.get_image_features
    original_text = model.get_text_features

    def get_image_features(*args, **kwargs):
        raw = original_image(*args, **kwargs)
        return _clip_features_to_tensor(raw, getattr(model, "visual_projection", None))

    def get_text_features(*args, **kwargs):
        raw = original_text(*args, **kwargs)
        return _clip_features_to_tensor(raw, getattr(model, "text_projection", None))

    model.get_image_features = get_image_features
    model.get_text_features = get_text_features
    model._fashion_clip_compat_patched = True


def load_fashion_clip(model_name: str = "fashion-clip"):
    """Load FashionCLIP with transformers v5 compatibility patches applied."""
    apply_transformers_compat_patch()
    from fashion_clip.fashion_clip import FashionCLIP

    fclip = FashionCLIP(model_name)
    patch_clip_model_feature_outputs(fclip.model)
    return fclip


def clip_features_to_numpy(raw, projection_layer=None) -> np.ndarray:
    if hasattr(raw, "cpu"):
        return raw.cpu().numpy()

    pooled = getattr(raw, "pooler_output", None)
    if pooled is None:
        raise TypeError(f"Unsupported feature output type: {type(raw)}")

    if projection_layer is not None:
        in_features = getattr(projection_layer, "in_features", None)
        if in_features is None or pooled.shape[-1] == in_features:
            pooled = projection_layer(pooled)

    return pooled.cpu().numpy()
