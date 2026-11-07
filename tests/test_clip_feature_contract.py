"""Optional real SDK check with a tiny random CLIP: no model download or DB."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from app.utils.fashion_clip_compat import (  # noqa: E402
    clip_features_to_numpy,
    patch_clip_model_feature_outputs,
)


def test_sdk_features_match_exactly_one_explicit_projection_for_both_modalities():
    torch.manual_seed(7)
    config = transformers.CLIPConfig(
        projection_dim=8,
        text_config=dict(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=8,
            eos_token_id=2,
        ),
        vision_config=dict(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
        ),
    )
    model = transformers.CLIPModel(config).eval()
    text = torch.tensor([[1, 6, 2]])
    pixels = torch.zeros(1, 3, 8, 8)
    with torch.inference_mode():
        reference_text = model.text_projection(model.text_model(text).pooler_output)
        reference_image = model.visual_projection(
            model.vision_model(pixels).pooler_output
        )
        np.testing.assert_allclose(
            clip_features_to_numpy(model.get_text_features(text)),
            reference_text.numpy(),
            atol=1e-7,
            rtol=0,
        )
        np.testing.assert_allclose(
            clip_features_to_numpy(model.get_image_features(pixels)),
            reference_image.numpy(),
            atol=1e-7,
            rtol=0,
        )
        patch_clip_model_feature_outputs(model)
        torch.testing.assert_close(model.get_text_features(text), reference_text)
        torch.testing.assert_close(model.get_image_features(pixels), reference_image)
