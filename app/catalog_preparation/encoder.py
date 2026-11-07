"""Pinned, CPU-only FashionCLIP adapter, with no alternate-model fallback.

The upstream FashionCLIP wrapper maps 'fashion-clip' to this Hugging Face repo.
We use its CLIPModel/CLIPProcessor directly, avoiding its unrelated dataset and
nearest-neighbor dependencies. Text uses the upstream fixed 77-token padding;
images use the same processor and weights. Both return raw projected features,
which the caller normalizes to match the existing offline scripts.

The model directory must contain the official snapshot at MODEL_REVISION.
Its weight checksum is checked before loading safetensors. Record every local
processor hash too: numerical query verification remains mandatory even when
the weight checksum matches. No model code is downloaded or executed.
"""

import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from app.utils.fashion_clip_compat import clip_features_to_numpy

MODEL_REPOSITORY = "patrickjohncyh/fashion-clip"
MODEL_REVISION = "7e3ba62ce16b379a1ab479346b66f192e76f51b7"
WEIGHTS_SHA256 = "4977e3a54929eccf065ce449aeaf296f0e5cb6b28e8798c3c97d67cb2f6dafc9"
PROCESSOR_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "merges.txt",
    "vocab.json",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class PinnedFashionClip:
    def __init__(self, directory: Path) -> None:
        # Imports remain local: ordinary backend tests never need Torch.
        import torch
        from transformers import CLIPModel, CLIPProcessor

        weights_hash = file_sha256(directory / "model.safetensors")
        if weights_hash != WEIGHTS_SHA256:
            raise ValueError("FashionCLIP weights do not match the pinned checksum")
        self.provenance = {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "weights_sha256": weights_hash,
            "processor_sha256": {
                name: file_sha256(directory / name) for name in PROCESSOR_FILES
            },
            "packages": {
                name: version(name)
                for name in (
                    "torch",
                    "transformers",
                    "tokenizers",
                    "safetensors",
                    "Pillow",
                    "numpy",
                )
            },
            "device": "cpu",
            "attention_implementation": "eager",
            "text_padding": "max_length=77",
            "projection_contract": "one learned projection per modality",
        }
        torch.set_num_threads(4)
        self.model = (
            CLIPModel.from_pretrained(
                directory,
                local_files_only=True,
                use_safetensors=True,
                attn_implementation="eager",
            )
            .to("cpu")
            .eval()
        )
        self.processor = CLIPProcessor.from_pretrained(
            directory, local_files_only=True, use_fast=False
        )

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        features = []
        with torch.inference_mode():
            for start in range(0, len(texts), 32):
                inputs = self.processor(
                    text=texts[start : start + 32],
                    max_length=77,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                features.append(
                    clip_features_to_numpy(self.model.get_text_features(**inputs))
                )
        return np.concatenate(features)

    def encode_reference_text(self, texts: list[str]) -> np.ndarray:
        """Independently check the feature API against exactly one projection.

        This path bypasses get_text_features and our output adapter. A shared
        double-projection bug therefore cannot make both checks pass silently.
        """
        import torch

        features = []
        with torch.inference_mode():
            for start in range(0, len(texts), 32):
                inputs = self.processor(
                    text=texts[start : start + 32],
                    max_length=77,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                hidden = self.model.text_model(**inputs, return_dict=True).pooler_output
                features.append(self.model.text_projection(hidden).cpu().numpy())
        return np.concatenate(features)

    def encode_image(self, image_path: Path) -> np.ndarray:
        import torch

        with Image.open(image_path) as image:
            inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        with torch.inference_mode():
            return clip_features_to_numpy(self.model.get_image_features(**inputs))

    def encode_reference_image(self, image_path: Path) -> np.ndarray:
        """Check image features through vision_model -> visual_projection once."""
        import torch

        with Image.open(image_path) as image:
            inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        with torch.inference_mode():
            hidden = self.model.vision_model(**inputs, return_dict=True).pooler_output
            return self.model.visual_projection(hidden).cpu().numpy()
