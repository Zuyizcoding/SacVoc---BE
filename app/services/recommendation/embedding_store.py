"""Ưu tiên vector đã encode, chỉ khởi tạo encoder khi cần."""

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.utils.fashion_clip_compat import clip_features_to_numpy, load_fashion_clip

from .query_builder import _current_query_signature
from .types import (
    PROJECTED_EMBEDDING_VERSION,
    QueryVectorBundle,
    RecommendationNotReadyError,
)

logger = logging.getLogger(__name__)


class RuntimeFashionClipTextEncoder:
    """Lazy wrapper for a real text encoder."""

    def __init__(self, model_name: str) -> None:
        self._backend = "unknown"
        self._model_name = model_name
        self._version = PROJECTED_EMBEDDING_VERSION
        self._model = None
        self._processor = None

        try:
            self._model = load_fashion_clip(model_name)
            self._backend = "fashion_clip"
            return
        except Exception as exc:
            logger.info(
                "FashionCLIP runtime unavailable, fallback to transformers CLIP: %s",
                str(exc),
            )

        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._backend = "hf_clip"
        self._model_name = "openai/clip-vit-base-patch32"

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if self._backend == "fashion_clip":
            return self._model.encode_text(texts, batch_size=32)

        import torch

        with torch.no_grad():
            inputs = self._processor(
                text=texts,
                images=None,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            raw = self._model.get_text_features(**inputs)
        return clip_features_to_numpy(raw)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        return self._version


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_query_vectors_path() -> Path:
    configured = Path(settings.F3_QUERY_VECTORS_PATH)
    if configured.is_absolute():
        return configured
    return _project_root() / configured


def _resolve_query_meta_path() -> Path:
    return _resolve_query_vectors_path().with_name("query_vectors.meta.json")


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vec = vector.astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Embedding vector has zero norm")
    return vec / norm


@lru_cache(maxsize=1)
def _load_precomputed_query_vectors() -> dict[str, np.ndarray]:
    path = _resolve_query_vectors_path()
    if not path.exists():
        logger.info("F3 query vector file not found: %s", path)
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read query vector file %s: %s", path, str(exc))
        return {}

    vectors: dict[str, np.ndarray] = {}
    for query_text, vector_list in raw.items():
        try:
            vector = np.asarray(vector_list, dtype=np.float32)
            if vector.shape != (512,):
                raise ValueError(f"unexpected shape {vector.shape}")
            vectors[str(query_text)] = _normalize_vector(vector)
        except Exception as exc:
            logger.warning(
                "Skip invalid query vector for '%s': %s", query_text, str(exc)
            )

    logger.info("Loaded %d precomputed F3 query vectors from %s", len(vectors), path)
    return vectors


@lru_cache(maxsize=1)
def _load_query_vector_manifest() -> dict[str, str]:
    path = _resolve_query_meta_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read query vector metadata %s: %s", path, str(exc))
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if value is not None}


@lru_cache(maxsize=1)
def _get_runtime_text_encoder() -> RuntimeFashionClipTextEncoder | None:
    try:
        return RuntimeFashionClipTextEncoder(settings.F3_RUNTIME_TEXT_ENCODER_MODEL)
    except ImportError:
        logger.info("Runtime FashionCLIP is unavailable in current environment")
        return None
    except Exception as exc:
        logger.warning("Failed to initialize runtime FashionCLIP encoder: %s", str(exc))
        return None


def _resolve_query_vector(
    query_text: str,
    text_encoder=None,
) -> QueryVectorBundle:
    if text_encoder is not None:
        encoded = text_encoder.encode_text([query_text])[0]
        return QueryVectorBundle(
            vector=_normalize_vector(np.asarray(encoded, dtype=np.float32)),
            source="injected_encoder",
            backend=getattr(text_encoder, "backend", None),
            model=getattr(text_encoder, "model_name", None),
            version=getattr(text_encoder, "version", None),
            query_signature=_current_query_signature(),
        )

    precomputed_vectors = _load_precomputed_query_vectors()
    if query_text in precomputed_vectors:
        manifest = _load_query_vector_manifest()
        return QueryVectorBundle(
            vector=precomputed_vectors[query_text],
            source="precomputed_query_vector",
            backend=manifest.get("backend"),
            model=manifest.get("model"),
            version=manifest.get("version"),
            query_signature=manifest.get("query_signature"),
        )

    runtime_encoder = _get_runtime_text_encoder()
    if runtime_encoder is not None:
        encoded = runtime_encoder.encode_text([query_text])[0]
        return QueryVectorBundle(
            vector=_normalize_vector(np.asarray(encoded, dtype=np.float32)),
            source=f"runtime_{runtime_encoder.backend}",
            backend=runtime_encoder.backend,
            model=runtime_encoder.model_name,
            version=runtime_encoder.version,
            query_signature=_current_query_signature(),
        )

    path = _resolve_query_vectors_path()
    raise RecommendationNotReadyError(
        "Chức năng gợi ý trang phục chưa sẵn sàng. "
        f"Cần encoder FashionCLIP thật hoặc file query vectors tại '{path}'."
    )


def _assert_query_vector_consistency(
    query_bundle: QueryVectorBundle, outfit_meta: dict[str, str]
) -> None:
    if query_bundle.source == "injected_encoder" and not query_bundle.backend:
        return

    if (
        query_bundle.backend != outfit_meta["backend"]
        or query_bundle.model != outfit_meta["model"]
    ):
        raise RecommendationNotReadyError(
            "Query vectors và outfit vectors không cùng backend/model. "
            "Hãy encode lại cả query lẫn catalog bằng cùng embedding stack."
        )

    if query_bundle.version and query_bundle.version != outfit_meta["version"]:
        raise RecommendationNotReadyError(
            "Query vectors và outfit vectors khác embedding version/schema. "
            "Hãy rebuild đồng bộ embeddings."
        )

    if query_bundle.source == "precomputed_query_vector":
        manifest = _load_query_vector_manifest()
        if not manifest:
            raise RecommendationNotReadyError(
                "Thiếu query vector metadata. Hãy chạy lại script pre-encode query vectors."
            )
        if manifest.get("query_signature") != _current_query_signature():
            raise RecommendationNotReadyError(
                "Query vectors hiện không khớp với query template mới. "
                "Hãy chạy lại script encode_text_queries."
            )
