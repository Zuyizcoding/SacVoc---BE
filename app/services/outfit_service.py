"""Recommendation service for the male-focused outfit flow."""

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Outfit
from app.schemas.outfit import OutfitItem, OutfitRecommendResponse

logger = logging.getLogger(__name__)

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


class RuntimeFashionClipTextEncoder:
    """Lazy wrapper for a real text encoder."""

    def __init__(self, model_name: str) -> None:
        self._backend = "unknown"
        self._model_name = model_name
        self._version = EMBEDDING_SCHEMA_VERSION
        self._model = None
        self._processor = None

        try:
            from fashion_clip.fashion_clip import FashionCLIP

            self._model = FashionCLIP(model_name)
            self._backend = "fashion_clip"
            return
        except Exception as exc:
            logger.info("FashionCLIP runtime unavailable, fallback to transformers CLIP: %s", str(exc))

        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._backend = "hf_clip"
        self._model_name = "openai/clip-vit-base-patch32"

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if self._backend == "fashion_clip":
            return self._model.encode_text(texts, batch_size=32)

        with __import__("torch").no_grad():
            inputs = self._processor(
                text=texts,
                images=None,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            raw = self._model.get_text_features(**inputs)
            if hasattr(raw, "cpu"):
                features = raw
            else:
                pooled = getattr(raw, "pooler_output", None)
                if pooled is None:
                    raise TypeError(f"Unsupported text feature output: {type(raw)}")
                projection = getattr(self._model, "text_projection", None)
                if projection is not None:
                    in_features = getattr(projection, "in_features", None)
                    if in_features is None or pooled.shape[-1] == in_features:
                        pooled = projection(pooled)
                features = pooled
        return features.cpu().numpy()

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
    return Path(__file__).resolve().parents[2]


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
            logger.warning("Skip invalid query vector for '%s': %s", query_text, str(exc))

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


SEASON_KEYWORDS: dict[str, str] = {
    "Light Spring": "warm light peachy coral pastel clean menswear soft contrast fresh",
    "True Spring": "warm golden vibrant green orange yellow energetic menswear crisp cheerful",
    "Bright Spring": "warm vivid bright saturated bold colorful menswear high energy contrast",
    "Light Summer": "cool soft pastel lavender baby blue rose powder airy menswear refined",
    "True Summer": "cool muted dusty rose berry mauve elegant menswear soft tonal tailoring",
    "Soft Summer": "cool muted neutral taupe sage soft blended menswear low contrast relaxed tailoring",
    "Soft Autumn": "warm muted olive khaki camel beige earthy soft menswear textured understated",
    "True Autumn": "warm rich earthy terracotta rust brown burnt orange menswear rugged classic",
    "Deep Autumn": "warm deep dark brown forest green burgundy menswear rich contrast mature",
    "Deep Winter": "cool deep dark navy black jewel tone dramatic menswear sharp contrast sleek",
    "True Winter": "cool bold contrast black white red icy pure menswear crisp monochrome clean",
    "Bright Winter": "cool vivid bright cobalt electric saturated menswear glossy high contrast statement",
}

BODY_KEYWORDS: dict[str, str] = {
    "Trapezoid": "menswear tailored fit balanced torso clean shoulder line structured blazer tapered chinos regular trousers",
    "Rectangle": "layered menswear overshirt bomber trucker jacket straight fit trousers texture and structure",
    "Inverted Triangle": "menswear V-neck raglan top straight leg wide leg trousers balanced lower body clean drape",
    "Triangle": "menswear structured shoulder jacket crew neck darker top lighter bottom straight trousers balanced proportion",
    "Oval": "menswear vertical line outfit open layer mid-rise straight leg clean minimal streamlined torso",
}

OCCASION_KEYWORDS: dict[str, str] = {
    "daily": "mens casual everyday comfortable relaxed streetwear clean sneakers overshirt",
    "work": "mens office professional business casual blazer chinos loafers polished neat",
    "party": "mens evening smart casual sharp clean dark tonal outfit refined texture",
    "date": "mens date night smart casual polished minimal style soft tailoring elevated basics",
}

SEASON_COLOR_FAMILY: dict[str, set[str]] = {
    "Spring": {"warm_bright", "earth", "rich"},
    "Summer": {"soft_cool", "cool", "neutral"},
    "Autumn": {"earth", "rich", "neutral"},
    "Winter": {"high_contrast", "cool", "rich"},
}

BODY_SHAPE_COMPATIBILITY: dict[str, set[str]] = {
    "Trapezoid": {"trapezoid", "rectangle"},
    "Rectangle": {"rectangle", "trapezoid"},
    "Inverted Triangle": {"inverted triangle", "trapezoid"},
    "Triangle": {"triangle", "oval"},
    "Oval": {"oval", "triangle"},
}


def _all_query_templates() -> list[str]:
    queries: list[str] = []
    for season in SEASON_KEYWORDS:
        for body_shape in BODY_KEYWORDS:
            for occasion in OCCASION_KEYWORDS:
                queries.append(build_outfit_query(season, body_shape, occasion))
    return queries


def _current_query_signature() -> str:
    payload = "\n".join(sorted(_all_query_templates())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _primary_season(sub_season: str) -> str:
    normalized = str(sub_season or "").strip().lower()
    if "spring" in normalized:
        return "Spring"
    if "summer" in normalized:
        return "Summer"
    if "autumn" in normalized:
        return "Autumn"
    if "winter" in normalized:
        return "Winter"
    return ""


def _canonicalize_body_shape_male(shape: str) -> str:
    raw = str(shape or "").strip().lower()
    aliases = {
        "trapezoid": "Trapezoid",
        "hourglass": "Trapezoid",
        "soft hourglass": "Trapezoid",
        "rectangle": "Rectangle",
        "inverted triangle": "Inverted Triangle",
        "triangle": "Triangle",
        "oval": "Oval",
    }
    return aliases.get(raw, str(shape).strip())


def _confidence_signal(
    confidence: float | None,
    *,
    photo_quality_score: float | None = None,
    fallback: float = 0.75,
) -> float:
    if confidence is None:
        signal = fallback
    else:
        signal = float(confidence)

    signal = max(0.35, min(signal, 1.0))
    if photo_quality_score is None:
        return signal

    quality_signal = max(0.25, min(float(photo_quality_score), 1.0))
    return max(0.3, min(0.7 * signal + 0.3 * quality_signal, 1.0))


def build_outfit_query(
    color_sub_season: str,
    body_shape: str,
    occasion: str = "daily",
) -> str:
    """Build a menswear-focused semantic search query."""
    season_kw = SEASON_KEYWORDS.get(color_sub_season, "refined menswear color palette")
    canonical_shape = _canonicalize_body_shape_male(body_shape)
    body_kw = BODY_KEYWORDS.get(canonical_shape, "menswear flattering well-fitted proportions")
    occasion_kw = OCCASION_KEYWORDS.get(occasion, "casual everyday menswear")
    return (
        f"A men's outfit with {season_kw}, "
        f"suited for a {canonical_shape} body using {body_kw}, "
        f"{occasion_kw}, ecommerce product photo, realistic styling, wearable proportions"
    )


def _parse_bias_tags(value: str | None) -> set[str]:
    if not value:
        return set()
    raw_parts = [
        part.strip()
        for part in str(value).replace(",", "|").replace("/", "|").split("|")
        if part.strip()
    ]
    return {part.casefold() for part in raw_parts}


def _load_outfit_vectors(
    db: Session,
    occasion: str | None = None,
    gender: str | None = None,
) -> tuple[list[Outfit], np.ndarray]:
    query = db.query(Outfit).filter(Outfit.vector_blob.isnot(None))

    if occasion:
        query = query.filter(Outfit.occasion == occasion)

    aliases = ("male", "m", "nam", "man", "men")
    if (gender or "").lower() == "male":
        query = query.filter(func.lower(Outfit.gender).in_(aliases))

    outfits = query.all()
    if not outfits:
        return ([], np.empty((0, 512), dtype=np.float32))

    unique_outfits: list[Outfit] = []
    seen_images: set[str] = set()
    for outfit in outfits:
        image_key = str(outfit.image_url or "").strip()
        if image_key and image_key in seen_images:
            continue
        if image_key:
            seen_images.add(image_key)
        unique_outfits.append(outfit)
    outfits = unique_outfits

    vectors: list[np.ndarray] = []
    for outfit in outfits:
        vec = np.frombuffer(outfit.vector_blob, dtype=np.float32)
        if vec.shape != (512,):
            raise RecommendationNotReadyError(
                f"Outfit ID {outfit.id} có embedding shape không hợp lệ: {vec.shape}"
            )
        vectors.append(vec)

    return (outfits, np.stack(vectors))


def _cosine_similarity_search(query_vector: np.ndarray, outfit_matrix: np.ndarray) -> np.ndarray:
    if outfit_matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    return outfit_matrix @ query_vector


def _load_outfit_embedding_meta(outfits: list[Outfit]) -> dict[str, str]:
    if not outfits:
        return {}

    meta_rows = {
        (
            str(outfit.embedding_backend or "").strip(),
            str(outfit.embedding_model or "").strip(),
            str(outfit.embedding_version or "").strip(),
        )
        for outfit in outfits
    }
    if any(not backend or not model or not version for backend, model, version in meta_rows):
        raise RecommendationNotReadyError(
            "Outfit vectors đang thiếu metadata encoder/model/version. "
            "Hãy encode lại outfit vectors thật trước khi dùng F3."
        )
    if len(meta_rows) != 1:
        raise RecommendationNotReadyError(
            "Catalog outfit đang trộn nhiều loại embedding backend/model khác nhau. "
            "Hãy re-encode toàn bộ catalog bằng một encoder thống nhất."
        )

    backend, model, version = next(iter(meta_rows))
    if backend not in SUPPORTED_EMBEDDING_BACKENDS:
        raise RecommendationNotReadyError(
            "Catalog outfit hiện không dùng embedding thật được hỗ trợ. "
            "Dummy vectors không được phép dùng cho recommendation accuracy."
        )
    return {"backend": backend, "model": model, "version": version}


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


def _assert_query_vector_consistency(query_bundle: QueryVectorBundle, outfit_meta: dict[str, str]) -> None:
    if query_bundle.source == "injected_encoder" and not query_bundle.backend:
        return

    if query_bundle.backend != outfit_meta["backend"] or query_bundle.model != outfit_meta["model"]:
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


def _season_alignment_score(outfit: Outfit, color_sub_season: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    bias = _parse_bias_tags(outfit.season_bias)
    if not bias:
        return (0.4, reasons)

    sub_season = color_sub_season.casefold()
    primary = _primary_season(color_sub_season).casefold()
    if sub_season and sub_season in bias:
        reasons.append("exact_season")
        return (1.0, reasons)
    if primary and primary in bias:
        reasons.append("primary_season")
        return (0.82, reasons)
    return (0.15, reasons)


def _body_shape_alignment_score(outfit: Outfit, body_shape: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    bias = _parse_bias_tags(outfit.body_shape_bias)
    if not bias:
        return (0.4, reasons)

    canonical = _canonicalize_body_shape_male(body_shape)
    key = canonical.casefold()
    if key in bias:
        reasons.append("exact_shape")
        return (1.0, reasons)
    compatible = BODY_SHAPE_COMPATIBILITY.get(canonical, set())
    if bias.intersection({item.casefold() for item in compatible}):
        reasons.append("compatible_shape")
        return (0.72, reasons)
    return (0.15, reasons)


def _color_family_bonus(outfit: Outfit, color_sub_season: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    family = str(outfit.color_family or "").strip().casefold()
    if not family:
        return (0.4, reasons)
    primary = _primary_season(color_sub_season)
    preferred = {item.casefold() for item in SEASON_COLOR_FAMILY.get(primary, set())}
    if family in preferred:
        reasons.append("color_family_match")
        return (1.0, reasons)
    return (0.25, reasons)


def _fit_bonus(outfit: Outfit, body_shape: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    fit = str(outfit.fit or "").strip().casefold()
    canonical = _canonicalize_body_shape_male(body_shape)
    if not fit:
        return (0.4, reasons)
    preferred_fit = {
        "Trapezoid": {"tailored", "regular"},
        "Rectangle": {"regular", "relaxed"},
        "Inverted Triangle": {"regular", "relaxed"},
        "Triangle": {"tailored", "regular"},
        "Oval": {"regular", "tailored"},
    }
    if fit in preferred_fit.get(canonical, set()):
        reasons.append("fit_match")
        return (1.0, reasons)
    return (0.3, reasons)


def _formality_bonus(outfit: Outfit, occasion: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    formality = str(outfit.formality or "").strip().casefold()
    if not formality:
        return (0.4, reasons)
    target = {
        "daily": {"casual", "smart_casual"},
        "work": {"business_casual", "formal"},
        "party": {"smart_casual", "formal"},
        "date": {"smart_casual", "business_casual"},
    }
    if formality in target.get(occasion, set()):
        reasons.append("formality_match")
        return (1.0, reasons)
    return (0.2, reasons)


def _rank_candidates(
    outfits: list[Outfit],
    raw_scores: np.ndarray,
    *,
    color_sub_season: str,
    body_shape: str,
    occasion: str,
    color_confidence: float | None = None,
    body_confidence: float | None = None,
    color_photo_quality_score: float | None = None,
    body_photo_quality_score: float | None = None,
) -> list[RankedCandidate]:
    ranked: list[RankedCandidate] = []
    color_signal = _confidence_signal(
        color_confidence,
        photo_quality_score=color_photo_quality_score,
    )
    body_signal = _confidence_signal(
        body_confidence,
        photo_quality_score=body_photo_quality_score,
    )

    for index, outfit in enumerate(outfits):
        retrieval_score = float(raw_scores[index])
        semantic_score = max(0.0, min(1.0, (retrieval_score + 1.0) / 2.0))
        season_score, season_reasons = _season_alignment_score(outfit, color_sub_season)
        shape_score, shape_reasons = _body_shape_alignment_score(outfit, body_shape)
        color_score, color_reasons = _color_family_bonus(outfit, color_sub_season)
        fit_score, fit_reasons = _fit_bonus(outfit, body_shape)
        formality_score, formality_reasons = _formality_bonus(outfit, occasion)

        final_score = (
            0.56 * semantic_score
            + 0.16 * season_score * (0.4 + 0.6 * color_signal)
            + 0.14 * shape_score * (0.4 + 0.6 * body_signal)
            + 0.06 * color_score * (0.5 + 0.5 * color_signal)
            + 0.04 * fit_score * (0.5 + 0.5 * body_signal)
            + 0.04 * formality_score
        )
        final_score = max(0.0, min(1.0, final_score))
        reason_tags = tuple(
            dict.fromkeys(
                season_reasons
                + shape_reasons
                + color_reasons
                + fit_reasons
                + formality_reasons
                + (["low_color_confidence"] if color_signal < 0.55 else [])
                + (["low_body_confidence"] if body_signal < 0.55 else [])
                + (["low_color_photo_quality"] if (color_photo_quality_score or 1.0) < 0.58 else [])
                + (["low_body_photo_quality"] if (body_photo_quality_score or 1.0) < 0.58 else [])
            )
        )
        ranked.append(
            RankedCandidate(
                outfit=outfit,
                final_score=final_score,
                retrieval_score=retrieval_score,
                reason_tags=reason_tags,
            )
        )

    ranked.sort(key=lambda candidate: (candidate.final_score, candidate.retrieval_score), reverse=True)
    return ranked


def _select_diverse_top_k(candidates: list[RankedCandidate], top_k: int) -> list[RankedCandidate]:
    selected: list[RankedCandidate] = []
    category_counts: Counter[str] = Counter()
    seen_images: set[str] = set()

    for candidate in candidates:
        category = candidate.outfit.category
        if candidate.outfit.image_url in seen_images:
            continue
        if category_counts[category] >= 2:
            continue
        selected.append(candidate)
        category_counts[category] += 1
        seen_images.add(candidate.outfit.image_url)
        if len(selected) >= top_k:
            return selected

    for candidate in candidates:
        if candidate in selected:
            continue
        if candidate.outfit.image_url in seen_images:
            continue
        selected.append(candidate)
        seen_images.add(candidate.outfit.image_url)
        if len(selected) >= top_k:
            return selected
    return selected


def recommend_outfits(
    db: Session,
    color_sub_season: str,
    body_shape: str,
    occasion: str = "daily",
    gender: str = "male",
    top_k: int = 6,
    text_encoder=None,
    color_confidence: float | None = None,
    body_confidence: float | None = None,
    color_photo_quality_score: float | None = None,
    body_photo_quality_score: float | None = None,
) -> OutfitRecommendResponse:
    query_text = build_outfit_query(color_sub_season, body_shape, occasion)
    query_bundle = _resolve_query_vector(query_text, text_encoder=text_encoder)

    outfits, outfit_matrix = _load_outfit_vectors(db, occasion=occasion, gender=gender)
    if len(outfits) == 0:
        raise RecommendationNotReadyError(
            "Dữ liệu outfit chưa sẵn sàng cho gợi ý. "
            "Hãy seed outfit và encode vector_blob thật trước khi dùng F3."
        )

    outfit_meta = _load_outfit_embedding_meta(outfits)
    _assert_query_vector_consistency(query_bundle, outfit_meta)

    raw_scores = _cosine_similarity_search(query_bundle.vector, outfit_matrix)
    ranked = _rank_candidates(
        outfits,
        raw_scores,
        color_sub_season=color_sub_season,
        body_shape=body_shape,
        occasion=occasion,
        color_confidence=color_confidence,
        body_confidence=body_confidence,
        color_photo_quality_score=color_photo_quality_score,
        body_photo_quality_score=body_photo_quality_score,
    )
    selected = _select_diverse_top_k(ranked, top_k)

    logger.info(
        "F3 recommend diagnostics=%s",
        json.dumps(
            {
                "query": query_text,
                "query_source": query_bundle.source,
                "query_backend": query_bundle.backend,
                "query_model": query_bundle.model,
                "color_confidence": color_confidence,
                "body_confidence": body_confidence,
                "color_photo_quality_score": color_photo_quality_score,
                "body_photo_quality_score": body_photo_quality_score,
                "outfit_backend": outfit_meta["backend"],
                "outfit_model": outfit_meta["model"],
                "candidate_count": len(ranked),
                "selected": [
                    {
                        "id": candidate.outfit.id,
                        "score": round(candidate.final_score, 4),
                        "retrieval_score": round(candidate.retrieval_score, 4),
                        "reasons": list(candidate.reason_tags),
                    }
                    for candidate in selected
                ],
            },
            ensure_ascii=False,
        ),
    )

    outfit_items = [
        OutfitItem(
            id=candidate.outfit.id,
            name=candidate.outfit.name,
            category=candidate.outfit.category,
            image_url=candidate.outfit.image_url,
            shopee_url=candidate.outfit.shopee_url,
            price_range=candidate.outfit.price_range,
            score=round(candidate.final_score, 4),
            retrieval_score=round(candidate.retrieval_score, 4),
            reason_tags=list(candidate.reason_tags) or None,
        )
        for candidate in selected
    ]

    return OutfitRecommendResponse(
        query_used=query_text,
        query_source=query_bundle.source,
        outfits=outfit_items,
    )
