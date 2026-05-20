"""
Outfit Recommendation Service — Semantic Search bằng FashionCLIP.

Tuân thủ:
  - backend-python #1: nằm trong services/ — business logic
  - backend-python #3: Type Hints cho mọi tham số và return
  - Tính năng #15: Đọc sub_season + shape từ AnalysisHistory (F1+F2→F3)
  - Tính năng #30: build_outfit_query() + cosine similarity → top-6
  - Tính năng #31: Trả 6 outfit phù hợp nhất
  - SHOULD: Extended Query Template thêm color + silhouette keywords

FLOW CHÍNH:
  1. Đọc sub_season (F1) + shape (F2) từ DB
  2. build_outfit_query() → chuỗi text tiếng Anh
  3. Encode text → vector 512 dims (FashionCLIP hoặc query vector đã pre-encode)
  4. Load tất cả outfit vectors từ DB
  5. Cosine similarity = dot product (đã normalize sẵn)
  6. Trả top-6 outfit có score cao nhất

LƯU Ý:
  - 80 items × 512 dims → matrix nhỏ, tính toán < 1ms
  - KHÔNG cần FAISS hay vector DB — numpy đủ cho 80 items
  - Mặc định occasion="daily" nếu user chưa chọn
"""
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Outfit
from app.schemas.outfit import OutfitItem, OutfitRecommendResponse

logger = logging.getLogger(__name__)


class RecommendationNotReadyError(RuntimeError):
    """F3 chưa sẵn sàng vì thiếu embedding/query vector thật."""


class RuntimeFashionClipTextEncoder:
    """Lazy wrapper cho text encoder thật (FashionCLIP hoặc CLIP fallback)."""

    def __init__(self, model_name: str) -> None:
        self._backend = "unknown"
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

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if self._backend == "fashion_clip":
            return self._model.encode_text(texts, batch_size=32)

        # HuggingFace CLIP fallback
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


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_query_vectors_path() -> Path:
    configured = Path(settings.F3_QUERY_VECTORS_PATH)
    if configured.is_absolute():
        return configured
    return _project_root() / configured


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
) -> tuple[np.ndarray, str]:
    """Lấy query vector thật cho F3, không fallback sang vector giả."""
    if text_encoder is not None:
        encoded = text_encoder.encode_text([query_text])[0]
        return _normalize_vector(np.asarray(encoded, dtype=np.float32)), "injected_encoder"

    precomputed_vectors = _load_precomputed_query_vectors()
    if query_text in precomputed_vectors:
        return precomputed_vectors[query_text], "precomputed_query_vector"

    runtime_encoder = _get_runtime_text_encoder()
    if runtime_encoder is not None:
        encoded = runtime_encoder.encode_text([query_text])[0]
        source = f"runtime_{runtime_encoder.backend}"
        return _normalize_vector(np.asarray(encoded, dtype=np.float32)), source

    path = _resolve_query_vectors_path()
    raise RecommendationNotReadyError(
        "Chức năng gợi ý trang phục chưa sẵn sàng. "
        f"Cần encoder FashionCLIP thật hoặc file query vectors tại '{path}'."
    )


# ═══════════════════════════════════════════════
# QUERY BUILDER — Biến F1+F2 thành text tiếng Anh
# ═══════════════════════════════════════════════

# Season → adjective + color keywords cho FashionCLIP hiểu
# SHOULD Extended Query: thêm color keywords cụ thể
SEASON_KEYWORDS: dict[str, str] = {
    "Light Spring": "warm light peachy coral pastel bright floral",
    "True Spring":  "warm golden vibrant green orange yellow energetic",
    "Bright Spring": "warm vivid bright saturated bold colorful",
    "Light Summer": "cool soft pastel lavender baby blue rose powder",
    "True Summer":  "cool muted dusty rose berry mauve elegant",
    "Soft Summer":  "cool muted neutral taupe sage soft blended",
    "Soft Autumn":  "warm muted olive khaki camel beige earthy soft",
    "True Autumn":  "warm rich earthy terracotta rust brown burnt orange",
    "Deep Autumn":  "warm deep dark brown forest green burgundy",
    "Deep Winter":  "cool deep dark navy black jewel tone dramatic",
    "True Winter":  "cool bold contrast black white red icy pure",
    "Bright Winter": "cool vivid bright neon electric magenta cobalt",
}

# Body shape → silhouette keywords cho FashionCLIP hiểu
# SHOULD Extended Query: thêm silhouette-specific terms
BODY_KEYWORDS: dict[str, str] = {
    "Hourglass":          "tailored fitted waist cinched belted wrap structured",
    "Soft Hourglass":     "soft tailored draping relaxed fitted waist gentle curves",
    "Rectangle":          "structured layered overshirt jacket belt contrast volume",
    "Triangle":           "balance upper body structured shoulders jacket crew neck",
    "Inverted Triangle":  "V-neck straight leg wide leg balance lower body",
}

# Occasion → style keywords
OCCASION_KEYWORDS: dict[str, str] = {
    "daily":  "casual everyday comfortable relaxed streetwear",
    "work":   "office professional formal blazer tailored",
    "party":  "evening glamorous sparkle sequin cocktail",
    "date":   "romantic elegant chic smart casual",
}


def build_outfit_query(
    color_sub_season: str,
    body_shape: str,
    occasion: str = "daily",
) -> str:
    """Biến kết quả F1 + F2 thành chuỗi text query cho FashionCLIP.

    Tính năng #30 + SHOULD Extended Query Template.

    Chuỗi query kết hợp 3 yếu tố:
      1. Season color keywords (từ F1)
      2. Body silhouette keywords (từ F2)
      3. Occasion style keywords

    FashionCLIP sẽ encode chuỗi này thành vector → tính cosine similarity
    với image vectors trong DB → outfit gần nhất về semantic = phù hợp nhất.

    Args:
        color_sub_season: VD "Soft Autumn" — từ AnalysisHistory.color_sub_season
        body_shape: VD "Hourglass" — từ AnalysisHistory.body_shape
        occasion: "daily" | "work" | "party" | "date" — mặc định "daily"

    Returns:
        Chuỗi text tiếng Anh tối ưu cho FashionCLIP semantic search.

    Example:
        >>> build_outfit_query("Soft Autumn", "Hourglass", "daily")
        'A warm muted olive khaki camel beige earthy soft outfit for
         fitted waist cinched belted wrap dress bodycon body type,
         casual everyday comfortable relaxed streetwear style'
    """
    # Lookup keywords — fallback nếu không tìm thấy
    season_kw = SEASON_KEYWORDS.get(color_sub_season, "fashionable stylish")
    body_kw = BODY_KEYWORDS.get(body_shape, "flattering well-fitted")
    occasion_kw = OCCASION_KEYWORDS.get(occasion, "casual everyday")

    # Template query — format tối ưu cho FashionCLIP
    query = (
        f"A {season_kw} outfit for "
        f"{body_kw} body type, "
        f"{occasion_kw} style"
    )

    return query


# ═══════════════════════════════════════════════
# COSINE SIMILARITY — numpy thuần, không cần FAISS
# ═══════════════════════════════════════════════

def _load_outfit_vectors(
    db: Session,
    occasion: str | None = None,
    gender: str | None = None,
) -> tuple[list[Outfit], np.ndarray]:
    """Load tất cả outfit có vector từ DB → numpy matrix.

    Args:
        db: SQLAlchemy Session
        occasion: Nếu có → filter theo occasion (COULD feature).
                  Nếu None → load tất cả (MUST — default behavior).

    Returns:
        (outfits, matrix):
          - outfits: list Outfit objects (để lấy metadata)
          - matrix: shape (N, 512) — N outfit × 512 dims, đã normalize
    """
    query = db.query(Outfit).filter(Outfit.vector_blob.isnot(None))

    if occasion:
        query = query.filter(Outfit.occasion == occasion)

    if gender and gender != "any":
        query = query.filter(Outfit.gender == gender)

    outfits = query.all()

    if not outfits:
        return ([], np.empty((0, 512), dtype=np.float32))

    # Deserialize BLOB → numpy vectors
    vectors = []
    for outfit in outfits:
        vec = np.frombuffer(outfit.vector_blob, dtype=np.float32)
        vectors.append(vec)

    matrix = np.stack(vectors)  # shape: (N, 512)
    return (outfits, matrix)


def _cosine_similarity_search(
    query_vector: np.ndarray,
    outfit_matrix: np.ndarray,
    top_k: int = 6,
) -> list[tuple[int, float]]:
    """Tính cosine similarity giữa query vector và tất cả outfit vectors.

    Vì cả 2 đều đã normalize (L2 norm = 1):
      cosine_similarity = dot_product (nhanh hơn)

    Args:
        query_vector: shape (512,) — text query embedding, đã normalize
        outfit_matrix: shape (N, 512) — outfit image embeddings, đã normalize
        top_k: số lượng outfit trả về (mặc định 6 — Tính năng #31)

    Returns:
        list[(index, score)] — top-K outfit indices + similarity scores,
        sorted giảm dần theo score.
    """
    if outfit_matrix.shape[0] == 0:
        return []

    # Dot product = cosine similarity (vì đã normalize)
    scores = outfit_matrix @ query_vector  # shape: (N,)

    # Lấy top-K indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(int(idx), float(scores[idx])) for idx in top_indices]


# ═══════════════════════════════════════════════
# MAIN FUNCTION — Gọi từ Router
# ═══════════════════════════════════════════════

def recommend_outfits(
    db: Session,
    color_sub_season: str,
    body_shape: str,
    occasion: str = "daily",
    gender: str = "any",
    top_k: int = 6,
    text_encoder=None,
) -> OutfitRecommendResponse:
    """Gợi ý top-K outfit phù hợp dựa trên F1 + F2 + occasion.

    Tính năng #30 + #31: Semantic Search pipeline hoàn chỉnh.

    Flow:
      1. build_outfit_query() → text tiếng Anh
      2. Encode text → vector 512D (FashionCLIP)
      3. Load outfit vectors từ DB
      4. Cosine similarity → top-K
      5. Map index → OutfitItem schema

    Args:
        db: SQLAlchemy Session (từ Depends(get_db))
        color_sub_season: Từ AnalysisHistory — VD "Soft Autumn"
        body_shape: Từ AnalysisHistory — VD "Hourglass"
        occasion: "daily" | "work" | "party" | "date" (mặc định "daily")
        top_k: Số outfit trả về (mặc định 6)
        text_encoder: FashionCLIP instance hoặc hàm encode text → vector.
                      Nếu None → thử query vectors pre-encode, rồi mới thử runtime encoder.

    Returns:
        OutfitRecommendResponse chứa query_used + list OutfitItem (top-6)

    Raises:
        ValueError: Nếu không có outfit nào trong DB có vector
    """
    # Bước 1: Build text query
    query_text = build_outfit_query(color_sub_season, body_shape, occasion)

    # Bước 2: Encode text → vector
    query_vector, query_source = _resolve_query_vector(
        query_text,
        text_encoder=text_encoder,
    )
    logger.info("F3 query vector source=%s | query=%s", query_source, query_text)

    # Bước 3: Load outfit vectors từ DB
    outfits, outfit_matrix = _load_outfit_vectors(db, occasion=occasion, gender=gender)

    if len(outfits) == 0:
        raise RecommendationNotReadyError(
            "Dữ liệu outfit chưa sẵn sàng cho gợi ý. "
            "Hãy seed outfit và encode vector_blob thật trước khi dùng F3."
        )

    # Bước 4: Cosine similarity
    results = _cosine_similarity_search(query_vector, outfit_matrix, top_k)

    # Bước 5: Map → OutfitItem schema
    outfit_items: list[OutfitItem] = []
    for idx, score in results:
        outfit = outfits[idx]
        outfit_items.append(
            OutfitItem(
                id=outfit.id,
                name=outfit.name,
                category=outfit.category,
                image_url=outfit.image_url,
                shopee_url=outfit.shopee_url,
                price_range=outfit.price_range,
                score=round(score, 4),
            )
        )

    return OutfitRecommendResponse(
        query_used=query_text,
        outfits=outfit_items,
    )
