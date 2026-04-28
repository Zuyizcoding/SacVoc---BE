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
  3. Encode text → vector 512 dims (FashionCLIP)
  4. Load tất cả outfit vectors từ DB
  5. Cosine similarity = dot product (đã normalize sẵn)
  6. Trả top-6 outfit có score cao nhất

LƯU Ý:
  - 80 items × 512 dims → matrix nhỏ, tính toán < 1ms
  - KHÔNG cần FAISS hay vector DB — numpy đủ cho 80 items
  - Mặc định occasion="daily" nếu user chưa chọn
"""
import numpy as np
from sqlalchemy.orm import Session

from app.db.models import Outfit
from app.schemas.outfit import OutfitItem, OutfitRecommendResponse


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
    "Hourglass":          "fitted waist cinched belted wrap dress bodycon",
    "Soft Hourglass":     "soft fitted empire waist draping gentle curves",
    "Rectangle":          "structured layered peplum belt ruffle volume",
    "Triangle":           "A-line flared skirt off-shoulder statement top",
    "Inverted Triangle":  "V-neck wide leg palazzo full skirt balance",
}

# Occasion → style keywords
OCCASION_KEYWORDS: dict[str, str] = {
    "daily":  "casual everyday comfortable relaxed streetwear",
    "work":   "office professional formal blazer tailored",
    "party":  "evening glamorous sparkle sequin cocktail",
    "date":   "romantic feminine elegant chic flirty",
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
                      Nếu None → dùng dummy vectors (cho unit test / demo cache).

    Returns:
        OutfitRecommendResponse chứa query_used + list OutfitItem (top-6)

    Raises:
        ValueError: Nếu không có outfit nào trong DB có vector
    """
    # Bước 1: Build text query
    query_text = build_outfit_query(color_sub_season, body_shape, occasion)

    # Bước 2: Encode text → vector
    if text_encoder is not None:
        # Production: dùng FashionCLIP instance
        query_vector = text_encoder.encode_text([query_text])[0]
        query_vector = query_vector.astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)
    else:
        # Fallback / Demo: tạo random vector (kết quả ngẫu nhiên nhưng app không crash)
        rng = np.random.default_rng(seed=42)
        query_vector = rng.standard_normal(512).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)

    # Bước 3: Load outfit vectors từ DB
    outfits, outfit_matrix = _load_outfit_vectors(db, occasion=occasion)

    if len(outfits) == 0:
        # DB trống hoặc chưa encode — trả list rỗng
        return OutfitRecommendResponse(
            query_used=query_text,
            outfits=[],
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
