"""
Pre-encode text queries cho F3 Semantic Search (Cách A — optional).

Nếu server Railway không đủ RAM load FashionCLIP:
  1. Chạy script này trên máy dev
  2. Xuất ra file query_vectors.json
  3. Server đọc file này thay vì encode realtime

Cách chạy:
    cd src/backend
    python -m scripts.encode_text_queries
"""
import json
import numpy as np

# ═══════════════════════════════════════════════
# DANH SÁCH TEMPLATE QUERIES
# ═══════════════════════════════════════════════
# Mỗi query = 1 tổ hợp (season × body_shape × occasion)
# Format: "A {season_adjective} outfit for {body_shape} body, {occasion} style"

QUERY_TEMPLATES: list[str] = [
    # Spring × daily
    "A warm peachy spring outfit for hourglass body, casual daily style",
    "A warm peachy spring outfit for rectangle body, casual daily style",
    "A warm peachy spring outfit for triangle body, casual daily style",
    # Summer × daily
    "A cool pastel summer outfit for hourglass body, casual daily style",
    "A cool pastel summer outfit for rectangle body, casual daily style",
    # Autumn × daily
    "A warm earthy autumn outfit for hourglass body, casual daily style",
    "A warm earthy autumn outfit for rectangle body, casual daily style",
    # Winter × daily
    "A cool bold winter outfit for hourglass body, casual daily style",
    "A cool bold winter outfit for rectangle body, casual daily style",
    # ... Thêm cho work/party/date × các dáng khác ...
]


def encode_queries() -> None:
    """Encode tất cả template queries → lưu file JSON."""
    try:
        from fashion_clip.fashion_clip import FashionCLIP
    except ImportError:
        print("❌ pip install fashion-clip")
        return

    print("🔄 Loading FashionCLIP...")
    fclip = FashionCLIP("fashion-clip")

    print(f"📝 Encoding {len(QUERY_TEMPLATES)} queries...")
    text_embeddings = fclip.encode_text(QUERY_TEMPLATES, batch_size=32)

    # Normalize
    norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    text_embeddings = text_embeddings / norms

    # Lưu file
    output = {}
    for query, vector in zip(QUERY_TEMPLATES, text_embeddings):
        output[query] = vector.astype(np.float32).tolist()

    output_path = "data/query_vectors.json"
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)

    print(f"✅ Saved {len(output)} vectors → {output_path}")


if __name__ == "__main__":
    import os
    encode_queries()
