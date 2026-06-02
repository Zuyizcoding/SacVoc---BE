# Sắc Vóc — Backend API

API backend cho ứng dụng mobile **Sắc Vóc**: phân tích màu sắc cá nhân (F1), dáng người (F2) và gợi ý trang phục (F3).

| Thành phần | Công nghệ |
|------------|-----------|
| Framework | FastAPI + Uvicorn |
| AI (F1, F2) | Google Gemini (`gemini-2.5-flash`) |
| Gợi ý outfit (F3) | Rule-based query + vector similarity (FashionCLIP embeddings) |
| Database | SQLite + SQLAlchemy |
| Python | **3.12+** |

Repo frontend: [`CS4153.Q24.CTTT---FE`](../CS4153.Q24.CTTT---FE) (Flutter).

---

## Tính năng chính

### F1 — Phân tích màu (`POST /api/v1/analyze/color`)

- Upload ảnh selfie → Gemini suy luận mùa sắc (12 sub-season).
- Chuẩn hóa palette / gợi ý từ `app/data/season_data.py`.
- Lưu kết quả theo `X-Device-ID` (SHA256), **không lưu ảnh người dùng**.

### F2 — Phân tích dáng (`POST /api/v1/analyze/body`)

- Upload ảnh toàn thân → Gemini xác định dáng người + gợi ý trang phục.
- Kiểm tra chất lượng ảnh (kích thước, khoảng cách, đồ ôm sát).

### F3 — Gợi ý outfit (`GET /api/v1/outfits/recommend`)

- **Không gọi Gemini** lúc recommend.
- Kết hợp F1 + F2 từ DB → build text query → cosine similarity với embedding catalog.
- Trả tối đa **6 outfit**; filter `occasion`: `daily` | `work` | `party` | `date`.
- Hiện tại catalog demo: **nam** (`gender=male`).

### Khác

- `GET /api/v1/analyze/history` — xem lại F1/F2 đã lưu.
- `GET /api/v1/outfits/{id}` — chi tiết outfit (ảnh, giá, link Shopee).
- `GET /health`, `GET /health/ready` — health check / readiness F3.

---

## Cấu trúc thư mục

```text
app/
  api/endpoints/     # analyze.py, outfit.py
  core/              # config, prompts
  data/              # season_data.py
  db/                # models, crud, database
  services/          # gemini_client, outfit_service, demo_cache
  assets/demo_cache/ # JSON fallback khi AI quá tải
data/
  query_vectors.json # Vector query F3 (pre-encode)
scripts/             # seed, encode vectors (offline)
tests/
outfitseed.csv       # Nguồn seed catalog
.env.example         # Mẫu biến môi trường
```

---

## Cài đặt & chạy

### 1. Môi trường

```bash
cd CS4153.Q24.CTTT---BE
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest, FashionCLIP (cho script encode)
```

### 2. Biến môi trường

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Điền `GEMINI_API_KEY` tại [Google AI Studio](https://aistudio.google.com/apikey).

> File `.env` và `*.db` **không được commit** (đã có trong `.gitignore`).

### 3. Khởi tạo catalog F3 (lần đầu)

Repo đã có sẵn `data/query_vectors.json`. Nếu clone mới hoặc đổi CSV:

```bash
python -m scripts.seed_outfits_csv
python -m scripts.encode_outfits      # cần requirements-dev (FashionCLIP)
python -m scripts.encode_text_queries
```

Kiểm tra readiness:

```bash
curl http://127.0.0.1:8000/health/ready
```

`recommendation.ready` phải là `true` trước khi demo F3.

### 4. Chạy server

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health  

---

## API nhanh

| Method | Path | Mô tả |
|--------|------|--------|
| `POST` | `/api/v1/analyze/color` | F1 — ảnh selfie + header `X-Device-ID` |
| `POST` | `/api/v1/analyze/body` | F2 — ảnh toàn thân |
| `GET` | `/api/v1/analyze/history` | Lịch sử F1/F2 |
| `GET` | `/api/v1/outfits/recommend` | F3 — query `occasion`, `gender=male` |
| `GET` | `/api/v1/outfits/{id}` | Chi tiết outfit |
| `GET` | `/health` | Liveness |
| `GET` | `/health/ready` | F3 catalog / vectors sẵn sàng |

Header bắt buộc cho analyze & recommend:

```http
X-Device-ID: <sha256-64-hex-chars>
```

---

## Fallback khi Gemini hết quota

Khi API trả **429** hoặc **timeout**, F1/F2 tự động dùng **demo cache** (`app/assets/demo_cache/`) và đánh dấu:

```json
"analysis_source": "degraded_demo"
```

App mobile hiển thị banner cảnh báo. F3 vẫn chạy nếu đã có kết quả F1/F2 (kể cả degraded).

---

## Chạy test

```bash
pytest
# hoặc
pytest tests/test_api_demo_smoke.py -q
```

---

## Deploy gợi ý

- Mount volume cho SQLite production: `DATABASE_URL=sqlite:////data/sac_voc.db`
- Upload `sac_voc.db` đã seed + giữ `data/query_vectors.json` trên server.
- Đổi `baseUrl` ở Flutter (`lib/core/constants/api_constants.dart`) trỏ tới URL public.

---

## License / nhóm

Dự án môn học **CS4153** — UIT. Liên hệ nhóm phát triển qua repository Git của course.
