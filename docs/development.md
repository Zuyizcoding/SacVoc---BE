# Chạy và kiểm tra backend

## Môi trường

Dùng Python 3.12+; `.python-version` ghi 3.12.13. Các lệnh sau chạy ở thư mục backend:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-test.txt
Copy-Item .env.example .env
```

Điền `GEMINI_API_KEY` trong `.env` rồi chạy:

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

`/docs` là Swagger; `/health` kiểm tra process; `/health/ready` kiểm tra catalog/vector phục vụ F3. Readiness mặc định kiểm tra trường hợp daily/male, không xác nhận mọi tổ hợp dịp sử dụng.

## Tests và style

```powershell
python -m pytest -q
ruff check app scripts tests
ruff format --check app scripts tests
```

`tests/conftest.py` đặt DB trong bộ nhớ và khóa giả trước khi import app. Tests không cần `.env`, không gọi Gemini, không tải FashionCLIP và không sửa DB demo. Client tests cover the real pinned SDK over fake HTTP as well as scripted inference; backoff waits are disabled in tests. Tests F3 dùng vectors cố định và kiểm tra query signature đã commit.

See the English [evaluation guide](evaluation.md) for the isolated runner, dataset schemas, pipeline intuition, metrics, and synthetic smoke commands.

See [experiments.md](experiments.md) for F3 ablation, paired analysis comparisons,
and commands matching the backend GitHub Actions workflow. The lightweight
test requirements do not install FashionCLIP or Torch.

## Catalog và embeddings

Các tác vụ offline cần thêm dependencies trong `requirements-dev.txt`, có thể tải model lớn. Chỉ chạy khi cần chuẩn bị catalog thực:

```powershell
python -m scripts.seed_outfits_csv
python -m scripts.encode_outfits
python -m scripts.encode_text_queries
```

Các lệnh seed/encode ghi DB hoặc file vectors. `--dry-run` của seed/prune rollback thay đổi row nhưng vẫn gọi `init_db`, nên muốn kiểm tra hoàn toàn tách biệt hãy trỏ `DATABASE_URL` tới DB tạm. Script dummy encoder không tạo semantic embeddings hợp lệ cho recommendation.

## Khi gặp lỗi

- Missing Gemini key: offline schema/evaluation tools can import settings without credentials. Constructing a real Gemini client requires `GEMINI_API_KEY` in the environment or `.env`; the synthetic provider needs no key.
- F3 chưa sẵn sàng: xem `/health/ready`; kiểm tra có catalog, vector thật và metadata cùng backend/model/version.
- Query signature không khớp: template và vectors thuộc hai phiên bản khác nhau. Khôi phục bộ đồng bộ hoặc chủ động encode lại cả bộ khi thực sự đổi template.
- AI hiển thị demo: xem `analysis_source` và log retry/breaker; kiểm tra quota và cấu hình timeout.
- Migration cảnh báo trong log: kiểm tra schema của DB hiện hành; cơ chế tự migration vẫn giữ chính sách bỏ qua lỗi của bản MVP.
- Thiết bị không thấy kết quả cũ: xác nhận header device ID và DB đang sử dụng; history gắn với thiết bị.
