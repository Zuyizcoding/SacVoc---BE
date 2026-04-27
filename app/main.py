"""
Entry point FastAPI — BỔ SUNG init_db() so với CORE_MUST.

Tuân thủ: backend-python #1 — app/main.py khởi tạo ứng dụng.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.database import init_db
from app.api.endpoints import analyze
from app.api.endpoints import outfit  # Phase 5


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: tạo bảng DB nếu chưa có. Shutdown: cleanup nếu cần."""
    init_db()  # ← Tính năng #7: tạo bảng lần đầu
    yield
    # Shutdown logic (nếu có) đặt ở đây


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,  # ← thay on_event("startup") deprecated
)

# --- CORS (cho phép Flutter gọi API) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(analyze.router, tags=["analyze"])  # Phase 3
app.include_router(outfit.router, prefix="/api/v1", tags=["outfit"])  # Phase 5


@app.get("/health")
async def health():
    """Tính năng #35 — warm-up + verify deploy."""
    return {"status": "ok", "version": settings.APP_VERSION}
