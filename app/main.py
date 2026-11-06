"""Khởi tạo FastAPI, router và vòng đời database."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import analyze, outfit
from app.core.config import settings
from app.db.database import SessionLocal, init_db
from app.services.outfit_service import get_recommendation_readiness


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: tạo bảng DB nếu chưa có. Shutdown: cleanup nếu cần."""
    init_db()
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
app.include_router(analyze.router, tags=["analyze"])
app.include_router(outfit.router, prefix="/api/v1", tags=["outfit"])


@app.get("/health")
async def health():
    """Tính năng #35 — warm-up + verify deploy."""
    return {"status": "ok", "version": settings.APP_VERSION}


@app.get("/health/ready")
async def health_ready():
    """Readiness probe for demo/preflight checks, especially F3."""
    db = SessionLocal()
    try:
        recommendation = get_recommendation_readiness(db)
        return {
            "status": "ok" if recommendation["ready"] else "degraded",
            "version": settings.APP_VERSION,
            "recommendation": recommendation,
        }
    finally:
        db.close()
