"""
Kết nối SQLite — Session Factory cho FastAPI.

Tuân thủ:
  - backend-python #2: Depends(get_db) inject DB Session
  - backend-python #4: DATABASE_URL từ Settings (pydantic-settings)
  - Tính năng #7: SQLite + Railway Volume mount

LƯU Ý:
  - SQLite trên Railway: file DB nằm trong volume mount (/data/sac_voc.db)
  - connect_args check_same_thread=False: bắt buộc với SQLite + FastAPI (async)
  - Tạo bảng tự động khi app khởi động (create_all)
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from typing import Generator

from app.core.config import settings


# ═══════════════════════════════════════════════
# ENGINE + SESSION FACTORY
# ═══════════════════════════════════════════════

engine = create_engine(
    settings.DATABASE_URL,
    # SQLite bắt buộc — cho phép dùng từ nhiều thread trong FastAPI
    connect_args={"check_same_thread": False},
    # Log SQL ra console khi DEBUG=True — tắt khi production
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ═══════════════════════════════════════════════
# BASE CLASS — Tất cả model kế thừa từ đây
# ═══════════════════════════════════════════════

class Base(DeclarativeBase):
    """Base class cho mọi SQLAlchemy model.
    Dùng DeclarativeBase (SQLAlchemy 2.0 style) thay vì declarative_base() cũ.
    """
    pass


# ═══════════════════════════════════════════════
# DEPENDENCY INJECTION — Dùng trong FastAPI endpoint
# ═══════════════════════════════════════════════

def get_db() -> Generator[Session, None, None]:
    """Inject DB Session vào endpoint qua Depends(get_db).

    Tuân thủ backend-python #2 — Dependency Injection pattern.

    Sử dụng trong router:
        @router.get("/outfits")
        def get_outfits(db: Session = Depends(get_db)):
            ...

    Session tự đóng sau mỗi request (finally block).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ═══════════════════════════════════════════════
# INIT — Gọi khi app khởi động
# ═══════════════════════════════════════════════

def init_db() -> None:
    """Tạo tất cả bảng nếu chưa tồn tại.

    Gọi trong main.py → lifespan event:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            init_db()
            yield

    LƯU Ý: Chỉ tạo bảng mới — KHÔNG xóa/thay đổi bảng đã có.
    Nếu cần migration → dùng Alembic (COULD — không cần cho MVP).
    """
    Base.metadata.create_all(bind=engine)
