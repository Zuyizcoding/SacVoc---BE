"""Engine, session theo request và migration SQLite tối thiểu."""

import logging
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

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

    # Lightweight migration for MVP: add missing columns on existing SQLite DB.
    # SQLite supports ALTER TABLE ... ADD COLUMN.
    try:
        with engine.begin() as conn:
            # PRAGMA table_info returns rows: (cid, name, type, notnull, dflt_value, pk)
            cols = conn.execute(text("PRAGMA table_info(outfits)")).fetchall()
            col_names = {row[1] for row in cols}
            pending_columns = {
                "gender": "ALTER TABLE outfits ADD COLUMN gender VARCHAR(10) NOT NULL DEFAULT 'any'",
                "fit": "ALTER TABLE outfits ADD COLUMN fit VARCHAR(20)",
                "silhouette": "ALTER TABLE outfits ADD COLUMN silhouette VARCHAR(30)",
                "formality": "ALTER TABLE outfits ADD COLUMN formality VARCHAR(20)",
                "fabric": "ALTER TABLE outfits ADD COLUMN fabric VARCHAR(30)",
                "color_family": "ALTER TABLE outfits ADD COLUMN color_family VARCHAR(30)",
                "contrast_level": "ALTER TABLE outfits ADD COLUMN contrast_level VARCHAR(20)",
                "body_shape_bias": "ALTER TABLE outfits ADD COLUMN body_shape_bias TEXT",
                "season_bias": "ALTER TABLE outfits ADD COLUMN season_bias TEXT",
                "embedding_backend": "ALTER TABLE outfits ADD COLUMN embedding_backend VARCHAR(30)",
                "embedding_model": "ALTER TABLE outfits ADD COLUMN embedding_model VARCHAR(80)",
                "embedding_version": "ALTER TABLE outfits ADD COLUMN embedding_version VARCHAR(40)",
                "embedding_generated_at": "ALTER TABLE outfits ADD COLUMN embedding_generated_at DATETIME",
            }
            for column_name, statement in pending_columns.items():
                if column_name not in col_names:
                    conn.execute(text(statement))
    except Exception:
        logging.getLogger(__name__).warning(
            "Không áp dụng được migration SQLite tự động", exc_info=True
        )
