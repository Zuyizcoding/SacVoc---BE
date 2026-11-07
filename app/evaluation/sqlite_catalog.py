"""Read an existing outfit catalog without application startup or migrations.

Intuition: importing the app's DB module can create an engine tied to user
configuration. This boundary instead receives an explicit Path and opens SQLite
with mode=ro. query_only adds a second barrier against accidental SQL writes.
Only outfits are read: no device IDs, analysis history, init_db, or encoder calls.

A normal SQLite read transaction includes committed WAL data. SQLite may use its
normal lock/shared-memory mechanisms; use a consistent offline DB copy when the
source must have absolutely no sidecar activity.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from app.services.analysis.tracing import content_hash

from .retrieval_schema import EmbeddingIdentity, SnapshotOutfit

CATALOG_FIELDS = (
    "id",
    "name",
    "category",
    "image_url",
    "shopee_url",
    "price_range",
    "occasion",
    "gender",
    "season_bias",
    "body_shape_bias",
    "color_family",
    "fit",
    "formality",
)


@dataclass(frozen=True)
class CatalogExportData:
    outfits: list[SnapshotOutfit]
    embedding: EmbeddingIdentity
    unencoded_count: int


def catalog_fingerprint(outfits: list[SnapshotOutfit]) -> str:
    """Bind annotation and export to the exact same ordered records/vectors."""
    payload = json.dumps(
        [outfit.model_dump() for outfit in outfits],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return content_hash(payload)


@contextmanager
def readonly_catalog(database_path: Path) -> Iterator[sqlite3.Connection]:
    """Never create a missing DB; all statements run against one read snapshot."""
    source = database_path.resolve(strict=True)
    if not source.is_file():
        raise ValueError("Catalog path must refer to an existing SQLite file")
    connection = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.rollback()
        connection.close()


def read_catalog(database_path: Path) -> CatalogExportData:
    """Validate all encoded records, retaining ID order and image duplicates.

    Export is deliberately stricter than serving one profile: every exported
    encoded row needs valid vectors and compatible metadata, even if a later
    filter would hide it. Unencoded rows are counted and excluded, as in runtime.
    """
    fields = ", ".join(
        (
            *CATALOG_FIELDS,
            "vector_blob",
            "embedding_backend",
            "embedding_model",
            "embedding_version",
        )
    )
    with readonly_catalog(database_path) as connection:
        try:
            rows = connection.execute(
                f"SELECT {fields} FROM outfits WHERE vector_blob IS NOT NULL ORDER BY id"
            ).fetchall()
            unencoded_count = connection.execute(
                "SELECT COUNT(*) FROM outfits WHERE vector_blob IS NULL"
            ).fetchone()[0]
        except sqlite3.DatabaseError as exc:
            raise ValueError(
                "Catalog lacks the expected outfits schema; no migration was attempted"
            ) from exc
    if not rows:
        raise ValueError("Catalog has no encoded outfits")
    outfits = []
    identity = None
    for row in rows:
        current_identity = EmbeddingIdentity(
            backend=row["embedding_backend"],
            model=row["embedding_model"],
            version=row["embedding_version"],
        )
        if identity is not None and current_identity != identity:
            raise ValueError("Encoded catalog rows have mixed embedding identities")
        identity = current_identity
        blob = row["vector_blob"]
        if not isinstance(blob, bytes) or len(blob) != 512 * 4:
            raise ValueError(f"Outfit {row['id']} needs a 512-dimensional float32 BLOB")
        # Existing encoders store little-endian float32 on the project's platforms.
        vector = np.frombuffer(blob, dtype="<f4").tolist()
        outfits.append(
            SnapshotOutfit(
                **{field: row[field] for field in CATALOG_FIELDS}, vector=vector
            )
        )
    return CatalogExportData(
        outfits=outfits, embedding=identity, unencoded_count=unencoded_count
    )
