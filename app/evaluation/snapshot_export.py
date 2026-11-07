"""Export pipeline: request -> committed vectors -> read-only catalog -> snapshot.

The request contains profiles and independent relevance judgments, not generated
embeddings. Every query is built with the production template and looked up in the
supplied precomputed bundle. Missing/incompatible data is an error: export never
loads an encoder, invents relevance, normalizes vectors, or falls back to dummy
data. The resulting snapshot can enter the existing preflight/ablation tools.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, TypeAdapter, model_validator

from app.services.analysis.tracing import content_hash
from app.services.recommendation.query_builder import (
    _all_query_templates,
    _current_query_signature,
    build_outfit_query,
)

from .provenance import source_fingerprint
from .retrieval import candidate_pool, eligible_judgments
from .retrieval_schema import (
    EmbeddingIdentity,
    RetrievalCase,
    RetrievalProfile,
    RetrievalSnapshot,
    UnitVector,
    require_unit_vector,
)
from .schemas import RelevanceGrade, StrictModel
from .sqlite_catalog import catalog_fingerprint, read_catalog


class ExportCase(StrictModel):
    case_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    subject_id: str = Field(min_length=1)
    split: Literal["dev", "test"]
    profile: RetrievalProfile
    relevance: dict[str, RelevanceGrade] = Field(min_length=1)


class ExportRequest(StrictModel):
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    synthetic: bool
    annotation_protocol: str = Field(min_length=1)
    cases: list[ExportCase] = Field(min_length=1)
    expected_catalog_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    annotation_packet_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_case_identity(self):
        case_ids: set[str] = set()
        subject_splits: dict[str, str] = {}
        for case in self.cases:
            if case.case_id in case_ids:
                raise ValueError("Duplicate export case IDs")
            case_ids.add(case.case_id)
            if subject_splits.setdefault(case.subject_id, case.split) != case.split:
                raise ValueError("Subject appears in both dev and test")
        return self


class QueryManifest(EmbeddingIdentity):
    query_count: int = Field(gt=0, strict=True)
    query_signature: str
    generated_at: str | None = None


@dataclass(frozen=True)
class SavedQueries:
    vectors: dict[str, list[float]]
    embedding: EmbeddingIdentity
    vectors_sha256: str
    metadata_sha256: str


def read_saved_queries(vectors_path: Path, metadata_path: Path) -> SavedQueries:
    """Validate the complete 240-query bundle instead of silently skipping rows."""
    vector_bytes, metadata_bytes = vectors_path.read_bytes(), metadata_path.read_bytes()
    manifest = QueryManifest.model_validate_json(metadata_bytes.decode("utf-8-sig"))
    if manifest.query_signature != _current_query_signature():
        raise ValueError("Query metadata signature does not match current templates")
    vectors = json.loads(vector_bytes.decode("utf-8-sig"))
    if not isinstance(vectors, dict) or set(vectors) != set(_all_query_templates()):
        raise ValueError(
            "Saved queries must contain exactly the 240 production templates"
        )
    if manifest.query_count != len(vectors):
        raise ValueError("query_count disagrees with the saved vector bundle")
    # Pydantic validates finite values/length before normalization is checked.
    adapter = TypeAdapter(UnitVector)
    validated = {}
    for text, values in vectors.items():
        validated[text] = adapter.validate_python(values)
        require_unit_vector(validated[text])
    identity = EmbeddingIdentity.model_validate(
        manifest.model_dump(include={"backend", "model", "version"})
    )
    return SavedQueries(
        validated, identity, content_hash(vector_bytes), content_hash(metadata_bytes)
    )


def export_snapshot(
    request_path: Path,
    database_path: Path,
    vectors_path: Path,
    metadata_path: Path,
    output_dir: Path,
) -> dict:
    """Validate completely before creating output; never overwrite an experiment."""
    request_bytes = request_path.read_bytes()
    request = ExportRequest.model_validate_json(request_bytes.decode("utf-8-sig"))
    queries = read_saved_queries(vectors_path, metadata_path)
    if request.synthetic != (queries.embedding.backend == "synthetic"):
        raise ValueError("Request and query-vector synthetic declarations disagree")
    catalog = read_catalog(database_path)
    catalog_hash = catalog_fingerprint(catalog.outfits)
    if (
        request.expected_catalog_sha256 is not None
        and catalog_hash != request.expected_catalog_sha256
    ):
        raise ValueError(
            "Catalog changed since annotation preparation; review/rebuild the annotation packet"
        )
    if catalog.embedding != queries.embedding:
        raise ValueError("Query and outfit embedding backend/model/version must match")
    cases = []
    for case in request.cases:
        query_text = build_outfit_query(
            case.profile.color_sub_season,
            case.profile.body_shape,
            case.profile.occasion,
        )
        cases.append(
            RetrievalCase(
                **case.model_dump(),
                query_text=query_text,
                query_vector=queries.vectors[query_text],
            )
        )
    snapshot = RetrievalSnapshot(
        dataset_id=request.dataset_id,
        version=request.version,
        synthetic=request.synthetic,
        annotation_protocol=request.annotation_protocol,
        embedding=catalog.embedding,
        query_signature=_current_query_signature(),
        outfits=catalog.outfits,
        cases=cases,
    )
    for case in snapshot.cases:
        eligible_judgments(candidate_pool(snapshot, case), case)
    payload = snapshot.model_dump(mode="json")
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic": request.synthetic,
        "sqlite_version": sqlite3.sqlite_version,
        "source_sha256": source_fingerprint(),
        "request_sha256": content_hash(request_bytes),
        "query_vectors_sha256": queries.vectors_sha256,
        "query_metadata_sha256": queries.metadata_sha256,
        "catalog_snapshot_sha256": catalog_hash,
        "annotation_packet_id": request.annotation_packet_id,
        "encoded_outfit_count": len(catalog.outfits),
        "excluded_unencoded_count": catalog.unencoded_count,
        "case_count": len(cases),
        "embedding": catalog.embedding.model_dump(),
        "catalog_order": "ascending outfit ID",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "snapshot.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return provenance
