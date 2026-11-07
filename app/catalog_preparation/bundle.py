"""Build a corrected, versioned text/image bundle without replacing the baseline.

Intuition: a new coordinate system needs a new identity. The old query bundle
remains evidence, not the acceptance target for the corrected geometry.
Schema: the familiar query JSON/metadata and catalog.db form one bundle; reports
record source availability, reference-path checks and hashes of every artifact.
Pipeline: freeze baseline hashes -> generate queries -> verify ONE projection
through independent model components -> audit images -> seed NEW DB -> verify
and encode available images -> read back -> publish the final bundle manifest.

The reference methods bypass CLIP's feature API and our compatibility adapter.
Comparing repeated calls through the same buggy adapter would not expose a
shared projection bug. Model training, Gemini and relevance labels are absent.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from app.services.recommendation.query_builder import (
    _all_query_templates,
    _current_query_signature,
)
from app.services.recommendation.types import PROJECTED_EMBEDDING_VERSION

from .compatibility import (
    MAX_COORDINATE_ERROR,
    normalized_vectors,
    read_reference_queries,
    verify_compatibility,
)
from .encoder import file_sha256
from .sources import audit_sources
from .workflow import (
    CatalogEncoder,
    encode_available_images,
    read_seed_rows,
    seed_separate_database,
    write_json,
)

BUNDLE_IDENTITY = {
    "backend": "fashion_clip",
    "model": "fashion-clip",
    "version": PROJECTED_EMBEDDING_VERSION,
}


class ReferenceCatalogEncoder(CatalogEncoder, Protocol):
    def encode_reference_text(self, texts: list[str]) -> np.ndarray: ...
    def encode_reference_image(self, image_path: Path) -> np.ndarray: ...


@dataclass
class ReferenceTextEncoder:
    encoder: ReferenceCatalogEncoder

    def encode_text(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode_reference_text(texts)


class VerifiedImageEncoder:
    """Require every saved image vector to agree with one explicit projection."""

    def __init__(self, encoder: ReferenceCatalogEncoder) -> None:
        self.encoder = encoder
        self.checks: list[dict] = []

    def encode_image(self, image_path: Path) -> np.ndarray:
        actual = normalized_vectors(self.encoder.encode_image(image_path), 1)
        expected = normalized_vectors(
            self.encoder.encode_reference_image(image_path), 1
        )
        error = float(np.max(np.abs(actual - expected)))
        passed = error <= MAX_COORDINATE_ERROR
        self.checks.append(
            {
                "cache_file": image_path.name,
                "max_coordinate_error": error,
                "passed": passed,
            }
        )
        if not passed:
            raise ValueError(
                "Image features disagree with the one-projection reference"
            )
        print(f"Verified image {len(self.checks)}: {image_path.name[:12]}", flush=True)
        return actual


def validate_saved_bundle(output: Path, expected_rows: int) -> dict:
    """Use existing read-only exporters to check actual saved bytes and identity."""
    from app.evaluation.snapshot_export import read_saved_queries
    from app.evaluation.sqlite_catalog import read_catalog

    queries = read_saved_queries(
        output / "query_vectors.json", output / "query_vectors.meta.json"
    )
    catalog = read_catalog(output / "catalog.db")
    if queries.embedding != catalog.embedding:
        raise ValueError("Saved query/image bundle identities differ")
    if queries.embedding.model_dump() != BUNDLE_IDENTITY:
        raise ValueError("Saved bundle does not have the corrected identity")
    if len(catalog.outfits) + catalog.unencoded_count != expected_rows:
        raise ValueError("Saved catalog lost or added CSV rows")
    return {
        "query_count": len(queries.vectors),
        "encoded_rows": len(catalog.outfits),
        "unencoded_rows": catalog.unencoded_count,
        "identity": queries.embedding.model_dump(),
    }


def build_corrected_bundle(
    csv_path: Path,
    baseline_queries: Path,
    baseline_metadata: Path,
    output: Path,
    encoder_factory: Callable[[], ReferenceCatalogEncoder],
    *,
    preserved_files: tuple[Path, ...] = (),
) -> dict:
    """Refuse existing output; publish manifest only after all validations pass."""
    rows = read_seed_rows(csv_path)
    baseline = read_reference_queries(baseline_queries, baseline_metadata)
    input_paths = (csv_path, baseline_queries, baseline_metadata, *preserved_files)
    inputs = {str(path.resolve()): file_sha256(path) for path in input_paths}
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "building",
        "embedding": BUNDLE_IDENTITY,
        "csv_rows": len(rows),
        "input_sha256": inputs,
        "application_switched": False,
    }
    write_json(output / "summary.json", summary)
    try:
        print(
            "Loading pinned encoder and generating 240 corrected queries...", flush=True
        )
        encoder = encoder_factory()
        write_json(output / "encoder.json", encoder.provenance)
        queries = _all_query_templates()
        matrix = normalized_vectors(encoder.encode_text(queries), 240)
        generated = {query: vector.tolist() for query, vector in zip(queries, matrix)}
        write_json(output / "query_vectors.json", generated)
        write_json(
            output / "query_vectors.meta.json",
            {
                **BUNDLE_IDENTITY,
                "query_count": 240,
                "query_signature": _current_query_signature(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        # Read the JSON back before checking its geometry, not just in-memory data.
        generated = read_reference_queries(
            output / "query_vectors.json",
            output / "query_vectors.meta.json",
            identity=BUNDLE_IDENTITY,
        )
        check = verify_compatibility(ReferenceTextEncoder(encoder), generated)
        write_json(output / "query-reference-check.json", check)
        if not check["compatible"]:
            raise ValueError("Corrected queries disagree with one-projection reference")
        baseline_check = verify_compatibility(encoder, baseline)
        baseline_check["interpretation"] = (
            "Diagnostic comparison only. A mismatch with the preserved legacy "
            "queries is expected after correcting projection."
        )
        write_json(output / "baseline-comparison.json", baseline_check)
        print("Query reference check passed. Auditing image sources...", flush=True)
        sources = audit_sources(rows, output / "images")
        write_json(output / "sources.json", {"sources": [s.to_dict() for s in sources]})
        database = output / "catalog.db"
        seed_separate_database(database, csv_path, rows)
        image_encoder = VerifiedImageEncoder(encoder)
        count = encode_available_images(
            database,
            image_encoder,
            sources,
            output / "images",
            identity=BUNDLE_IDENTITY,
        )
        write_json(
            output / "image-reference-check.json",
            {
                "images_checked": len(image_encoder.checks),
                "passed": all(check["passed"] for check in image_encoder.checks),
                "images": image_encoder.checks,
            },
        )
        if count == 0:
            raise ValueError("No available images could be encoded")
        validation = validate_saved_bundle(output, len(rows))
        write_json(output / "readback-check.json", validation)
        if not all(file_sha256(Path(p)) == digest for p, digest in inputs.items()):
            raise ValueError("Baseline inputs changed while building the bundle")
        artifacts = (
            "catalog.db",
            "query_vectors.json",
            "query_vectors.meta.json",
            "encoder.json",
            "sources.json",
            "query-reference-check.json",
            "image-reference-check.json",
            "baseline-comparison.json",
            "readback-check.json",
        )
        write_json(
            output / "manifest.json",
            {
                "embedding": BUNDLE_IDENTITY,
                "query_signature": _current_query_signature(),
                "artifact_sha256": {
                    name: file_sha256(output / name) for name in artifacts
                },
                "baseline_sha256": inputs,
            },
        )
        summary.update(
            status="verified_bundle",
            encoded_rows=count,
            unencoded_rows=len(rows) - count,
            unique_urls=len(sources),
            encoded_unique_urls=len(image_encoder.checks),
            unavailable_csv_rows=[
                row
                for source in sources
                if source.status != "ready"
                for row in source.csv_rows
            ],
            input_files_unchanged=True,
        )
    except Exception as exc:
        # Keep diagnostic evidence, but never leave a failed run marked ready.
        summary.update(status="failed", error=str(exc))
        raise
    finally:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "summary.json", summary)
    return summary
