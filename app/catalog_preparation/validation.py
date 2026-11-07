"""Check saved bundle serving behavior without AI calls or human relevance labels.

This is a software/data check, not a fashion-quality benchmark. Open the new DB
read-only, compare ORM and snapshot filtering, and run the unchanged ranker twice
for every supported profile. No device profile is created and no settings change.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.evaluation.retrieval import candidates_for_profile
from app.evaluation.retrieval_schema import RetrievalProfile
from app.evaluation.snapshot_export import read_saved_queries
from app.evaluation.sqlite_catalog import read_catalog
from app.services.recommendation.catalog import (
    _load_outfit_embedding_meta,
    _load_outfit_vectors,
)
from app.services.recommendation.embedding_store import _assert_query_vector_consistency
from app.services.recommendation.query_builder import (
    BODY_KEYWORDS,
    OCCASION_KEYWORDS,
    SEASON_KEYWORDS,
    build_outfit_query,
)
from app.services.recommendation.ranking import (
    _cosine_similarity_search,
    _rank_candidates,
    _select_diverse_top_k,
)
from app.services.recommendation.types import QueryVectorBundle

from .bundle import BUNDLE_IDENTITY


def verify_serving_paths(directory: Path) -> dict:
    """Verify all 12 x 5 x 4 profiles, using only the explicit bundle paths."""
    database = (directory / "catalog.db").resolve(strict=True)
    queries = read_saved_queries(
        directory / "query_vectors.json", directory / "query_vectors.meta.json"
    )
    catalog = read_catalog(database)
    if (
        catalog.embedding != queries.embedding
        or queries.embedding.model_dump() != BUNDLE_IDENTITY
    ):
        raise ValueError(
            "Bundle query and image identities must match the corrected version"
        )

    def connect_readonly():
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        return connection

    with closing(connect_readonly()) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
        if (
            connection.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]
            != 0
        ):
            raise ValueError("Preparation DB must not contain device history")

    engine = create_engine("sqlite://", creator=connect_readonly)
    results = []
    try:
        with Session(engine) as session:
            # Only the occasion changes the candidate pool. Load each once.
            pools = {
                occasion: _load_outfit_vectors(session, occasion, "male")
                for occasion in OCCASION_KEYWORDS
            }
            for season in SEASON_KEYWORDS:
                for body in BODY_KEYWORDS:
                    for occasion in OCCASION_KEYWORDS:
                        profile = RetrievalProfile(
                            color_sub_season=season, body_shape=body, occasion=occasion
                        )
                        outfits, matrix = pools[occasion]
                        if not outfits:
                            raise ValueError(f"No encoded male outfits for {occasion}")
                        exported_pool = candidates_for_profile(catalog.outfits, profile)
                        if [o.id for o in outfits] != [o.id for o in exported_pool]:
                            raise ValueError(
                                "Runtime and export candidate filters disagree"
                            )
                        text = build_outfit_query(season, body, occasion)
                        vector = np.asarray(queries.vectors[text], dtype=np.float32)
                        _assert_query_vector_consistency(
                            QueryVectorBundle(
                                vector=vector,
                                source="explicit_bundle_verification",
                                **queries.embedding.model_dump(),
                            ),
                            _load_outfit_embedding_meta(outfits),
                        )
                        scores = _cosine_similarity_search(vector, matrix)

                        def select():
                            ranked = _rank_candidates(
                                outfits,
                                scores,
                                color_sub_season=season,
                                body_shape=body,
                                occasion=occasion,
                            )
                            return _select_diverse_top_k(ranked, 6)

                        first, second = select(), select()

                        def signature(selected):
                            return [
                                (c.outfit.id, c.final_score, c.retrieval_score)
                                for c in selected
                            ]

                        if signature(first) != signature(second):
                            raise ValueError(
                                "Repeated ranking produced different results"
                            )
                        if len(first) != min(6, len(outfits)):
                            raise ValueError("Unexpected result count")
                        images = [c.outfit.image_url for c in first]
                        if len(set(images)) != len(images):
                            raise ValueError(
                                "Duplicate image in selected recommendations"
                            )
                        if not np.isfinite(scores).all() or any(
                            not 0 <= c.final_score <= 1 for c in first
                        ):
                            raise ValueError("Invalid recommendation scores")
                        results.append(
                            {
                                "profile": profile.model_dump(),
                                "candidate_count": len(outfits),
                                "outfit_ids": [c.outfit.id for c in first],
                                "match_scores": [c.final_score for c in first],
                                "cosine_scores": [c.retrieval_score for c in first],
                            }
                        )
    finally:
        engine.dispose()
    return {
        "passed": True,
        "profiles_checked": len(results),
        "embedding": queries.embedding.model_dump(),
        "interpretation": "Read-only serving/filter/determinism check; not fashion accuracy.",
        "profiles": results,
    }
