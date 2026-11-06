"""Kiểm thử F3 xuyên qua catalog và ranking bằng vectors xác định trước."""

import json
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.database import Base
from app.db.models import Outfit
from app.services.outfit_service import recommend_outfits
from app.services.recommendation.query_builder import (
    _all_query_templates,
    _current_query_signature,
)


def test_precomputed_queries_never_initialize_a_runtime_model(monkeypatch):
    from app.services.recommendation import embedding_store
    from app.services.recommendation.query_builder import build_outfit_query

    def unexpected_model_load():
        raise AssertionError("Precomputed query must not download or load a model")

    monkeypatch.setattr(
        embedding_store, "_get_runtime_text_encoder", unexpected_model_load
    )
    query = build_outfit_query("Soft Autumn", "Rectangle", "daily")
    first = embedding_store._resolve_query_vector(query)
    second = embedding_store._resolve_query_vector(query)
    assert first.source == second.source == "precomputed_query_vector"
    assert first.vector is second.vector


class FixedEncoder:
    def encode_text(self, texts):
        vector = np.zeros((len(texts), 512), dtype=np.float32)
        vector[:, 0] = 1
        return vector


def test_all_precomputed_queries_keep_the_committed_signature():
    data = Path(__file__).resolve().parents[1] / "data"
    vectors = json.loads((data / "query_vectors.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (data / "query_vectors.meta.json").read_text(encoding="utf-8")
    )
    assert len(vectors) == len(_all_query_templates()) == 240
    assert set(vectors) == set(_all_query_templates())
    assert (
        _current_query_signature()
        == metadata["query_signature"]
        == "9c04976c9db0fc3619050c0da9820c187d38c7d6314606f217ca94e9db191fb4"
    )
    assert {len(vector) for vector in vectors.values()} == {512}


def test_recommendation_filters_catalog_and_keeps_scores():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            for ident, season, family, gender, occasion, image in [
                (1, "Soft Autumn", "earth", "male", "daily", "one.jpg"),
                (2, "True Winter", "cool", "male", "daily", "two.jpg"),
                (3, "Soft Autumn", "earth", "female", "daily", "female.jpg"),
                (4, "Soft Autumn", "earth", "male", "work", "work.jpg"),
                (5, "Soft Autumn", "earth", "male", "daily", "one.jpg"),
            ]:
                vector = np.zeros(512, dtype=np.float32)
                vector[0 if ident != 2 else 1] = 1
                db.add(
                    Outfit(
                        id=ident,
                        name=f"item {ident}",
                        category="top",
                        image_url=image,
                        shopee_url="https://example.com/buy",
                        price_range="100k",
                        occasion=occasion,
                        gender=gender,
                        season_bias=season,
                        color_family=family,
                        body_shape_bias="Rectangle",
                        fit="regular",
                        formality="casual",
                        vector_blob=vector.tobytes(),
                        embedding_backend="fashion_clip",
                        embedding_model="fashion-clip",
                        embedding_version="2026-05-mens-v1",
                    )
                )
            db.commit()
            result = recommend_outfits(
                db, "Soft Autumn", "Rectangle", text_encoder=FixedEncoder()
            )
            assert [item.id for item in result.outfits] == [1, 2]
            assert [item.score for item in result.outfits] == pytest.approx(
                [0.9425, 0.5075]
            )
            assert result.outfits[0].retrieval_score == 1
    finally:
        engine.dispose()
