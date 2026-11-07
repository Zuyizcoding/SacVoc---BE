"""A frozen input snapshot for fair F3 experiments.

Intuition: changing the candidate pool while changing the ranker confounds the
comparison. A snapshot freezes profiles, query/image vectors, encoder identity,
catalog order, and independently supplied relevance judgments in one file.
Synthetic vectors are explicitly labeled; they can never masquerade as real
FashionCLIP embeddings. Validation checks shape/unit norm, not visual meaning.
"""

from typing import Annotated, Literal

import numpy as np
from pydantic import Field, FiniteFloat, model_validator

from app.data.body_shapes import MALE_BODY_SHAPES
from app.services.recommendation.query_builder import (
    SEASON_KEYWORDS,
    _current_query_signature,
    build_outfit_query,
)

from .schemas import RelevanceGrade, StrictModel

UnitVector = Annotated[list[FiniteFloat], Field(min_length=512, max_length=512)]
Signal = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


def require_unit_vector(values: list[float]) -> None:
    if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-3, rtol=0):
        raise ValueError(
            "Vectors must already be normalized; evaluation never repairs them"
        )


class EmbeddingIdentity(StrictModel):
    backend: Literal["synthetic", "fashion_clip", "hf_clip"]
    model: str = Field(min_length=1)
    version: str = Field(min_length=1)


class RetrievalProfile(StrictModel):
    color_sub_season: str
    body_shape: str
    occasion: Literal["daily", "work", "party", "date"] = "daily"
    gender: Literal["male"] = "male"
    color_confidence: Signal | None = None
    body_confidence: Signal | None = None
    color_photo_quality_score: Signal | None = None
    body_photo_quality_score: Signal | None = None

    @model_validator(mode="after")
    def validate_taxonomy(self):
        if (
            self.color_sub_season not in SEASON_KEYWORDS
            or self.body_shape not in MALE_BODY_SHAPES
        ):
            raise ValueError("Use canonical color seasons and male body shapes")
        return self


class SnapshotOutfit(StrictModel):
    """A plain record with the same attributes read from an ORM outfit row."""

    id: int = Field(gt=0, strict=True)
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    image_url: str
    shopee_url: str = ""
    price_range: str = ""
    occasion: str
    gender: str
    season_bias: str | None = None
    body_shape_bias: str | None = None
    color_family: str | None = None
    fit: str | None = None
    formality: str | None = None
    vector: UnitVector

    @model_validator(mode="after")
    def validate_vector(self):
        require_unit_vector(self.vector)
        return self


class RetrievalCase(StrictModel):
    case_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    subject_id: str = Field(min_length=1)
    split: Literal["dev", "test"]
    profile: RetrievalProfile
    query_text: str
    query_vector: UnitVector
    relevance: dict[str, RelevanceGrade] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_query(self):
        expected = build_outfit_query(
            self.profile.color_sub_season,
            self.profile.body_shape,
            self.profile.occasion,
        )
        if self.query_text != expected:
            raise ValueError("query_text must match the unchanged production template")
        require_unit_vector(self.query_vector)
        return self


class RetrievalSnapshot(StrictModel):
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    synthetic: bool
    annotation_protocol: str = Field(min_length=1)
    embedding: EmbeddingIdentity
    query_signature: str
    outfits: list[SnapshotOutfit] = Field(min_length=1)
    cases: list[RetrievalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_snapshot(self):
        if self.synthetic != (self.embedding.backend == "synthetic"):
            raise ValueError("Synthetic snapshots require synthetic embedding identity")
        if self.query_signature != _current_query_signature():
            raise ValueError("Query signature does not match production templates")
        outfit_ids = {str(outfit.id) for outfit in self.outfits}
        if len(outfit_ids) != len(self.outfits):
            raise ValueError("Duplicate outfit IDs")
        case_ids: set[str] = set()
        subject_splits: dict[str, str] = {}
        for case in self.cases:
            if case.case_id in case_ids:
                raise ValueError("Duplicate retrieval case IDs")
            case_ids.add(case.case_id)
            if subject_splits.setdefault(case.subject_id, case.split) != case.split:
                raise ValueError("Subject appears in both dev and test")
            if set(case.relevance) - outfit_ids:
                raise ValueError("Relevance references an unknown outfit ID")
        return self
