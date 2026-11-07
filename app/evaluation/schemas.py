"""Dataset contracts: stable IDs, subject-separated splits, and reference labels.

Intuition: a benchmark is a versioned agreement about what to measure. A reference
label is a human judgment, not an objective measurement of a person's appearance.
Synthetic cases test software only and are explicitly separated from human data.
Paths are resolved relative to the manifest, never to the current working folder.
"""

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.data.body_shapes import MALE_BODY_SHAPES
from app.data.season_data import SEASON_DATA

Feature = Literal["color", "body"]
PRIMARY_SEASONS = ("Spring", "Summer", "Autumn", "Winter")
RelevanceGrade = Annotated[int, Field(strict=True, ge=0, le=3)]


class StrictModel(BaseModel):
    # Reject misspelled fields instead of silently losing annotation information.
    model_config = ConfigDict(extra="forbid")


class ReferenceLabels(StrictModel):
    primary_season: str | None = None
    sub_season: str | None = None
    shape: str | None = None


def validate_reference_labels(feature: Feature, reference: ReferenceLabels) -> None:
    """One taxonomy check for manifests and stored evaluation evidence."""
    if feature == "color":
        if reference.primary_season not in PRIMARY_SEASONS:
            raise ValueError("Color cases require a canonical primary_season")
        if reference.shape is not None:
            raise ValueError("Color references cannot include body shape")
        if reference.sub_season is not None:
            entry = SEASON_DATA.get(reference.sub_season)
            if not entry or entry["primary"] != reference.primary_season:
                raise ValueError("sub_season must agree with primary_season")
    elif reference.shape not in MALE_BODY_SHAPES:
        raise ValueError("Body cases require a canonical shape")
    elif reference.primary_season is not None or reference.sub_season is not None:
        raise ValueError("Body references cannot include color labels")


class SyntheticReply(StrictModel):
    """A mock provider response; it is never used with the Gemini provider."""

    payload: dict | None = None
    raw_text: str | None = None
    http_status: int = Field(default=200, ge=200, le=599)
    finish_reason: str = "STOP"

    @model_validator(mode="after")
    def validate_body(self):
        if self.payload is not None and self.raw_text is not None:
            raise ValueError("Use either payload or raw_text, not both")
        return self


class AnalysisCase(StrictModel):
    case_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    subject_id: str = Field(min_length=1)
    feature: Feature
    split: Literal["dev", "test"]
    reference: ReferenceLabels
    image_path: Path | None = None
    consent_confirmed: bool = False
    synthetic_replies: list[SyntheticReply] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_reference(self):
        validate_reference_labels(self.feature, self.reference)
        return self


class AnalysisDataset(StrictModel):
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    synthetic: bool
    reference_kind: Literal["synthetic", "human_consensus"]
    annotation_protocol: str = Field(min_length=1)
    cases: list[AnalysisCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dataset(self):
        if self.synthetic != (self.reference_kind == "synthetic"):
            raise ValueError("synthetic and reference_kind must agree")
        case_ids: set[str] = set()
        subject_splits: dict[str, str] = {}
        for case in self.cases:
            if case.case_id in case_ids:
                raise ValueError(f"Duplicate case_id: {case.case_id}")
            case_ids.add(case.case_id)
            previous = subject_splits.setdefault(case.subject_id, case.split)
            if previous != case.split:
                raise ValueError(f"Subject appears in dev and test: {case.subject_id}")
            if self.synthetic:
                if case.image_path is not None or not case.synthetic_replies:
                    raise ValueError(
                        "Synthetic cases use generated images and scripted replies"
                    )
            elif not case.consent_confirmed or case.image_path is None:
                raise ValueError("Human cases require consent and image_path")
            elif case.synthetic_replies:
                raise ValueError("Human cases cannot contain synthetic replies")
        return self


def load_dataset(path: Path) -> AnalysisDataset:
    return AnalysisDataset.model_validate_json(path.read_text(encoding="utf-8-sig"))


class RankingCase(StrictModel):
    """Independent relevance judgments: outfit ID -> grade (0..3).

    Every returned item must be judged. An unjudged item is unknown, not irrelevant.
    The judgment pool should be shared across compared ranking systems.
    """

    case_id: str = Field(min_length=1)
    relevance: dict[str, RelevanceGrade] = Field(min_length=1)


class RankingDataset(StrictModel):
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    synthetic: bool
    cases: list[RankingCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_case_ids(self):
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Ranking case IDs must be unique")
        return self


def load_ranking_dataset(path: Path) -> RankingDataset:
    return RankingDataset.model_validate(
        json.loads(path.read_text(encoding="utf-8-sig"))
    )
