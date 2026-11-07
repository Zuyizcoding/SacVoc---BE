"""Annotation contracts: profiles first, independent grades later.

The coordinator's packet freezes context and catalog identity. Reviewer files
contain only display fields and blank grades; private bias tags, vectors, model
scores and confidence signals are never included in those files.
"""

from typing import Literal

from pydantic import Field, model_validator

from .retrieval_schema import RetrievalProfile
from .schemas import StrictModel

TaskKey = tuple[str, str]


class AnnotationCase(StrictModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    subject_id: str = Field(min_length=1)
    split: Literal["dev", "test"]
    profile: RetrievalProfile


class AnnotationPlan(StrictModel):
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    synthetic: bool
    annotation_protocol: str = Field(min_length=1)
    cases: list[AnnotationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_identity(self):
        ids = set()
        splits = {}
        for case in self.cases:
            if case.case_id in ids:
                raise ValueError("Duplicate annotation case IDs")
            ids.add(case.case_id)
            if splits.setdefault(case.subject_id, case.split) != case.split:
                raise ValueError("Subject appears in both dev and test")
        return self


class AnnotationTask(StrictModel):
    case_id: str
    outfit_id: str
    season: str
    shape: str
    occasion: str
    outfit_name: str
    category: str
    image_url: str

    @property
    def key(self) -> TaskKey:
        return self.case_id, self.outfit_id


class AnnotationPacket(StrictModel):
    schema_version: Literal[1] = 1
    packet_id: str = Field(min_length=1)
    plan: AnnotationPlan
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewers: list[str] = Field(min_length=2, max_length=2)
    tasks: list[AnnotationTask] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tasks(self):
        if len(set(self.reviewers)) != 2:
            raise ValueError("Two distinct reviewer IDs are required")
        keys = [task.key for task in self.tasks]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate annotation task keys")
        cases = {case.case_id: case for case in self.plan.cases}
        if {task.case_id for task in self.tasks} != set(cases):
            raise ValueError("Every planned case needs annotation tasks")
        for task in self.tasks:
            profile = cases[task.case_id].profile
            if (task.season, task.shape, task.occasion) != (
                profile.color_sub_season,
                profile.body_shape,
                profile.occasion,
            ):
                raise ValueError("Task context disagrees with its profile")
        return self
