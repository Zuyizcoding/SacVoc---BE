"""Read and validate complete analysis-run artifacts before comparing anything.

Intuition: joining by case ID alone is unsafe. A fair pair also needs the same
manifest/split, references, subject identities, and input image bytes. A model or
prompt may change (that is the experiment); the report must disclose the change.
Reports are recomputed from evidence, never trusted from cached aggregate JSON.
"""

import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.body import BodyApiResponse
from app.schemas.color import ColorApiResponse
from app.services.analysis.tracing import AnalysisTrace

from .schemas import ReferenceLabels, StrictModel, validate_reference_labels

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RunMetadata(BaseModel):
    # Additive metadata stays compatible; the version gates structural changes.
    model_config = ConfigDict(extra="allow")
    artifact_schema_version: Literal[1]
    run_id: str
    dataset_id: str
    dataset_version: str
    manifest_sha256: Digest
    synthetic: bool
    reference_kind: Literal["synthetic", "human_consensus"]
    split: Literal["dev", "test"]
    provider: Literal["synthetic", "gemini"]
    execution_mode: Literal["evaluation"]
    selected_case_ids: list[str] = Field(min_length=1)
    model_requested: str
    sdk_version: str
    source_sha256: Digest
    prompt_bundle_sha256: Digest
    inference_config: dict
    runtime_dependencies: dict[str, str]

    @model_validator(mode="after")
    def validate_selection(self):
        if len(self.selected_case_ids) != len(set(self.selected_case_ids)):
            raise ValueError("Metadata contains duplicate selected case IDs")
        if self.synthetic != (self.provider == "synthetic") or self.synthetic != (
            self.reference_kind == "synthetic"
        ):
            raise ValueError("Provider, synthetic flag and reference kind disagree")
        return self


class EvidenceError(StrictModel):
    type: str
    code: str | None = None


class CaseEvidence(StrictModel):
    case_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    subject_id: str = Field(min_length=1)
    feature: Literal["color", "body"]
    split: Literal["dev", "test"]
    reference: ReferenceLabels
    input_image_sha256: Digest | None
    status: Literal["success", "error"]
    final_result: dict | None
    error: EvidenceError | None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    trace: AnalysisTrace
    run_id: str
    synthetic: bool

    @model_validator(mode="after")
    def validate_outcome(self):
        validate_reference_labels(self.feature, self.reference)
        if self.status == "success":
            if self.final_result is None or self.error is not None:
                raise ValueError("Successful cases require a result and no error")
            schema = ColorApiResponse if self.feature == "color" else BodyApiResponse
            result = schema.model_validate(self.final_result)
            if result.analysis_source != "live":
                raise ValueError("Demo results cannot enter evaluation comparisons")
        elif self.final_result is not None or self.error is None:
            raise ValueError("Failed cases require an error and no final result")
        if self.input_image_sha256 is None and self.status == "success":
            raise ValueError("Successful cases need an input image hash")
        for call in self.trace.calls:
            for attempt in call.attempts:
                if not math.isfinite(attempt.latency_ms) or attempt.latency_ms < 0:
                    raise ValueError("Attempt latency must be finite and non-negative")
                if any(
                    value is not None and value < 0 for value in attempt.usage.values()
                ):
                    raise ValueError("Token usage cannot be negative")
        return self


class AnalysisRun(BaseModel):
    metadata: RunMetadata
    cases: dict[str, CaseEvidence]


def load_analysis_run(directory: Path) -> AnalysisRun:
    raw_metadata = json.loads((directory / "run.json").read_text(encoding="utf-8-sig"))
    if "artifact_schema_version" not in raw_metadata:
        raise ValueError(
            "Unversioned run artifacts: rerun evaluation with the current runner"
        )
    metadata = RunMetadata.model_validate(raw_metadata)
    cases = {}
    for line_number, line in enumerate(
        (directory / "cases.jsonl").read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            raise ValueError(f"Blank evidence row at line {line_number}")
        case = CaseEvidence.model_validate(json.loads(line))
        if case.case_id in cases:
            raise ValueError(f"Duplicate evidence case ID: {case.case_id}")
        if (
            case.run_id != metadata.run_id
            or case.synthetic != metadata.synthetic
            or case.split != metadata.split
        ):
            raise ValueError(
                f"Case identity disagrees with run metadata: {case.case_id}"
            )
        cases[case.case_id] = case
    if set(cases) != set(metadata.selected_case_ids):
        raise ValueError(
            "Incomplete/extra case evidence; every selected case must be present"
        )
    return AnalysisRun(metadata=metadata, cases=cases)


def validate_pair(baseline: AnalysisRun, candidate: AnalysisRun) -> list[str]:
    """Reject changed inputs; disclose missing input proof instead of dropping errors."""
    identity_fields = (
        "dataset_id",
        "dataset_version",
        "manifest_sha256",
        "split",
        "synthetic",
        "reference_kind",
        "provider",
        "execution_mode",
    )
    for field in identity_fields:
        if getattr(baseline.metadata, field) != getattr(candidate.metadata, field):
            raise ValueError(f"Incompatible runs: {field} differs")
    if set(baseline.cases) != set(candidate.cases):
        raise ValueError("Incompatible case sets")
    warnings = []
    for case_id, before in baseline.cases.items():
        after = candidate.cases[case_id]
        for field in ("subject_id", "feature", "reference"):
            if getattr(before, field) != getattr(after, field):
                raise ValueError(f"Changed {field} for case {case_id}")
        if before.input_image_sha256 and after.input_image_sha256:
            if before.input_image_sha256 != after.input_image_sha256:
                raise ValueError(f"Changed input image for case {case_id}")
        else:
            warnings.append(
                f"Input bytes could not be verified for failed case {case_id}"
            )
        if before.trace.processed_image_sha256 != after.trace.processed_image_sha256:
            warnings.append(
                f"Processed image differs or is unavailable for case {case_id}"
            )
    return warnings
