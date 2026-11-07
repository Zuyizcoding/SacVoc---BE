"""Opt-in evidence for evaluation; normal API requests retain nothing.

Pipeline: workflow stage -> logical model call -> one or more HTTP attempts.
A repair is a new call; a retry repeats the same call. Keeping these levels
separate prevents a successful repair from hiding an invalid first response.
ContextVar isolates concurrent analyses without changing analyze()'s interface.
No image bytes, device identifiers, or credentials are stored here.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from typing import Iterator

from pydantic import BaseModel, Field


def content_hash(content: str | bytes) -> str:
    data = content.encode("utf-8") if isinstance(content, str) else content
    return sha256(data).hexdigest()


class AttemptTrace(BaseModel):
    """One provider request, including failures before a valid result exists."""

    number: int
    latency_ms: float = 0
    raw_text: str | None = None
    schema_valid: bool = False
    error_type: str | None = None
    http_status: int | None = None
    finish_reason: str | None = None
    prompt_block_reason: str | None = None
    model_version: str | None = None
    usage: dict[str, int | None] = Field(default_factory=dict)
    retry_delay_seconds: float | None = None


class ModelCallTrace(BaseModel):
    """The exact input contract and normalized output of one logical call."""

    stage: str
    prompt: str
    system_instruction: str
    prompt_sha256: str
    schema_name: str
    schema_sha256: str
    attempts: list[AttemptTrace] = Field(default_factory=list)
    validated_result: dict | None = None


class AnalysisTrace(BaseModel):
    photo_quality: dict = Field(default_factory=dict)
    processed_image_sha256: str | None = None
    calls: list[ModelCallTrace] = Field(default_factory=list)


_active_trace: ContextVar[AnalysisTrace | None] = ContextVar(
    "analysis_trace", default=None
)
_active_stage: ContextVar[str] = ContextVar("analysis_stage", default="initial")


def current_trace() -> AnalysisTrace | None:
    return _active_trace.get()


def current_stage() -> str:
    return _active_stage.get()


@contextmanager
def capture_analysis_trace() -> Iterator[AnalysisTrace]:
    trace = AnalysisTrace()
    token = _active_trace.set(trace)
    try:
        yield trace
    finally:
        _active_trace.reset(token)


@contextmanager
def trace_stage(stage: str) -> Iterator[None]:
    token = _active_stage.set(stage)
    try:
        yield
    finally:
        _active_stage.reset(token)
