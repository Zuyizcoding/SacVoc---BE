"""Translate application schemas into Google GenAI request configuration.

Intuition: JSON schema describes the shape of a response, not whether the model's
conclusion is correct. Request structured JSON, then independently validate it
with Pydantic. Business completeness remains the workflow's responsibility.
"""

import json
import re

from google.genai import types
from pydantic import BaseModel

from app.core.config import settings


def build_generation_config(
    response_schema: type[BaseModel], system_instruction: str
) -> types.GenerateContentConfig:
    """Use Pydantic support instead of silently stripping schema constraints.

    The new SDK supports nested objects and nullable fields. Configuration errors
    surface rather than silently weakening the contract. Safety thresholds retain
    the application's existing image-analysis policy.
    """
    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=response_schema,
        thinking_config=types.ThinkingConfig(
            thinking_budget=settings.GEMINI_THINKING_BUDGET
        ),
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"
            ),
        ],
    )


def parse_gemini_json(text: str) -> dict:
    """Retain fenced/prose JSON compatibility; reject non-object JSON values.

    Structured output still needs defensive parsing: empty, truncated, or blocked
    responses can occur. Pydantic subsequently checks field types and validators.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```json\s*([\s\S]*?)```", text)
        if match is None:
            match = re.search(r"\{[\s\S]*\}", text)
        if match is None:
            raise ValueError("Gemini response does not contain JSON") from None
        payload = json.loads(match.group(1) if match.lastindex else match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Gemini response must be a JSON object")
    return payload
