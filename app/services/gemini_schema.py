"""Chuyển schema, parse JSON và cấu hình SDK; tách khỏi chính sách retry."""

import inspect
import json
import logging
import re
from copy import deepcopy
from typing import Type

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)


def _deref_json_schema_refs(schema: object, defs: dict[str, object]) -> object:
    """Resolve internal Pydantic `$ref` that point to `#/$defs/...`.

    google-generativeai expects a simplified schema object and doesn't understand
    `$ref`/`$defs`. We inline referenced definitions to keep the schema self-contained.
    """

    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.split("/")[-1]
            if name in defs:
                return _deref_json_schema_refs(defs[name], defs)

        return {
            k: _deref_json_schema_refs(v, defs)
            for k, v in schema.items()
            if k != "$defs"
        }

    if isinstance(schema, list):
        return [_deref_json_schema_refs(item, defs) for item in schema]

    return schema


def _sanitize_schema_for_genai(
    schema: object, *, _in_properties_map: bool = False
) -> object:
    """Strip JSONSchema keywords unsupported by google-generativeai Schema.

    The SDK's `Schema` proto doesn't accept constraints like `maximum`, `minimum`,
    `pattern`, etc. Keeping only a small subset avoids `Unknown field for Schema`.
    """

    allowed_keys = {
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "description",
        "format",
        "nullable",
    }

    if isinstance(schema, dict):
        # When we're inside a `properties` object, keys are *field names* and must
        # be preserved verbatim.
        if _in_properties_map:
            return {k: _sanitize_schema_for_genai(v) for k, v in schema.items()}

        sanitized: dict = {}
        for k, v in schema.items():
            if k not in allowed_keys:
                continue

            if k == "properties" and isinstance(v, dict):
                sanitized[k] = _sanitize_schema_for_genai(v, _in_properties_map=True)
            else:
                sanitized[k] = _sanitize_schema_for_genai(v)

        # Defensive: if required mentions fields not present in properties,
        # drop required to avoid INVALID_ARGUMENT from the API.
        try:
            req = sanitized.get("required")
            props = sanitized.get("properties")
            if isinstance(req, list) and isinstance(props, dict):
                if any(
                    (not isinstance(name, str)) or (name not in props) for name in req
                ):
                    sanitized.pop("required", None)
        except Exception:
            pass

        return sanitized

    if isinstance(schema, list):
        return [_sanitize_schema_for_genai(item) for item in schema]

    return schema


def _pydantic_model_to_genai_response_schema(model: Type[BaseModel]) -> dict:
    """Convert a Pydantic model into a genai-compatible response_schema dict."""

    raw = deepcopy(model.model_json_schema())
    defs = raw.get("$defs") or {}
    deref = _deref_json_schema_refs(raw, defs)
    sanitized = _sanitize_schema_for_genai(deref)
    assert isinstance(sanitized, dict)
    return sanitized


def parse_gemini_json(text: str) -> dict:
    """Trích xuất JSON từ Gemini response.

    Khi dùng response_schema, Gemini LUÔN trả JSON hợp lệ.
    Hàm này là fallback cưỡng bức cho edge case cực hiếm.
    """
    # Happy path — response_schema đảm bảo JSON hợp lệ
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback 1: strip markdown code block ```json...```
    match = re.search(r"```json\s*([\s\S]*?)```", text)
    if match:
        return json.loads(match.group(1))

    # Fallback 2: tìm {...} bất kỳ
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Gemini không trả về JSON hợp lệ")


def build_gemini_model(
    response_schema: Type[BaseModel], system_instruction: str
) -> genai.GenerativeModel:
    """Tạo model instance với response_schema + safety settings.

    Tính năng #11: response_schema ép JSON — không bao giờ JSONDecodeError.
    Tính năng #12: BLOCK_NONE cho ảnh toàn thân.
    """
    gen_config_kwargs: dict = {"response_mime_type": "application/json"}
    try:
        gen_config_kwargs["response_schema"] = _pydantic_model_to_genai_response_schema(
            response_schema
        )
    except Exception as exc:
        logger.warning(
            "response_schema sanitization failed; continuing without it: %s", str(exc)
        )
    try:
        gen_config_sig = inspect.signature(genai.GenerationConfig)
        supports_thinking_config = "thinking_config" in gen_config_sig.parameters
    except Exception:
        supports_thinking_config = False
    if supports_thinking_config and hasattr(
        getattr(genai, "types", None), "ThinkingConfig"
    ):
        try:
            gen_config_kwargs["thinking_config"] = genai.types.ThinkingConfig(
                thinking_budget=settings.GEMINI_THINKING_BUDGET
            )
        except Exception as exc:
            logger.warning("ThinkingConfig unsupported by current SDK: %s", str(exc))
    try:
        generation_config = genai.GenerationConfig(**gen_config_kwargs)
    except Exception as exc:
        logger.warning(
            "GenerationConfig rejected response_schema; disabling structured schema: %s",
            str(exc),
        )
        gen_config_kwargs.pop("response_schema", None)
        generation_config = genai.GenerationConfig(**gen_config_kwargs)
    return genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        system_instruction=system_instruction,
        generation_config=generation_config,
        safety_settings={
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
    )
