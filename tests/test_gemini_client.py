import asyncio

import pytest

from app.schemas.color import ColorGeminiResponse
from app.services.gemini_client import GeminiClient, GeminiInvalidResponseError


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModel:
    async def generate_content(self, *_args, **_kwargs):
        return _FakeResponse("{}")


def test_invalid_structured_response_does_not_open_circuit_breaker(monkeypatch) -> None:
    client = GeminiClient()
    monkeypatch.setattr(client, "_generate_once", _FakeModel().generate_content)

    with pytest.raises(GeminiInvalidResponseError):
        asyncio.run(
            client.analyze(
                image_bytes=b"fake-image",
                prompt="prompt",
                response_schema=ColorGeminiResponse,
                system_instruction="system",
            )
        )

    assert client._breaker.is_open is False
