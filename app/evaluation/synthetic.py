"""An explicitly fake provider for offline pipeline checks.

MockTransport intercepts the SDK's HTTP boundary. We therefore exercise real SDK
serialization, response parsing, application validation, retries, and repairs
without making network requests. Script exhaustion is an error, not fake success.
The generated patterned image contains no person and has no scientific labels.
"""

import json
from io import BytesIO

import httpx
from google import genai
from google.genai import types
from PIL import Image

from .schemas import SyntheticReply


def generated_image() -> bytes:
    image = Image.new("RGB", (480, 640), (140, 130, 120))
    pixels = image.load()
    for y in range(640):
        for x in range(480):
            shade = (x * 17 + y * 31) % 80
            pixels[x, y] = (100 + shade, 95 + shade, 90 + shade)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def scripted_sdk(
    replies: list[SyntheticReply], *, timeout_seconds: int = 30
) -> genai.Client:
    remaining = iter(replies)

    def respond(request: httpx.Request) -> httpx.Response:
        try:
            reply = next(remaining)
        except StopIteration:
            raise RuntimeError("Synthetic response script exhausted") from None
        if reply.http_status != 200:
            return httpx.Response(
                reply.http_status,
                json={
                    "error": {"code": reply.http_status, "message": "Synthetic failure"}
                },
            )
        text = (
            reply.raw_text
            if reply.raw_text is not None
            else json.dumps(reply.payload or {})
        )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": text}]},
                        "finishReason": reply.finish_reason,
                    }
                ],
                "modelVersion": "synthetic-not-a-real-model",
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            },
        )

    return genai.Client(
        vertexai=False,
        api_key="synthetic-unused-key",
        http_options=types.HttpOptions(
            api_version="v1beta",
            timeout=timeout_seconds * 1000,
            retry_options=types.HttpRetryOptions(attempts=1),
            client_args={"transport": httpx.MockTransport(respond)},
        ),
    )
