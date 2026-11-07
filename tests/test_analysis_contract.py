"""Ghi nhận hành vi API trước khi tách workflow khỏi endpoint."""

import asyncio
from copy import deepcopy
from io import BytesIO

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.endpoints import analyze
from app.db.database import Base, get_db
from app.main import app
from app.services.analysis.errors import GeminiOverloadedError
from app.services.analysis.photo_types import PhotoQualityReport, ProcessedImage
from app.services.demo_cache import get_demo_body_result
from app.services.gemini_client import GeminiSafetyBlockedError

DEVICE_HEADERS = {"X-Device-ID": "a" * 64}
COLOR = {
    "sub_season": "Soft Autumn",
    "confidence": 0.84,
    "secondary_season": "True Autumn",
    "explanation": "Da ấm và tương phản mềm.",
    "advice": "Ưu tiên màu olive.",
}
QUALITY = {"photo_quality_score": 0.9, "photo_quality_band": "excellent"}


class ScriptedGemini:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    async def analyze(self, **kwargs):
        self.calls.append(kwargs)
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return deepcopy(result)


@pytest.fixture
def api_client(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)

    def session_override():
        with sessions() as session:
            yield session

    monkeypatch.setattr("app.main.init_db", lambda: None)
    app.dependency_overrides[get_db] = session_override
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def configure_ai(monkeypatch, results):
    fake = ScriptedGemini(results)
    app.dependency_overrides[analyze.get_gemini_client] = lambda: fake
    monkeypatch.setattr(
        "app.services.analysis.workflows.preprocess_image_for_ai",
        lambda *args, **kwargs: ProcessedImage(b"jpeg", PhotoQualityReport(**QUALITY)),
    )
    return fake


def upload(client, feature):
    return client.post(
        f"/api/v1/analyze/{feature}",
        headers=DEVICE_HEADERS,
        files={"file": ("photo.jpg", b"photo", "image/jpeg")},
    )


@pytest.mark.parametrize("feature", ["color", "body"])
@pytest.mark.parametrize(
    "failure", [TimeoutError(), GeminiOverloadedError("rate limit")]
)
def test_overload_uses_demo_and_persists_source(
    api_client, monkeypatch, feature, failure
):
    fake = configure_ai(monkeypatch, [failure])
    response = upload(api_client, feature)
    assert response.status_code == 200
    assert response.json()["data"]["analysis_source"] == "degraded_demo"
    history = api_client.get("/api/v1/analyze/history", headers=DEVICE_HEADERS).json()
    assert history["data"][feature]["analysis_source"] == "degraded_demo"
    assert len(fake.calls) == 1


@pytest.mark.parametrize("feature", ["color", "body"])
def test_unreadable_photo_keeps_422_error_contract(api_client, feature):
    response = upload(api_client, feature)
    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "PHOTO_UNREADABLE"


@pytest.mark.parametrize("feature", ["color", "body"])
def test_structured_rejection_keeps_200_failure_envelope(
    api_client, monkeypatch, feature
):
    configure_ai(
        monkeypatch,
        [{"error_code": "NO_PERSON_DETECTED", "message": "Không có người."}],
    )
    response = upload(api_client, feature)
    assert response.status_code == 200
    assert response.json() == {
        "success": False,
        "data": None,
        "error": "Không có người.",
        "error_code": "NO_PERSON_DETECTED",
    }


def test_color_repairs_missing_text_once(api_client, monkeypatch):
    fake = configure_ai(monkeypatch, [{"sub_season": "Soft Autumn"}, COLOR])
    response = upload(api_client, "color")
    assert response.status_code == 200
    assert response.json()["data"]["advice"] == COLOR["advice"]
    assert len(fake.calls) == 2
    assert "bổ sung" in fake.calls[1]["prompt"]


def test_body_repairs_missing_details_once(api_client, monkeypatch):
    fake = configure_ai(monkeypatch, [{"shape": "Rectangle"}, get_demo_body_result()])
    response = upload(api_client, "body")
    assert response.json()["data"]["analysis_source"] == "live"
    assert len(fake.calls) == 2


def test_body_wide_clothing_retry_caps_confidence(api_client, monkeypatch):
    fake = configure_ai(
        monkeypatch,
        [
            {"error_code": "CLOTHING_TOO_WIDE", "message": "Đồ che dáng."},
            get_demo_body_result(),
        ],
    )
    response = upload(api_client, "body")
    assert response.json()["data"]["confidence"] == 0.45
    assert len(fake.calls) == 2


def test_incomplete_body_is_not_saved(api_client, monkeypatch):
    configure_ai(monkeypatch, [{"shape": "Rectangle"}, {"shape": "Rectangle"}])
    response = upload(api_client, "body")
    assert response.status_code == 200
    assert response.json()["error_code"] == "AI_INCOMPLETE"
    history = api_client.get("/api/v1/analyze/history", headers=DEVICE_HEADERS).json()
    assert history["data"]["body"] is None


@pytest.mark.parametrize("feature", ["color", "body"])
def test_safety_block_preserves_incomplete_response_contract(
    api_client, monkeypatch, feature
):
    fake = configure_ai(monkeypatch, [GeminiSafetyBlockedError("blocked")])
    response = upload(api_client, feature)
    assert response.status_code == 503
    assert response.json()["detail"]["error_code"] == "AI_INCOMPLETE"
    assert len(fake.calls) == 1
    history = api_client.get("/api/v1/analyze/history", headers=DEVICE_HEADERS).json()
    assert history["data"][feature] is None


def test_recommendation_requires_both_analyses(api_client):
    response = api_client.get("/api/v1/outfits/recommend", headers=DEVICE_HEADERS)
    assert response.status_code == 400
    assert response.json()["detail"]["missing"] == ["color", "body"]


def test_tiny_photo_preserves_the_resolution_error(api_client):
    image = BytesIO()
    Image.new("RGB", (100, 100)).save(image, format="PNG")
    response = api_client.post(
        "/api/v1/analyze/color",
        headers=DEVICE_HEADERS,
        files={"file": ("small.png", image.getvalue(), "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "PHOTO_TOO_SMALL"


@pytest.mark.parametrize("feature", ["color", "body"])
@pytest.mark.parametrize(
    "failure,code",
    [
        (ValueError("read failed"), "AI_INVALID_RESULT"),
        (OSError("read failed"), "ANALYSIS_FAILED"),
    ],
)
def test_upload_read_failure_keeps_the_existing_error_contract(feature, failure, code):
    class BrokenUpload:
        async def read(self):
            raise failure

    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            getattr(analyze, "analyze_" + feature)(
                file=BrokenUpload(),
                x_device_id="a" * 64,
                db=None,
                client=None,
            )
        )
    assert caught.value.status_code == 503
    assert caught.value.detail["error_code"] == code
