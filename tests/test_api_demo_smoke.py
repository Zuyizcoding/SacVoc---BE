from collections.abc import Generator

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.endpoints.analyze import get_gemini_client
from app.db.database import Base, get_db
from app.main import app
from app.schemas.outfit import OutfitItem, OutfitRecommendResponse


TEST_DEVICE_ID = "a" * 64
QUALITY_FLAGS = {
    "soft_far_subject_for_color": False,
    "soft_low_resolution": False,
    "soft_low_detail": False,
    "soft_bad_lighting": False,
    "soft_color_cast": False,
    "photo_quality_score": 0.74,
    "photo_quality_band": "good",
}


class _FakeGeminiClient:
    def __init__(self, *, overload_body: bool = False):
        self.overload_body = overload_body

    async def analyze(
        self,
        *,
        image_bytes: bytes,
        prompt: str,
        response_schema,
        system_instruction: str | None = None,
    ) -> dict:
        schema_name = getattr(response_schema, "__name__", "")
        if schema_name == "ColorInferenceResponse":
            return {
                "sub_season": "Soft Autumn",
                "confidence": 0.84,
                "secondary_season": "True Autumn",
                "explanation": "Tông da ấm và độ tương phản mềm.",
                "advice": "Ưu tiên bảng màu ấm và trầm.",
            }
        if self.overload_body:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "AI_OVERLOADED",
                    "message": "Gemini đang quá tải",
                },
            )
        return {
            "shape": "Rectangle",
            "confidence": 0.79,
            "measurements": {
                "shoulder_relative": "balanced",
                "hip_relative": "balanced",
                "waist_definition": "minimal",
            },
            "styling_tips": [
                "Ưu tiên layer tạo khối vai.",
                "Chọn fit regular để cân bằng tổng thể.",
                "Dùng đường dọc để kéo thân người.",
            ],
            "avoid": [
                "Đồ quá ôm làm lộ thân hình thẳng.",
                "Áo quá rộng che mất tỷ lệ vai.",
            ],
            "explanation": "Vai và hông cân bằng, eo ít rõ.",
        }


def _override_db(session_factory) -> Generator[Session, None, None]:
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def _make_client(monkeypatch, fake_client: _FakeGeminiClient) -> TestClient:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db() -> Generator[Session, None, None]:
        yield from _override_db(testing_session_local)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_gemini_client] = lambda: fake_client

    monkeypatch.setattr(
        "app.api.endpoints.analyze.preprocess_image_for_ai",
        lambda file_bytes, max_kb=1000: (file_bytes, QUALITY_FLAGS.copy()),
    )

    return TestClient(app)


def test_api_golden_path_color_body_history_and_outfit(monkeypatch) -> None:
    fake_client = _FakeGeminiClient()
    monkeypatch.setattr(
        "app.api.endpoints.outfit.recommend_outfits",
        lambda **kwargs: OutfitRecommendResponse(
            query_used="soft autumn rectangle daily menswear",
            query_source="precomputed_query_vector",
            outfits=[
                OutfitItem(
                    id=1,
                    name="Ao khoac daily",
                    category="outerwear",
                    image_url="https://example.com/look.jpg",
                    shopee_url="https://example.com/buy",
                    price_range="300k-500k",
                    score=0.91,
                    retrieval_score=0.88,
                    reason_tags=["primary_season", "fit_match", "formality_match"],
                )
            ],
        ),
    )

    with _make_client(monkeypatch, fake_client) as client:
        color_response = client.post(
            "/api/v1/analyze/color",
            headers={"X-Device-ID": TEST_DEVICE_ID},
            files={"file": ("selfie.jpg", b"fake-image", "image/jpeg")},
        )
        assert color_response.status_code == 200
        assert color_response.json()["data"]["analysis_source"] == "live"

        body_response = client.post(
            "/api/v1/analyze/body",
            headers={"X-Device-ID": TEST_DEVICE_ID},
            files={"file": ("body.jpg", b"fake-image", "image/jpeg")},
        )
        assert body_response.status_code == 200
        assert body_response.json()["data"]["analysis_source"] == "live"

        history_response = client.get(
            "/api/v1/analyze/history",
            headers={"X-Device-ID": TEST_DEVICE_ID},
        )
        history_payload = history_response.json()["data"]
        assert history_response.status_code == 200
        assert history_payload["color"]["sub_season"] == "Soft Autumn"
        assert history_payload["body"]["shape"] == "Rectangle"

        outfit_response = client.get(
            "/api/v1/outfits/recommend",
            headers={"X-Device-ID": TEST_DEVICE_ID},
        )
        outfit_payload = outfit_response.json()["data"]
        assert outfit_response.status_code == 200
        assert outfit_payload["query_used"] == "soft autumn rectangle daily menswear"
        assert outfit_payload["outfits"][0]["reason_tags"] == [
            "primary_season",
            "fit_match",
            "formality_match",
        ]

    app.dependency_overrides.clear()


def test_api_body_fallback_marks_degraded_demo_source(monkeypatch) -> None:
    fake_client = _FakeGeminiClient(overload_body=True)

    with _make_client(monkeypatch, fake_client) as client:
        response = client.post(
            "/api/v1/analyze/body",
            headers={"X-Device-ID": TEST_DEVICE_ID},
            files={"file": ("body.jpg", b"fake-image", "image/jpeg")},
        )
        payload = response.json()["data"]
        assert response.status_code == 200
        assert payload["analysis_source"] == "degraded_demo"

        history_response = client.get(
            "/api/v1/analyze/history",
            headers={"X-Device-ID": TEST_DEVICE_ID},
        )
        history_payload = history_response.json()["data"]
        assert history_payload["body"]["analysis_source"] == "degraded_demo"

    app.dependency_overrides.clear()
