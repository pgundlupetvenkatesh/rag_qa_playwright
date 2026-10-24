from collections.abc import Callable

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app


def test_health_reports_status_and_models(make_settings: Callable[..., Settings]) -> None:
    app.dependency_overrides[get_settings] = lambda: make_settings(groq_model="test/model")

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "embeddingModel": "all-MiniLM-L6-v2",
        "generationModel": "test/model",
    }