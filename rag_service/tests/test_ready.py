from collections.abc import Callable
from pathlib import Path
from typing import Any

import groq
import httpx
import pytest
from fastapi.testclient import TestClient

from app import readiness
from app.config import Settings, get_settings
from app.main import app
from app.readiness import get_model_verifier, verify_groq_model
from app.schemas import ReadinessCheck


def model_ok(_: Settings) -> ReadinessCheck:
    return ReadinessCheck(name="groqModel", ok=True, detail="stub")


def model_missing(_: Settings) -> ReadinessCheck:
    return ReadinessCheck(name="groqModel", ok=False, detail="stub: not available")


def get_ready(settings: Settings, verifier: Callable[[Settings], ReadinessCheck]) -> httpx.Response:
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_model_verifier] = lambda: verifier
    return TestClient(app).get("/ready")


def checks_by_name(response: httpx.Response) -> dict[str, dict[str, Any]]:
    return {check["name"]: check for check in response.json()["checks"]}


def test_ready_when_every_check_passes(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = make_settings(groq_api_key="k", chroma_path=tmp_path / "db")

    response = get_ready(settings, model_ok)

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert list(checks_by_name(response)) == ["groqApiKey", "groqModel", "chromaPath"]
    assert all(check["ok"] for check in response.json()["checks"])


def test_missing_key_is_not_ready_and_skips_the_model_call(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    # The real verifier: with no key it must return without touching the network.
    response = get_ready(make_settings(chroma_path=tmp_path), verify_groq_model)

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    checks = checks_by_name(response)
    assert checks["groqApiKey"]["ok"] is False
    assert checks["groqModel"] == {
        "name": "groqModel",
        "ok": False,
        "detail": "skipped: GROQ_API_KEY is not set",
    }


def test_unavailable_model_is_not_ready(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    response = get_ready(make_settings(groq_api_key="k", chroma_path=tmp_path), model_missing)

    assert response.status_code == 503
    assert checks_by_name(response)["groqModel"]["ok"] is False


def test_chroma_path_that_is_a_file_is_not_ready(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    blocker = tmp_path / "db"
    blocker.write_text("not a directory")

    response = get_ready(make_settings(groq_api_key="k", chroma_path=blocker), model_ok)

    assert response.status_code == 503
    assert checks_by_name(response)["chromaPath"]["detail"].endswith("is not a directory")


def test_ready_never_echoes_the_api_key(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = make_settings(groq_api_key="secret-value", chroma_path=tmp_path)

    assert "secret-value" not in get_ready(settings, model_ok).text


def model_entry(model_id: str) -> dict[str, Any]:
    return {"id": model_id, "object": "model", "created": 0, "owned_by": "x"}


@pytest.mark.parametrize(
    ("status_code", "listed", "ok", "detail"),
    [
        (200, ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"], True, "model 'openai/gpt-oss-120b' is available"),
        (200, ["qwen/qwen3.8-27b"], False, "model 'openai/gpt-oss-120b' is not available to this API key"),
        (401, [], False, "Groq rejected GROQ_API_KEY"),
        (500, [], False, "Groq returned HTTP 500"),
    ],
)
def test_verify_groq_model_maps_groq_responses(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: Callable[..., Settings],
    status_code: int,
    listed: list[str],
    ok: bool,
    detail: str,
) -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        body = {"object": "list", "data": [model_entry(model_id) for model_id in listed]}
        return httpx.Response(status_code, json=body if status_code == 200 else {"error": {}})

    real_groq = groq.Groq

    def offline_groq(**kwargs: Any) -> groq.Groq:
        return real_groq(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(readiness.groq, "Groq", offline_groq)

    check = verify_groq_model(make_settings(groq_api_key="k"))

    assert (check.ok, check.detail) == (ok, detail)
    assert requested == ["/openai/v1/models"]