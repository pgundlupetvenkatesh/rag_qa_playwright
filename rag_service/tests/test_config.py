from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import SERVICE_ROOT, Settings


def test_defaults_apply_when_unset(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings()

    assert settings.groq_api_key is None
    assert settings.groq_model == "openai/gpt-oss-20b"
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.chroma_path == SERVICE_ROOT / ".chroma"
    assert (settings.chunk_size, settings.chunk_overlap) == (1000, 200)
    assert settings.top_k == 4
    assert (settings.api_host, settings.api_port) == ("127.0.0.1", 8000)
    assert settings.request_timeout_seconds == 30.0


def test_environment_variables_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3-32b")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("API_PORT", "9001")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.groq_model == "qwen/qwen3-32b"
    assert settings.top_k == 7
    assert settings.api_port == 9001


def test_blank_api_key_counts_as_unset(make_settings: Callable[..., Settings]) -> None:
    assert make_settings(groq_api_key="  ").groq_api_key is None


def test_api_key_is_never_rendered(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(groq_api_key="secret-value")

    assert "secret-value" not in repr(settings)
    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "secret-value"


def test_relative_chroma_path_resolves_against_service_root(
    make_settings: Callable[..., Settings],
) -> None:
    assert make_settings(chroma_path="data/db").chroma_path == SERVICE_ROOT / "data" / "db"
    assert make_settings(chroma_path="/tmp/db").chroma_path == Path("/tmp/db")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"embedding_model": "bge-small"}, "unsupported embedding model"),
        ({"chunk_size": 100, "chunk_overlap": 100}, "must be smaller than CHUNK_SIZE"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"chunk_overlap": -1}, "chunk_overlap"),
        ({"top_k": 0}, "top_k"),
        ({"api_port": 70000}, "api_port"),
        ({"request_timeout_seconds": 0}, "request_timeout_seconds"),
        ({"groq_model": ""}, "groq_model"),
    ],
)
def test_invalid_values_are_rejected(
    make_settings: Callable[..., Settings], overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        make_settings(**overrides)