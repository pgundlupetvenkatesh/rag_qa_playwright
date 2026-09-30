from collections.abc import Callable, Iterator
from typing import Any

import pytest

from app.config import Settings
from app.main import app

ENV_VARS = (
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "EMBEDDING_MODEL",
    "CHROMA_PATH",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "API_HOST",
    "API_PORT",
    "REQUEST_TIMEOUT_SECONDS",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the developer's shell and rag_service/.env out of every test."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def make_settings() -> Callable[..., Settings]:
    """Build Settings from explicit values only, ignoring any .env file."""

    def build(**overrides: Any) -> Settings:
        return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]

    return build