import hashlib
import math
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.corpus import Corpus
from app.main import app, get_corpus

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
    "ALLOW_RESET",
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


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: offline, instant, no model download.

    Texts sharing words get similar vectors, which is enough to exercise
    storage and ranking. ``calls`` records every batch, to prove what was
    (and was not) re-embedded.
    """

    model_name = "fake-embedder"
    dimensions = 64

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            vector[0] = 1.0  # never all-zero, so cosine distance is defined
            for word in text.lower().split():
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % (self.dimensions - 1)
                vector[bucket + 1] += 1.0
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector])
        return vectors


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def make_corpus(tmp_path: Path, embedder: FakeEmbedder) -> Callable[..., Corpus]:
    """Open a corpus in a temp dir with the fake embedder. Same args -> same store."""

    def build(chunk_size: int = 1000, chunk_overlap: int = 200, embedder_: Any = None) -> Corpus:
        return Corpus(
            path=tmp_path / "chroma",
            embedder=embedder_ or embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    return build


@pytest.fixture
def client(make_settings: Callable[..., Settings], make_corpus: Callable[..., Corpus]) -> TestClient:
    """API client over a fake-embedder corpus. Reset is allowed for loopback callers only
    (the default), and the client connects from 127.0.0.1."""
    settings = make_settings()
    corpus = make_corpus()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_corpus] = lambda: corpus
    return TestClient(app, client=("127.0.0.1", 50000))