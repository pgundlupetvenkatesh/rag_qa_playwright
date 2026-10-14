from collections.abc import Callable, Sequence

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.corpus import Corpus, RetrievedChunk
from app.generation import Generation, GenerationError
from app.main import app, get_corpus, get_generator
from tests.conftest import FakeEmbedder

DOCUMENTS = [
    {"id": "normandy", "text": "Normandy is a region in the north of France.", "metadata": {"title": "Normans"}},
    {"id": "piano", "text": "The piano has eighty eight keys."},
]


class FakeGenerator:
    """Records what it was shown and returns a canned result or raises."""

    def __init__(self, result: Generation | GenerationError) -> None:
        self.result = result
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def generate(self, question: str, evidence: Sequence[RetrievedChunk]) -> Generation:
        self.calls.append((question, list(evidence)))
        if isinstance(self.result, GenerationError):
            raise self.result
        return self.result


ANSWER = Generation(
    answer="In the north of France.",
    abstained=False,
    model="openai/gpt-oss-120b",
    prompt_tokens=100,
    completion_tokens=10,
    total_tokens=110,
    provider_ms=42.0,
)


def use(generator: FakeGenerator) -> FakeGenerator:
    app.dependency_overrides[get_generator] = lambda: generator
    return generator


@pytest.fixture
def seeded(client: TestClient) -> TestClient:
    assert client.post("/corpus/documents", json={"documents": DOCUMENTS}).status_code == 200
    return client


def test_query_returns_answer_evidence_and_metadata(seeded: TestClient) -> None:
    generator = use(FakeGenerator(ANSWER))

    response = seeded.post("/query", json={"question": "Where is Normandy?", "topK": 2})

    assert response.status_code == 200
    body = response.json()
    # Exact key sets: an unexpected field (a leaked label, a prompt) fails here.
    assert set(body) == {"question", "answer", "abstained", "model", "evidence", "usage", "timings"}
    assert set(body["timings"]) == {"retrievalMs", "generationMs", "providerMs"}
    assert set(body["evidence"][0]) == {
        "chunkId", "documentId", "chunkIndex", "text", "title", "source", "distance", "similarity"
    }
    assert body["question"] == "Where is Normandy?"
    assert body["answer"] == "In the north of France."
    assert body["abstained"] is False
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["usage"] == {"promptTokens": 100, "completionTokens": 10, "totalTokens": 110}
    assert body["timings"]["providerMs"] == 42.0
    assert body["timings"]["retrievalMs"] >= 0
    assert body["timings"]["generationMs"] >= 0
    assert [chunk["documentId"] for chunk in body["evidence"]][0] == "normandy"

    # The evidence returned is exactly the evidence the model was shown.
    [(question, shown)] = generator.calls
    assert question == "Where is Normandy?"
    assert [chunk.chunk_id for chunk in shown] == [chunk["chunkId"] for chunk in body["evidence"]]


def test_query_reports_abstention_without_an_answer(seeded: TestClient) -> None:
    use(FakeGenerator(Generation(answer=None, abstained=True, model="qwen/qwen3.8-27b")))

    body = seeded.post("/query", json={"question": "Who won the 2030 World Cup?"}).json()

    assert body["abstained"] is True
    assert body["answer"] is None
    assert body["model"] == "qwen/qwen3.8-27b"
    assert body["usage"] == {"promptTokens": None, "completionTokens": None, "totalTokens": None}
    assert body["evidence"]  # retrieval ran; the model judged it insufficient


def test_query_uses_configured_top_k(seeded: TestClient, make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(top_k=1)
    app.dependency_overrides[get_settings] = lambda: settings
    generator = use(FakeGenerator(ANSWER))

    seeded.post("/query", json={"question": "piano"})

    assert len(generator.calls[0][1]) == 1


def test_empty_corpus_fails_in_retrieval_without_calling_the_model(client: TestClient) -> None:
    generator = use(FakeGenerator(ANSWER))

    response = client.post("/query", json={"question": "Where is Normandy?"})

    assert response.status_code == 409
    assert response.json()["detail"]["stage"] == "retrieval"
    assert response.json()["detail"]["code"] == "corpus_empty"
    assert generator.calls == []


def test_no_evidence_fails_in_retrieval_without_calling_the_model(
    seeded: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus: Corpus = app.dependency_overrides[get_corpus]()
    monkeypatch.setattr(corpus, "search", lambda question, top_k: [])
    generator = use(FakeGenerator(ANSWER))

    response = seeded.post("/query", json={"question": "Where is Normandy?"})

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "no_results"
    assert generator.calls == []  # an answer without evidence would be ungrounded


def test_missing_credentials_fail_before_retrieval(seeded: TestClient, embedder: FakeEmbedder) -> None:
    calls_before = len(embedder.calls)

    response = seeded.post("/query", json={"question": "Where is Normandy?"})

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "stage": "generation",
        "code": "missing_credentials",
        "message": "GROQ_API_KEY is not set",
    }
    assert len(embedder.calls) == calls_before  # the question was never embedded


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (GenerationError("provider_timeout", "Groq did not answer in time", 504), 504),
        (GenerationError("malformed_response", "Groq returned no choices", 502), 502),
        (GenerationError("model_unavailable", "model 'x' is not available to this API key", 503), 503),
    ],
)
def test_generation_failures_are_structured(seeded: TestClient, error: GenerationError, status: int) -> None:
    use(FakeGenerator(error))

    response = seeded.post("/query", json={"question": "Where is Normandy?"})

    assert response.status_code == status
    assert response.json() == {"detail": {"stage": "generation", "code": error.code, "message": error.message}}
    assert "retry-after" not in response.headers


def test_rate_limit_sets_retry_after(seeded: TestClient) -> None:
    use(FakeGenerator(GenerationError("rate_limited", "Groq rate limit reached; retry later", 429, "12")))

    response = seeded.post("/query", json={"question": "Where is Normandy?"})

    assert response.status_code == 429
    assert response.headers["retry-after"] == "12"
    assert response.json()["detail"]["code"] == "rate_limited"


def test_query_rejects_invalid_requests(client: TestClient) -> None:
    use(FakeGenerator(ANSWER))

    assert client.post("/query", json={"question": " "}).status_code == 422
    assert client.post("/query", json={"question": "q", "answerable": False}).status_code == 422