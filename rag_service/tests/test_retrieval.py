from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.corpus import Corpus, SourceDocument
from app.main import app, get_corpus, get_generator
from tests.conftest import FakeEmbedder

DOCS = [
    SourceDocument(id="volcano", text="Lava erupts from the volcano crater.", title="Volcanoes", source="test"),
    SourceDocument(id="piano", text="The piano has eighty eight keys.", title="Instruments"),
    SourceDocument(id="river", text="The river flows into the sea."),
]


def test_search_ranks_the_matching_document_first(make_corpus: Callable[..., Corpus]) -> None:
    corpus = make_corpus()
    corpus.ingest(DOCS)

    results = corpus.search("how many keys does a piano have", top_k=3)

    assert [chunk.document_id for chunk in results][0] == "piano"
    assert len(results) == 3
    distances = [chunk.distance for chunk in results]
    assert distances == sorted(distances)


def test_search_returns_chunk_fields(make_corpus: Callable[..., Corpus]) -> None:
    corpus = make_corpus()
    corpus.ingest(DOCS)

    [top] = corpus.search("lava volcano crater", top_k=1)

    assert (top.chunk_id, top.document_id, top.chunk_index) == ("volcano#0", "volcano", 0)
    assert top.text == DOCS[0].text
    assert (top.title, top.source) == ("Volcanoes", "test")
    assert 0 <= top.distance < 1


def test_search_caps_top_k_at_corpus_size(make_corpus: Callable[..., Corpus]) -> None:
    corpus = make_corpus()
    corpus.ingest(DOCS[:1])

    assert len(corpus.search("anything", top_k=10)) == 1


def test_search_on_empty_corpus_returns_nothing(make_corpus: Callable[..., Corpus]) -> None:
    assert make_corpus().search("anything", top_k=4) == []


def test_search_embeds_the_question_with_the_corpus_embedder(
    make_corpus: Callable[..., Corpus], embedder: FakeEmbedder
) -> None:
    corpus = make_corpus()
    corpus.ingest(DOCS)

    corpus.search("river sea", top_k=1)

    assert embedder.calls[-1] == ["river sea"]


@pytest.fixture
def seeded(client: TestClient) -> TestClient:
    documents = [{"id": doc.id, "text": doc.text} for doc in DOCS]
    assert client.post("/corpus/documents", json={"documents": documents}).status_code == 200
    return client


def test_retrieve_returns_camel_case_chunks(seeded: TestClient) -> None:
    response = seeded.post("/retrieve", json={"question": "piano keys", "topK": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "piano keys"
    assert body["topK"] == 2
    assert body["retrievalMs"] >= 0
    assert len(body["chunks"]) == 2
    top = body["chunks"][0]
    assert set(top) == {
        "chunkId", "documentId", "chunkIndex", "text", "title", "source", "distance", "similarity"
    }
    assert top["documentId"] == "piano"
    assert top["similarity"] == pytest.approx(1 - top["distance"])


def test_retrieve_defaults_to_configured_top_k(
    seeded: TestClient, make_settings: Callable[..., Settings]
) -> None:
    settings = make_settings(top_k=1)
    app.dependency_overrides[get_settings] = lambda: settings

    body = seeded.post("/retrieve", json={"question": "river"}).json()

    assert body["topK"] == 1
    assert len(body["chunks"]) == 1


def test_retrieve_never_calls_the_generator(seeded: TestClient) -> None:
    def fail() -> None:
        raise AssertionError("retrieval must not touch the generator")

    app.dependency_overrides[get_generator] = fail

    assert seeded.post("/retrieve", json={"question": "river"}).status_code == 200


def test_retrieve_on_empty_corpus_is_an_explicit_retrieval_error(client: TestClient) -> None:
    response = client.post("/retrieve", json={"question": "anything"})

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "stage": "retrieval",
        "code": "corpus_empty",
        "message": "the corpus is empty; ingest documents first",
    }


def test_retrieve_with_no_results_from_a_non_empty_corpus_is_an_explicit_retrieval_error(
    seeded: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus: Corpus = app.dependency_overrides[get_corpus]()
    monkeypatch.setattr(corpus, "search", lambda question, top_k: [])

    response = seeded.post("/retrieve", json={"question": "anything"})

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "stage": "retrieval",
        "code": "no_results",
        "message": "retrieval returned no chunks",
    }


@pytest.mark.parametrize(
    "body",
    [
        {"question": "   "},
        {"question": ""},
        {"question": "x" * 2001},
        {"question": "ok", "topK": 0},
        {"question": "ok", "topK": 51},
        {"question": "ok", "groundTruth": ["leak"]},
        {},
    ],
)
def test_retrieve_rejects_invalid_requests(client: TestClient, body: dict[str, object]) -> None:
    assert client.post("/retrieve", json=body).status_code == 422