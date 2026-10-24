from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.corpus import Corpus
from app.main import app, get_corpus
from app.schemas import MAX_DOCUMENT_CHARS, MAX_DOCUMENTS_PER_REQUEST

LONG_TEXT = " ".join(f"word{i}" for i in range(400))  # ~3,100 chars -> several 1000-char chunks


def doc(doc_id: str, text: str = "The Normans gave their name to Normandy.", **metadata: str) -> dict[str, Any]:
    body: dict[str, Any] = {"id": doc_id, "text": text}
    if metadata:
        body["metadata"] = metadata
    return body


def ingest(client: TestClient, *documents: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/corpus/documents", json={"documents": list(documents)})
    assert response.status_code == 200, response.text
    return response.json()


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ({"documents": []}, "at least 1"),
        ({}, "documents"),
        ({"documents": [doc("a", "")]}, "text must not be blank"),
        ({"documents": [doc("a", "  \n\t")]}, "text must not be blank"),
        ({"documents": [{"id": "a"}]}, "text"),
        ({"documents": [{"text": "hello"}]}, "id"),
        ({"documents": [doc("")]}, "id"),
        ({"documents": [doc("has space")]}, "id"),
        ({"documents": [doc("has#hash")]}, "id"),
        ({"documents": [doc("x" * 129)]}, "id"),
        ({"documents": [doc("a", "x" * (MAX_DOCUMENT_CHARS + 1))]}, "text"),
        ({"documents": [doc("a"), doc("a")]}, "duplicate document ids in request: a"),
        ({"documents": [doc("a")], "extra": 1}, "extra"),
        ({"documents": [{**doc("a"), "answerable": True}]}, "answerable"),
    ],
)
def test_malformed_requests_are_rejected(client: TestClient, body: dict[str, Any], fragment: str) -> None:
    response = client.post("/corpus/documents", json=body)

    assert response.status_code == 422
    assert fragment in response.text
    assert client.get("/corpus").json()["chunks"] == 0


@pytest.mark.parametrize("label", ["question", "answers", "groundTruth", "answerable", "squadId"])
def test_evaluation_labels_cannot_ride_in_metadata(client: TestClient, label: str) -> None:
    response = client.post(
        "/corpus/documents", json={"documents": [{**doc("a"), "metadata": {"title": "T", label: "x"}}]}
    )

    assert response.status_code == 422
    assert client.get("/corpus").json()["chunks"] == 0


def test_too_many_documents_are_rejected(client: TestClient) -> None:
    documents = [doc(f"d{i}") for i in range(MAX_DOCUMENTS_PER_REQUEST + 1)]

    assert client.post("/corpus/documents", json={"documents": documents}).status_code == 422


# --- ingestion --------------------------------------------------------------


def test_ingest_reports_counts_and_corpus_size(client: TestClient) -> None:
    result = ingest(client, doc("short"), doc("long", LONG_TEXT, title="Long", source="test"))

    assert result["documentsReceived"] == 2
    assert result["documentsCreated"] == 2
    assert (result["documentsReplaced"], result["documentsUnchanged"]) == (0, 0)
    assert result["chunksWritten"] > 2  # the long document was split
    assert result["chunksDeleted"] == 0
    assert result["corpus"] == {
        "documents": 2,
        "chunks": result["chunksWritten"],
        "embeddingModel": "fake-embedder",
    }
    assert client.get("/corpus").json() == result["corpus"]


def test_chunks_store_text_ids_and_metadata(client: TestClient) -> None:
    ingest(client, doc("long", LONG_TEXT, title="Long", source="test"))
    corpus: Corpus = app.dependency_overrides[get_corpus]()

    stored = corpus._collection.get(where={"documentId": "long"}, include=["documents", "metadatas"])
    by_index = sorted(zip(stored["ids"], stored["documents"], stored["metadatas"]), key=lambda r: r[2]["chunkIndex"])

    count = len(by_index)
    for index, (chunk_id, text, metadata) in enumerate(by_index):
        assert chunk_id == f"long#{index}"
        assert text == LONG_TEXT[metadata["charStart"] : metadata["charEnd"]]
        assert len(text) <= 1000
        assert metadata["documentId"] == "long"
        assert metadata["chunkIndex"] == index
        assert metadata["chunkCount"] == count
        assert metadata["title"] == "Long"
        assert metadata["source"] == "test"
        assert len(metadata["fingerprint"]) == 64
        assert set(metadata) == {
            "documentId", "chunkIndex", "chunkCount", "charStart", "charEnd", "fingerprint", "title", "source",
        }


# --- repeat ingestion and replacement ----------------------------------------


def test_identical_reingest_is_unchanged_and_not_reembedded(client: TestClient, embedder: Any) -> None:
    first = ingest(client, doc("a"), doc("b", LONG_TEXT))
    calls_before = len(embedder.calls)

    second = ingest(client, doc("a"), doc("b", LONG_TEXT))

    assert second["documentsUnchanged"] == 2
    assert (second["documentsCreated"], second["documentsReplaced"]) == (0, 0)
    assert (second["chunksWritten"], second["chunksDeleted"]) == (0, 0)
    assert second["corpus"] == first["corpus"]
    assert len(embedder.calls) == calls_before


def test_replacing_with_shorter_text_removes_stale_chunks(client: TestClient) -> None:
    first = ingest(client, doc("a", LONG_TEXT))
    old_chunks = first["corpus"]["chunks"]
    assert old_chunks > 1

    second = ingest(client, doc("a", "Now just one short sentence."))

    assert second["documentsReplaced"] == 1
    assert second["chunksWritten"] == 1
    assert second["chunksDeleted"] == old_chunks - 1
    assert second["corpus"] == {"documents": 1, "chunks": 1, "embeddingModel": "fake-embedder"}


def test_replacing_with_longer_text_adds_chunks_without_duplicates(client: TestClient) -> None:
    ingest(client, doc("a"))

    second = ingest(client, doc("a", LONG_TEXT))

    assert second["documentsReplaced"] == 1
    assert second["chunksDeleted"] == 0
    assert second["corpus"]["documents"] == 1
    assert second["corpus"]["chunks"] == second["chunksWritten"]


def test_metadata_only_change_replaces_the_document(client: TestClient) -> None:
    ingest(client, doc("a", title="Old"))

    second = ingest(client, doc("a", title="New"))

    assert second["documentsReplaced"] == 1
    assert second["corpus"]["chunks"] == 1


def test_mixed_batch_counts_each_document_once(client: TestClient) -> None:
    ingest(client, doc("same"), doc("changed"))

    result = ingest(client, doc("same"), doc("changed", "different text"), doc("new"))

    assert (result["documentsCreated"], result["documentsReplaced"], result["documentsUnchanged"]) == (1, 1, 1)
    assert result["corpus"]["documents"] == 3


# --- reset ------------------------------------------------------------------


def test_reset_empties_the_corpus(client: TestClient) -> None:
    filled = ingest(client, doc("a"), doc("b", LONG_TEXT))["corpus"]

    response = client.delete("/corpus")

    assert response.status_code == 200
    assert response.json() == {
        "documentsDeleted": 2,
        "chunksDeleted": filled["chunks"],
        "corpus": {"documents": 0, "chunks": 0, "embeddingModel": "fake-embedder"},
    }
    assert client.get("/corpus").json()["chunks"] == 0
    # The corpus stays usable after a reset.
    assert ingest(client, doc("a"))["documentsCreated"] == 1


def test_reset_of_empty_corpus_is_a_no_op(client: TestClient) -> None:
    response = client.delete("/corpus")

    assert response.status_code == 200
    assert (response.json()["documentsDeleted"], response.json()["chunksDeleted"]) == (0, 0)


@pytest.mark.parametrize(
    ("allow_reset", "client_host", "expected"),
    [
        (None, "127.0.0.1", 200),
        (None, "::1", 200),
        (None, "localhost", 200),
        (None, "192.168.1.20", 403),
        (None, "testclient", 403),
        (True, "192.168.1.20", 200),
        (False, "127.0.0.1", 403),
    ],
)
def test_reset_is_guarded(
    make_settings: Callable[..., Settings],
    make_corpus: Callable[..., Corpus],
    allow_reset: bool | None,
    client_host: str,
    expected: int,
) -> None:
    corpus = make_corpus()
    settings = make_settings(allow_reset=allow_reset)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_corpus] = lambda: corpus
    caller = TestClient(app, client=(client_host, 50000))
    ingest(caller, doc("keep"))

    response = caller.delete("/corpus")

    assert response.status_code == expected
    remaining = caller.get("/corpus").json()["documents"]
    assert remaining == (0 if expected == 200 else 1)