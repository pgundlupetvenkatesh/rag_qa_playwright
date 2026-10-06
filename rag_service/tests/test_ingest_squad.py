"""The SQuAD -> corpus mapping must carry passages only, never evaluation labels."""

import json

from scripts.ingest_squad import build_documents, passage_document_id

ROWS = [
    {
        "id": "q1",
        "title": "Normans",
        "context": "The Normans gave their name to Normandy, a region in France.",
        "question": "In what country is Normandy located?",
        "answers": {"text": ["France"], "answer_start": [52]},
    },
    {
        "id": "q2",
        "title": "Normans",
        "context": "The Normans gave their name to Normandy, a region in France.",
        "question": "What did the Normans give their name to?",
        "answers": {"text": ["Normandy"], "answer_start": [31]},
    },
    {
        "id": "q3",
        "title": "Oxygen",
        "context": "Oxygen is a chemical element with symbol O.",
        "question": "Who was the first emperor of Oxygen?",
        "answers": {"text": [], "answer_start": []},
    },
]


def test_documents_contain_passages_only() -> None:
    documents = build_documents(ROWS)

    serialised = json.dumps(documents)
    for row in ROWS:
        assert row["question"] not in serialised
        assert row["id"] not in serialised
    for document in documents:
        assert set(document) == {"id", "text", "metadata"}
        assert set(document["metadata"]) == {"title", "source"}


def test_passages_are_deduplicated_with_stable_content_ids() -> None:
    documents = build_documents(ROWS)

    assert len(documents) == 2
    assert {doc["text"] for doc in documents} == {row["context"] for row in ROWS}
    for document in documents:
        assert document["id"] == passage_document_id(document["text"])
        assert document["metadata"]["source"] == "squad_v2"
    assert [doc["id"] for doc in documents] == sorted(doc["id"] for doc in documents)


def test_output_does_not_depend_on_row_order() -> None:
    assert build_documents(ROWS) == build_documents(list(reversed(ROWS)))


def test_document_id_is_valid_for_the_api() -> None:
    import re

    from app.schemas import DOCUMENT_ID_PATTERN

    assert re.fullmatch(DOCUMENT_ID_PATTERN, passage_document_id("any passage"))