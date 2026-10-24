"""Corpus behaviour below the HTTP layer: persistence, model guard, parameters."""

from collections.abc import Callable

import pytest

from app.corpus import Corpus, EmbeddingModelMismatchError, SourceDocument
from tests.conftest import FakeEmbedder

LONG_TEXT = " ".join(f"word{i}" for i in range(400))


def test_corpus_persists_across_reopen(make_corpus: Callable[..., Corpus]) -> None:
    make_corpus().ingest([SourceDocument("a", LONG_TEXT)])

    reopened = make_corpus()

    assert reopened.size().documents == 1
    assert reopened.ingest([SourceDocument("a", LONG_TEXT)]).unchanged == 1


def test_changing_chunk_parameters_rebuilds_existing_documents(make_corpus: Callable[..., Corpus]) -> None:
    first = make_corpus(chunk_size=1000, chunk_overlap=200).ingest([SourceDocument("a", LONG_TEXT)])

    second = make_corpus(chunk_size=500, chunk_overlap=100).ingest([SourceDocument("a", LONG_TEXT)])

    assert second.replaced == 1
    assert second.chunks_written > first.chunks_written


def test_opening_with_a_different_embedding_model_is_refused(make_corpus: Callable[..., Corpus]) -> None:
    make_corpus().ingest([SourceDocument("a", "text")])

    class OtherEmbedder(FakeEmbedder):
        model_name = "other-model"

    with pytest.raises(EmbeddingModelMismatchError, match="fake-embedder"):
        make_corpus(embedder_=OtherEmbedder())


def test_reset_returns_what_it_deleted(make_corpus: Callable[..., Corpus]) -> None:
    corpus = make_corpus()
    corpus.ingest([SourceDocument("a", "text")])

    deleted = corpus.reset()

    assert (deleted.documents, deleted.chunks) == (1, 1)
    assert corpus.size().chunks == 0


def test_empty_ingest_is_a_no_op(make_corpus: Callable[..., Corpus], embedder: FakeEmbedder) -> None:
    stats = make_corpus().ingest([])

    assert stats.received == 0
    assert embedder.calls == []


def test_all_new_chunks_are_embedded_in_one_batch(
    make_corpus: Callable[..., Corpus], embedder: FakeEmbedder
) -> None:
    make_corpus().ingest([SourceDocument("a", LONG_TEXT), SourceDocument("b", "short")])

    assert len(embedder.calls) == 1