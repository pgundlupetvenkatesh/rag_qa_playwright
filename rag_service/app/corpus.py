"""The document corpus: chunked, embedded and persisted in ChromaDB.

Chunk ids are ``"<documentId>#<chunkIndex>"``, so re-ingesting a document
overwrites its chunks in place. Each chunk records a fingerprint of everything
that shaped it (text, metadata, chunking parameters, embedding model): an
identical re-ingest is detected and skipped, and a changed one replaces the old
chunks and deletes any left over, so no stale chunk survives.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.chunking import chunk_text
from app.embeddings import Embedder

COLLECTION_NAME = "rag_corpus"
_T = TypeVar("_T")


class EmbeddingModelMismatchError(RuntimeError):
    """The persisted collection was built with a different embedding model."""


@dataclass(frozen=True)
class SourceDocument:
    """A document to ingest, independent of the HTTP layer.

    :ivar id: Caller's stable document id.
    :ivar text: Full document text.
    :ivar title: Optional title, stored on every chunk.
    :ivar source: Optional source label (e.g. ``squad_v2``), stored on every chunk.
    """

    id: str
    text: str
    title: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class IngestStats:
    """What one ingestion call did. Each document lands in exactly one bucket.

    :ivar received: Documents in the call.
    :ivar created: Documents that were not in the corpus before.
    :ivar replaced: Existing documents whose chunks were rewritten.
    :ivar unchanged: Existing documents skipped because nothing changed.
    :ivar chunks_written: Chunks embedded and upserted.
    :ivar chunks_deleted: Stale chunks removed from replaced documents.
    """

    received: int
    created: int
    replaced: int
    unchanged: int
    chunks_written: int
    chunks_deleted: int


@dataclass(frozen=True)
class CorpusSize:
    """Document and chunk counts.

    :ivar documents: Distinct documents in the corpus.
    :ivar chunks: Chunks in the corpus.
    """

    documents: int
    chunks: int


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by :meth:`Corpus.search`.

    :ivar chunk_id: ``"<documentId>#<chunkIndex>"``.
    :ivar document_id: Id of the parent document.
    :ivar chunk_index: Position of the chunk within its document.
    :ivar text: Chunk text.
    :ivar title: Parent document title, if it had one.
    :ivar source: Parent document source label, if it had one.
    :ivar distance: Cosine distance to the question (0 = same direction, 2 = opposite).
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    title: str | None
    source: str | None
    distance: float


def _batches(items: Sequence[_T], size: int) -> Iterator[Sequence[_T]]:
    """Yield consecutive slices of at most ``size`` items.

    :param items: Items to split.
    :type items: Sequence
    :param size: Maximum slice length.
    :type size: int
    :returns: An iterator over the slices.
    :rtype: Iterator[Sequence]
    """
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


class Corpus:
    """Persistent, chunked document store backed by one Chroma collection.

    Writes are serialised with a lock, so concurrent requests cannot interleave
    a replacement or race a reset. Chroma's anonymous telemetry is disabled:
    the only outbound traffic is the one-time embedding model download.
    """

    def __init__(self, path: Path, embedder: Embedder, chunk_size: int, chunk_overlap: int) -> None:
        """Open (or create) the corpus at ``path``.

        :param path: Directory of the persistent Chroma store.
        :type path: Path
        :param embedder: Embedding implementation for chunk text.
        :type embedder: Embedder
        :param chunk_size: Maximum chunk length in characters.
        :type chunk_size: int
        :param chunk_overlap: Target overlap between neighbouring chunks.
        :type chunk_overlap: int
        :raises EmbeddingModelMismatchError: If the stored collection was built
            with a different embedding model.
        """
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._lock = threading.Lock()
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._open_collection()

    @property
    def embedding_model(self) -> str:
        """Name of the embedding model this corpus uses."""
        return self._embedder.model_name

    def _open_collection(self) -> Any:
        """Get or create the collection and check its embedding model.

        :returns: The Chroma collection.
        :raises EmbeddingModelMismatchError: If the collection records a
            different embedding model; mixing vectors from two models makes
            similarity scores meaningless.
        """
        collection = self._client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"embeddingModel": self.embedding_model},
            configuration={"hnsw": {"space": "cosine"}},
        )
        stored = (collection.metadata or {}).get("embeddingModel")
        if stored != self.embedding_model:
            raise EmbeddingModelMismatchError(
                f"collection {COLLECTION_NAME!r} was built with {stored!r}, "
                f"not {self.embedding_model!r}; reset the corpus to rebuild it"
            )
        return collection

    def _fingerprint(self, document: SourceDocument) -> str:
        """Hash everything that determines a document's stored chunks.

        :param document: The document.
        :type document: SourceDocument
        :returns: A hex digest; equal digests mean identical chunks.
        :rtype: str
        """
        payload = json.dumps(
            [
                document.text,
                document.title,
                document.source,
                self._chunk_size,
                self._chunk_overlap,
                self.embedding_model,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _size(self) -> CorpusSize:
        """Count chunks, and documents via their first chunk. Caller holds the lock.

        :returns: Current counts.
        :rtype: CorpusSize
        """
        first_chunks = self._collection.get(where={"chunkIndex": 0}, include=[])
        return CorpusSize(documents=len(first_chunks["ids"]), chunks=self._collection.count())

    def size(self) -> CorpusSize:
        """Return the current document and chunk counts.

        :returns: Current counts.
        :rtype: CorpusSize
        """
        with self._lock:
            return self._size()

    def ingest(self, documents: Sequence[SourceDocument]) -> IngestStats:
        """Add or replace documents by id.

        Unchanged documents are skipped. For the rest, new chunks are embedded
        and upserted first, then any chunk ids the new version no longer has
        are deleted, so a document is never missing mid-replacement.

        :param documents: Documents with unique ids and non-blank text.
        :type documents: Sequence[SourceDocument]
        :returns: Per-call counts.
        :rtype: IngestStats
        :raises ValueError: If a document's text is blank.
        """
        if not documents:
            return IngestStats(0, 0, 0, 0, 0, 0)
        with self._lock:
            existing = self._collection.get(
                where={"documentId": {"$in": [doc.id for doc in documents]}},
                include=["metadatas"],
            )
            stored_ids: dict[str, set[str]] = {}
            stored_fingerprints: dict[str, set[str]] = {}
            for chunk_id, metadata in zip(existing["ids"], existing["metadatas"] or [], strict=True):
                doc_id = str(metadata["documentId"])
                stored_ids.setdefault(doc_id, set()).add(chunk_id)
                stored_fingerprints.setdefault(doc_id, set()).add(str(metadata["fingerprint"]))

            created = replaced = unchanged = 0
            write_ids: list[str] = []
            write_texts: list[str] = []
            write_metadatas: list[dict[str, str | int]] = []
            stale_ids: list[str] = []

            for doc in documents:
                fingerprint = self._fingerprint(doc)
                chunks = chunk_text(doc.text, self._chunk_size, self._chunk_overlap)
                new_ids = [f"{doc.id}#{chunk.index}" for chunk in chunks]
                old_ids = stored_ids.get(doc.id, set())

                if old_ids == set(new_ids) and stored_fingerprints.get(doc.id) == {fingerprint}:
                    unchanged += 1
                    continue
                if old_ids:
                    replaced += 1
                    stale_ids.extend(sorted(old_ids - set(new_ids)))
                else:
                    created += 1

                for chunk_id, chunk in zip(new_ids, chunks, strict=True):
                    metadata: dict[str, str | int] = {
                        "documentId": doc.id,
                        "chunkIndex": chunk.index,
                        "chunkCount": len(chunks),
                        "charStart": chunk.start,
                        "charEnd": chunk.end,
                        "fingerprint": fingerprint,
                    }
                    if doc.title is not None:
                        metadata["title"] = doc.title
                    if doc.source is not None:
                        metadata["source"] = doc.source
                    write_ids.append(chunk_id)
                    write_texts.append(chunk.text)
                    write_metadatas.append(metadata)

            batch_size = self._client.get_max_batch_size()
            if write_ids:
                embeddings = self._embedder.embed(write_texts)
                for start in range(0, len(write_ids), batch_size):
                    end = start + batch_size
                    self._collection.upsert(
                        ids=write_ids[start:end],
                        embeddings=embeddings[start:end],
                        documents=write_texts[start:end],
                        metadatas=write_metadatas[start:end],
                    )
            for batch in _batches(stale_ids, batch_size):
                self._collection.delete(ids=list(batch))

            return IngestStats(
                received=len(documents),
                created=created,
                replaced=replaced,
                unchanged=unchanged,
                chunks_written=len(write_ids),
                chunks_deleted=len(stale_ids),
            )

    def search(self, question: str, top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks nearest to ``question``, closest first.

        :param question: Text to search for; embedded with the corpus's model.
        :type question: str
        :param top_k: Maximum number of chunks to return.
        :type top_k: int
        :returns: Up to ``top_k`` chunks; empty when the corpus is empty.
        :rtype: list[RetrievedChunk]
        """
        [vector] = self._embedder.embed([question])
        # Under the lock so a concurrent reset cannot swap the collection mid-query.
        # ponytail: one lock for reads and writes, so a long ingest blocks queries.
        with self._lock:
            count = self._collection.count()
            if count == 0:
                return []
            result = self._collection.query(
                query_embeddings=[vector],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=str(metadata["documentId"]),
                chunk_index=int(metadata["chunkIndex"]),
                text=text,
                title=metadata.get("title"),
                source=metadata.get("source"),
                distance=float(distance),
            )
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]

    def reset(self) -> CorpusSize:
        """Delete every document by dropping and recreating the collection.

        :returns: The counts that were deleted.
        :rtype: CorpusSize
        """
        with self._lock:
            deleted = self._size()
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = self._open_collection()
            return deleted