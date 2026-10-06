"""FastAPI entry point.

Run from ``rag_service/`` with ``python -m app`` (host and port from settings)
or ``uvicorn app.main:app``.
"""

from __future__ import annotations

import ipaddress
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status

from app.config import Settings, get_settings
from app.corpus import Corpus, EmbeddingModelMismatchError, SourceDocument
from app.embeddings import MiniLmOnnxEmbedder
from app.readiness import ModelVerifier, get_model_verifier, run_readiness_checks
from app.schemas import (
    CorpusStats,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ReadinessResponse,
    ResetResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load settings at startup so invalid configuration fails fast.

    Honours a test override of :func:`get_settings`.

    :param app: The application being started.
    :type app: FastAPI
    """
    app.dependency_overrides.get(get_settings, get_settings)()
    yield


app = FastAPI(title="rag_service", summary="Reference RAG service under test.", lifespan=lifespan)

SettingsDep = Annotated[Settings, Depends(get_settings)]


@lru_cache
def _open_corpus(settings: Settings) -> Corpus:
    """Open the corpus once per distinct settings object.

    The embedding model loads here, on the first corpus request rather than
    at startup, so ``/health`` and ``/ready`` stay fast.

    :param settings: Service settings.
    :type settings: Settings
    :returns: The shared corpus.
    :rtype: Corpus
    """
    return Corpus(
        path=settings.chroma_path,
        embedder=MiniLmOnnxEmbedder(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def get_corpus(settings: SettingsDep) -> Corpus:
    """FastAPI dependency returning the corpus; tests override it.

    :param settings: Injected service settings.
    :type settings: Settings
    :returns: The shared corpus.
    :rtype: Corpus
    :raises HTTPException: 503 if the stored corpus was built with a
        different embedding model.
    """
    try:
        return _open_corpus(settings)
    except EmbeddingModelMismatchError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error


CorpusDep = Annotated[Corpus, Depends(get_corpus)]


def _corpus_stats(corpus: Corpus) -> CorpusStats:
    """Current corpus counts as a wire model.

    :param corpus: The corpus.
    :type corpus: Corpus
    :returns: Document and chunk counts plus the embedding model.
    :rtype: CorpusStats
    """
    size = corpus.size()
    return CorpusStats(documents=size.documents, chunks=size.chunks, embedding_model=corpus.embedding_model)


def _is_loopback(host: str | None) -> bool:
    """Whether ``host`` is a loopback address.

    :param host: Client host as reported by the server, possibly ``None``.
    :type host: str | None
    :returns: True for 127.0.0.0/8, ``::1`` and ``localhost``.
    :rtype: bool
    """
    if host is None:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@app.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep) -> HealthResponse:
    """Liveness: the process is up. Contacts no dependency.

    :param settings: Injected service settings.
    :type settings: Settings
    :returns: Status and the configured model identifiers.
    :rtype: HealthResponse
    """
    return HealthResponse(
        status="ok",
        embedding_model=settings.embedding_model,
        generation_model=settings.groq_model,
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
def ready(
    settings: SettingsDep,
    verify_model: Annotated[ModelVerifier, Depends(get_model_verifier)],
    response: Response,
) -> ReadinessResponse:
    """Readiness: whether the service could do real work right now.

    Calls Groq once to confirm the configured model exists for the key.

    :param settings: Injected service settings.
    :type settings: Settings
    :param verify_model: Injected model availability check.
    :type verify_model: ModelVerifier
    :param response: Used to set HTTP 503 when any check fails.
    :type response: Response
    :returns: Overall status plus every individual check.
    :rtype: ReadinessResponse
    """
    checks = run_readiness_checks(settings, verify_model)
    if all(check.ok for check in checks):
        return ReadinessResponse(status="ready", checks=checks)
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="not_ready", checks=checks)


@app.get("/corpus", response_model=CorpusStats)
def corpus_stats(corpus: CorpusDep) -> CorpusStats:
    """Report how many documents and chunks the corpus holds.

    :param corpus: Injected corpus.
    :type corpus: Corpus
    :returns: Current counts.
    :rtype: CorpusStats
    """
    return _corpus_stats(corpus)


@app.post("/corpus/documents", response_model=IngestResponse)
def ingest_documents(body: IngestRequest, corpus: CorpusDep) -> IngestResponse:
    """Add documents, or replace existing ones with the same id.

    Safe to repeat: an identical document is reported as unchanged and not
    re-embedded; a changed one has its old chunks replaced, none left stale.

    :param body: Validated documents (422 on blank text, bad ids, duplicate
        ids, unknown fields).
    :type body: IngestRequest
    :param corpus: Injected corpus.
    :type corpus: Corpus
    :returns: Per-request counts and the corpus size afterwards.
    :rtype: IngestResponse
    """
    stats = corpus.ingest(
        [
            SourceDocument(id=doc.id, text=doc.text, title=doc.metadata.title, source=doc.metadata.source)
            for doc in body.documents
        ]
    )
    return IngestResponse(
        documents_received=stats.received,
        documents_created=stats.created,
        documents_replaced=stats.replaced,
        documents_unchanged=stats.unchanged,
        chunks_written=stats.chunks_written,
        chunks_deleted=stats.chunks_deleted,
        corpus=_corpus_stats(corpus),
    )


@app.delete(
    "/corpus",
    response_model=ResetResponse,
    responses={status.HTTP_403_FORBIDDEN: {"description": "Reset is disabled for this caller."}},
)
def reset_corpus(request: Request, settings: SettingsDep, corpus: CorpusDep) -> ResetResponse:
    """Delete every document. For local development and automated tests.

    Guarded by ``ALLOW_RESET``: unset allows only loopback callers, ``true``
    allows anyone, ``false`` refuses everyone.

    :param request: Used to read the caller's address.
    :type request: Request
    :param settings: Injected service settings.
    :type settings: Settings
    :param corpus: Injected corpus.
    :type corpus: Corpus
    :returns: What was deleted, and the (empty) corpus size.
    :rtype: ResetResponse
    :raises HTTPException: 403 when reset is not allowed for this caller.
    """
    if settings.allow_reset is False:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "corpus reset is disabled (ALLOW_RESET=false)")
    client_host = request.client.host if request.client else None
    if settings.allow_reset is None and not _is_loopback(client_host):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "corpus reset is only allowed from loopback addresses unless ALLOW_RESET=true",
        )
    deleted = corpus.reset()
    return ResetResponse(
        documents_deleted=deleted.documents,
        chunks_deleted=deleted.chunks,
        corpus=_corpus_stats(corpus),
    )