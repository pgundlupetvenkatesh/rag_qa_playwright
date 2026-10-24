"""FastAPI entry point.

Run from ``rag_service/`` with ``python -m app`` (host and port from settings)
or ``uvicorn app.main:app``.
"""

from __future__ import annotations

import ipaddress
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.corpus import Corpus, EmbeddingModelMismatchError, RetrievedChunk, SourceDocument
from app.embeddings import MiniLmOnnxEmbedder
from app.generation import Generator, GenerationError, build_generator
from app.readiness import ModelVerifier, get_model_verifier, run_readiness_checks
from app.schemas import (
    CorpusStats,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryResponse,
    QueryTimings,
    QuestionRequest,
    ReadinessResponse,
    ResetResponse,
    RetrievedChunkOut,
    RetrieveResponse,
    TokenUsage,
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


def get_generator(settings: SettingsDep) -> Generator:
    """FastAPI dependency returning the answer generator; tests override it.

    :param settings: Injected service settings.
    :type settings: Settings
    :returns: The Groq-backed generator.
    :rtype: Generator
    :raises GenerationError: ``missing_credentials`` when ``GROQ_API_KEY`` is
        unset, before any retrieval work is done.
    """
    return build_generator(settings)


GeneratorDep = Annotated[Generator, Depends(get_generator)]


@app.exception_handler(GenerationError)
def generation_error(request: Request, error: GenerationError) -> JSONResponse:
    """Render a :class:`GenerationError` as ``{"detail": {stage, code, message}}``.

    :param request: The failed request.
    :type request: Request
    :param error: The generation failure.
    :type error: GenerationError
    :returns: The error response, with ``Retry-After`` when Groq sent one.
    :rtype: JSONResponse
    """
    headers = {"Retry-After": error.retry_after} if error.retry_after else None
    body = {"detail": {"stage": "generation", "code": error.code, "message": error.message}}
    return JSONResponse(body, status_code=error.status_code, headers=headers)


def _retrieve(corpus: Corpus, question: str, top_k: int) -> tuple[list[RetrievedChunk], float]:
    """Run retrieval, failing explicitly when there is nothing to retrieve.

    :param corpus: The corpus.
    :type corpus: Corpus
    :param question: The question.
    :type question: str
    :param top_k: Maximum chunks to return.
    :type top_k: int
    :returns: The chunks, closest first, and the elapsed milliseconds.
    :rtype: tuple[list[RetrievedChunk], float]
    :raises HTTPException: 409 ``corpus_empty`` when nothing is ingested, or
        ``no_results`` when the corpus returned nothing.
    """
    started = time.perf_counter()
    chunks = corpus.search(question, top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not chunks:
        empty = corpus.size().chunks == 0
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {
                "stage": "retrieval",
                "code": "corpus_empty" if empty else "no_results",
                "message": "the corpus is empty; ingest documents first"
                if empty
                else "retrieval returned no chunks",
            },
        )
    return chunks, elapsed_ms


def _chunk_out(chunk: RetrievedChunk) -> RetrievedChunkOut:
    """Convert a retrieved chunk to its wire model.

    :param chunk: The chunk.
    :type chunk: RetrievedChunk
    :returns: The wire model, with similarity derived from distance.
    :rtype: RetrievedChunkOut
    """
    return RetrievedChunkOut(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        title=chunk.title,
        source=chunk.source,
        distance=chunk.distance,
        similarity=1 - chunk.distance,
    )


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

_STAGE_ERRORS = {status.HTTP_409_CONFLICT: {"model": ErrorResponse, "description": "Retrieval found nothing."}}


@app.post("/retrieve", response_model=RetrieveResponse, responses=_STAGE_ERRORS)
def retrieve(body: QuestionRequest, settings: SettingsDep, corpus: CorpusDep) -> RetrieveResponse:
    """Return the chunks nearest to the question. Never calls the language model.

    :param body: The question and optional ``topK``.
    :type body: QuestionRequest
    :param settings: Injected service settings (default ``TOP_K``).
    :type settings: Settings
    :param corpus: Injected corpus.
    :type corpus: Corpus
    :returns: Retrieved chunks, closest first.
    :rtype: RetrieveResponse
    :raises HTTPException: 409 when there is nothing to retrieve.
    """
    top_k = body.top_k or settings.top_k
    chunks, retrieval_ms = _retrieve(corpus, body.question, top_k)
    return RetrieveResponse(
        question=body.question,
        top_k=top_k,
        chunks=[_chunk_out(chunk) for chunk in chunks],
        retrieval_ms=retrieval_ms,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={
        **_STAGE_ERRORS,
        **{
            code: {"model": ErrorResponse, "description": "Answer generation failed."}
            for code in (429, 502, 503, 504)
        },
    },
)
def query(body: QuestionRequest, settings: SettingsDep, corpus: CorpusDep, generator: GeneratorDep) -> QueryResponse:
    """Retrieve evidence, then answer from it alone, or abstain.

    Errors carry ``detail.stage`` (``retrieval`` or ``generation``) so a caller
    can tell which half failed.

    :param body: The question and optional ``topK``.
    :type body: QuestionRequest
    :param settings: Injected service settings (default ``TOP_K``).
    :type settings: Settings
    :param corpus: Injected corpus.
    :type corpus: Corpus
    :param generator: Injected answer generator.
    :type generator: Generator
    :returns: The answer or abstention, the evidence shown to the model, and metadata.
    :rtype: QueryResponse
    :raises HTTPException: 409 when retrieval found nothing; the model is not called.
    :raises GenerationError: When the provider fails; rendered by :func:`generation_error`.
    """
    chunks, retrieval_ms = _retrieve(corpus, body.question, body.top_k or settings.top_k)
    started = time.perf_counter()
    generation = generator.generate(body.question, chunks)
    generation_ms = (time.perf_counter() - started) * 1000
    return QueryResponse(
        question=body.question,
        answer=generation.answer,
        abstained=generation.abstained,
        model=generation.model,
        evidence=[_chunk_out(chunk) for chunk in chunks],
        usage=TokenUsage(
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            total_tokens=generation.total_tokens,
        ),
        timings=QueryTimings(
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            provider_ms=generation.provider_ms,
        ),
    )
