"""HTTP request and response models.

Field names are snake_case in Python and camelCase on the wire, matching the
TypeScript side of the repo.
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator
from pydantic.alias_generators import to_camel

MAX_DOCUMENTS_PER_REQUEST = 1000
MAX_DOCUMENT_CHARS = 100_000

DOCUMENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
"""Stable document ids: 1-128 chars, no ``#`` (reserved for chunk ids) or spaces."""


class ApiModel(BaseModel):
    """Base for every wire model: camelCase aliases, snake_case accepted too."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class RequestModel(ApiModel):
    """Base for request bodies: unknown fields are rejected, not ignored."""

    model_config = ConfigDict(extra="forbid")


class DocumentMetadata(RequestModel):
    """Optional descriptive metadata stored on every chunk of a document.

    An explicit allowlist on purpose: there is no free-form field through
    which questions, answers or answerability labels could reach the corpus.
    """

    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)] | None = None
    source: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)] | None = None


class DocumentIn(RequestModel):
    """One document to ingest. ``id`` is the caller's stable identifier."""

    id: Annotated[str, StringConstraints(pattern=DOCUMENT_ID_PATTERN)]
    text: Annotated[str, StringConstraints(max_length=MAX_DOCUMENT_CHARS)]
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        """Reject text that is empty or only whitespace.

        :param value: Document text.
        :type value: str
        :returns: ``value`` unchanged; offsets refer to it verbatim.
        :rtype: str
        :raises ValueError: If ``value`` has no non-whitespace character.
        """
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class IngestRequest(RequestModel):
    """Body of ``POST /corpus/documents``."""

    documents: Annotated[list[DocumentIn], Field(min_length=1, max_length=MAX_DOCUMENTS_PER_REQUEST)]

    @model_validator(mode="after")
    def _ids_unique(self) -> Self:
        """Reject a batch that names the same document twice.

        :returns: The validated request.
        :rtype: IngestRequest
        :raises ValueError: If any document id repeats.
        """
        counts = Counter(doc.id for doc in self.documents)
        repeated = sorted(doc_id for doc_id, count in counts.items() if count > 1)
        if repeated:
            raise ValueError(f"duplicate document ids in request: {', '.join(repeated)}")
        return self


class CorpusStats(ApiModel):
    """Size of the corpus. Body of ``GET /corpus``."""

    documents: int
    chunks: int
    embedding_model: str


class IngestResponse(ApiModel):
    """Outcome of one ingestion request, plus the corpus size afterwards.

    Every received document is counted in exactly one of ``created``,
    ``replaced`` or ``unchanged``.
    """

    documents_received: int
    documents_created: int
    documents_replaced: int
    documents_unchanged: int
    chunks_written: int
    chunks_deleted: int
    corpus: CorpusStats


class ResetResponse(ApiModel):
    """Body of ``DELETE /corpus``."""

    documents_deleted: int
    chunks_deleted: int
    corpus: CorpusStats


class HealthResponse(ApiModel):
    """Body of ``GET /health``: liveness only, no dependency is contacted."""

    status: Literal["ok"]
    embedding_model: str
    generation_model: str


class ReadinessCheck(ApiModel):
    """Outcome of one readiness check."""

    name: str
    ok: bool
    detail: str


class ReadinessResponse(ApiModel):
    """Body of ``GET /ready``. Sent with HTTP 503 when ``status`` is ``not_ready``."""

    status: Literal["ready", "not_ready"]
    checks: list[ReadinessCheck]