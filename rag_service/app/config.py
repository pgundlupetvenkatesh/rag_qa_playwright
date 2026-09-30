"""Service configuration, read from environment variables and ``rag_service/.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Chroma's built-in ONNX embedding function ships only this model. It is also
# the one that runs on Intel Macs, where current torch wheels do not exist.
SUPPORTED_EMBEDDING_MODELS = frozenset({"all-MiniLM-L6-v2"})


class Settings(BaseSettings):
    """Runtime settings. Each field maps to the upper-cased env var of its name.

    Real environment variables take precedence over ``rag_service/.env``.
    Relative paths resolve against ``rag_service/`` so the working directory
    does not matter.
    """

    model_config = SettingsConfigDict(
        env_file=SERVICE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    groq_api_key: SecretStr | None = None
    """Groq API key. Optional at startup; ``/ready`` reports it missing."""

    groq_model: str = Field(default="openai/gpt-oss-20b", min_length=1)
    """Groq model id. Not assumed to exist: ``/ready`` verifies it with Groq."""

    embedding_model: str = "all-MiniLM-L6-v2"
    """Embedding model id; must be in :data:`SUPPORTED_EMBEDDING_MODELS`."""

    chroma_path: Path = Path(".chroma")
    """Persistent Chroma directory."""

    chunk_size: int = Field(default=1000, gt=0)
    """Maximum chunk length in characters. The embedding model reads at most
    256 tokens; 1000 characters keeps 99.5% of SQuAD validation chunks within
    that, so almost no chunk text is silently dropped from its embedding."""

    chunk_overlap: int = Field(default=200, ge=0)
    """Characters shared by consecutive chunks of a split passage."""

    top_k: int = Field(default=4, gt=0)
    """Passages retrieved per question."""

    api_host: str = "127.0.0.1"
    """Bind address used by ``python -m app``."""

    api_port: int = Field(default=8000, gt=0, lt=65536)
    """Bind port used by ``python -m app``."""

    request_timeout_seconds: float = Field(default=30.0, gt=0)
    """Timeout for outbound calls to Groq."""

    @field_validator("groq_api_key", mode="before")
    @classmethod
    def _blank_key_is_unset(cls, value: object) -> object:
        """Treat an empty ``GROQ_API_KEY=`` line as no key at all.

        :param value: Raw value from the environment.
        :type value: object
        :returns: ``None`` for a blank string, otherwise ``value`` unchanged.
        :rtype: object
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("embedding_model")
    @classmethod
    def _embedding_model_supported(cls, value: str) -> str:
        """Reject embedding models the service cannot load.

        :param value: Configured embedding model id.
        :type value: str
        :returns: ``value`` unchanged.
        :rtype: str
        :raises ValueError: If ``value`` is not a supported model.
        """
        if value not in SUPPORTED_EMBEDDING_MODELS:
            supported = ", ".join(sorted(SUPPORTED_EMBEDDING_MODELS))
            raise ValueError(f"unsupported embedding model {value!r}; supported: {supported}")
        return value

    @field_validator("chroma_path")
    @classmethod
    def _anchor_relative_path(cls, value: Path) -> Path:
        """Resolve a relative Chroma path against ``rag_service/``.

        :param value: Configured path.
        :type value: Path
        :returns: An absolute path.
        :rtype: Path
        """
        return value if value.is_absolute() else SERVICE_ROOT / value

    @model_validator(mode="after")
    def _overlap_below_size(self) -> Self:
        """Require the overlap to be smaller than the chunk size.

        :returns: The validated settings.
        :rtype: Settings
        :raises ValueError: If ``chunk_overlap >= chunk_size``, which would
            make chunking loop forever.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first use.

    Used as a FastAPI dependency, so tests substitute their own
    :class:`Settings` through ``app.dependency_overrides``.

    :returns: The cached settings.
    :rtype: Settings
    :raises pydantic.ValidationError: If any variable is invalid.
    """
    return Settings()