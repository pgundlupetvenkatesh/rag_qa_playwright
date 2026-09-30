"""HTTP request and response models.

Field names are snake_case in Python and camelCase on the wire, matching the
TypeScript side of the repo.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class ApiModel(BaseModel):
    """Base for every wire model: camelCase aliases, snake_case accepted too."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


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