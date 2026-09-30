"""FastAPI entry point.

Run from ``rag_service/`` with ``python -m app`` (host and port from settings)
or ``uvicorn app.main:app``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Response, status

from app.config import Settings, get_settings
from app.readiness import ModelVerifier, get_model_verifier, run_readiness_checks
from app.schemas import HealthResponse, ReadinessResponse


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