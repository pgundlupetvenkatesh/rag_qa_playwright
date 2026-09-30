"""Readiness checks behind ``GET /ready``.

Liveness (``/health``) says only that the process is up. Readiness says the
service could do real work: it has a Groq key, the configured model exists for
that key, and the Chroma directory is usable. Checks for the vector store's
contents arrive with ingestion.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import groq

from app.config import Settings
from app.schemas import ReadinessCheck

ModelVerifier = Callable[[Settings], ReadinessCheck]
"""Checks that ``settings.groq_model`` is available. Swappable in tests."""


def verify_groq_model(settings: Settings) -> ReadinessCheck:
    """Ask Groq whether the configured model exists for the configured key.

    Makes one network call with no retries, bounded by
    ``settings.request_timeout_seconds``. Never raises: every failure becomes
    a failed check with a reason.

    :param settings: Service settings; ``groq_api_key`` must be set.
    :type settings: Settings
    :returns: The ``groqModel`` check result.
    :rtype: ReadinessCheck
    """
    name = "groqModel"
    if settings.groq_api_key is None:
        return ReadinessCheck(name=name, ok=False, detail="skipped: GROQ_API_KEY is not set")

    client = groq.Groq(
        api_key=settings.groq_api_key.get_secret_value(),
        timeout=settings.request_timeout_seconds,
        max_retries=0,
    )
    model = settings.groq_model
    try:
        client.models.retrieve(model)
    except groq.NotFoundError:
        detail = f"model {model!r} is not available to this API key"
    except groq.AuthenticationError:
        detail = "Groq rejected GROQ_API_KEY"
    except groq.APITimeoutError:
        detail = f"Groq did not answer within {settings.request_timeout_seconds}s"
    except groq.APIConnectionError:
        detail = "could not reach the Groq API"
    except groq.APIStatusError as error:
        detail = f"Groq returned HTTP {error.status_code}"
    else:
        return ReadinessCheck(name=name, ok=True, detail=f"model {model!r} is available")
    finally:
        client.close()
    return ReadinessCheck(name=name, ok=False, detail=detail)


def check_groq_api_key(settings: Settings) -> ReadinessCheck:
    """Report whether a Groq API key is configured. The value is never echoed.

    :param settings: Service settings.
    :type settings: Settings
    :returns: The ``groqApiKey`` check result.
    :rtype: ReadinessCheck
    """
    if settings.groq_api_key is None:
        return ReadinessCheck(name="groqApiKey", ok=False, detail="GROQ_API_KEY is not set")
    return ReadinessCheck(name="groqApiKey", ok=True, detail="configured")


def check_chroma_path(settings: Settings) -> ReadinessCheck:
    """Check that the Chroma directory exists or can be created, and is writable.

    Read-only: a readiness probe must not create the directory itself.

    :param settings: Service settings.
    :type settings: Settings
    :returns: The ``chromaPath`` check result.
    :rtype: ReadinessCheck
    """
    path = settings.chroma_path
    if path.exists() and not path.is_dir():
        return ReadinessCheck(name="chromaPath", ok=False, detail=f"{path} is not a directory")

    # The directory itself, or the nearest ancestor it would be created under.
    probe: Path = path
    while not probe.exists():
        probe = probe.parent
    if not os.access(probe, os.W_OK):
        return ReadinessCheck(name="chromaPath", ok=False, detail=f"{probe} is not writable")
    state = "exists" if path.exists() else "will be created"
    return ReadinessCheck(name="chromaPath", ok=True, detail=f"{path} {state}")


def run_readiness_checks(settings: Settings, verify_model: ModelVerifier) -> list[ReadinessCheck]:
    """Run every readiness check, in a fixed order.

    :param settings: Service settings.
    :type settings: Settings
    :param verify_model: Model availability check, normally
        :func:`verify_groq_model`.
    :type verify_model: ModelVerifier
    :returns: One result per check. Failures are returned, not raised.
    :rtype: list[ReadinessCheck]
    """
    return [
        check_groq_api_key(settings),
        verify_model(settings),
        check_chroma_path(settings),
    ]


def get_model_verifier() -> ModelVerifier:
    """FastAPI dependency returning the model check; tests override it.

    :returns: :func:`verify_groq_model`.
    :rtype: ModelVerifier
    """
    return verify_groq_model