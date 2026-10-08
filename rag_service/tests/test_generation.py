"""GroqGenerator against a mocked Groq client: no credentials, no network, no charges."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import groq
import httpx
import pytest

from app.config import Settings
from app.corpus import RetrievedChunk
from app.generation import (
    ABSTAIN_SENTINEL,
    SYSTEM_PROMPT,
    GenerationError,
    GroqGenerator,
    build_generator,
)

EVIDENCE = [
    RetrievedChunk("doc-a#0", "doc-a", 0, "Normandy is a region in France.", "Normans", "squad_v2", 0.2),
    RetrievedChunk("doc-b#1", "doc-b", 1, "The Normans gave their name to Normandy.", None, None, 0.4),
]
REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def completion(
    content: str | None = "Normandy is in France.",
    finish_reason: str = "stop",
    model: str = "openai/gpt-oss-120b",
    usage: Any = None,
) -> SimpleNamespace:
    """A stand-in for groq's ChatCompletion with just the fields the generator reads."""
    if usage is None:
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150, total_time=0.25)
    message = SimpleNamespace(content=content)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], model=model, usage=usage
    )


def generator_returning(result: Any, model: str = "openai/gpt-oss-120b") -> tuple[GroqGenerator, MagicMock]:
    client = MagicMock()
    if isinstance(result, BaseException):
        client.chat.completions.create.side_effect = result
    else:
        client.chat.completions.create.return_value = result
    return GroqGenerator(client, model), client


def status_error(cls: type[groq.APIStatusError], code: int, headers: dict[str, str] | None = None) -> groq.APIStatusError:
    response = httpx.Response(code, request=REQUEST, headers=headers)
    return cls("provider says no", response=response, body=None)


def test_answer_and_metadata_are_returned() -> None:
    generator, _ = generator_returning(completion())

    result = generator.generate("Where is Normandy?", EVIDENCE)

    assert result.answer == "Normandy is in France."
    assert result.abstained is False
    assert result.model == "openai/gpt-oss-120b"
    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (120, 30, 150)
    assert result.provider_ms == pytest.approx(250)


def test_question_and_every_evidence_passage_reach_the_model() -> None:
    generator, client = generator_returning(completion())

    generator.generate("Where is Normandy?", EVIDENCE)

    kwargs = client.chat.completions.create.call_args.kwargs
    system, user = kwargs["messages"]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert "[1] (Normans)\nNormandy is a region in France." in user["content"]
    assert "[2]\nThe Normans gave their name to Normandy." in user["content"]
    assert user["content"].endswith("Question: Where is Normandy?")
    assert kwargs["temperature"] == 0
    assert kwargs["max_completion_tokens"] > 0


def test_prompt_demands_grounding_and_abstention() -> None:
    assert "only" in SYSTEM_PROMPT
    assert "outside knowledge" in SYSTEM_PROMPT
    assert ABSTAIN_SENTINEL in SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/gpt-oss-120b", {"include_reasoning": False, "reasoning_effort": "low"}),
        ("qwen/qwen3.8-27b", {}),
    ],
)
def test_reasoning_params_match_the_model_family(model: str, expected: dict[str, object]) -> None:
    generator, client = generator_returning(completion(model=model), model=model)

    generator.generate("q", EVIDENCE)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == model
    sent = {key: kwargs[key] for key in ("include_reasoning", "reasoning_effort", "reasoning_format") if key in kwargs}
    assert sent == expected


@pytest.mark.parametrize("reply", [ABSTAIN_SENTINEL, f"  {ABSTAIN_SENTINEL}.\n", f"`{ABSTAIN_SENTINEL}`"])
def test_sentinel_reply_is_an_abstention(reply: str) -> None:
    generator, _ = generator_returning(completion(content=reply))

    result = generator.generate("q", EVIDENCE)

    assert result.abstained is True
    assert result.answer is None


def test_sentinel_inside_a_longer_answer_is_not_an_abstention() -> None:
    generator, _ = generator_returning(completion(content=f"Partly {ABSTAIN_SENTINEL} but France."))

    assert generator.generate("q", EVIDENCE).abstained is False


def test_missing_usage_is_reported_as_unknown() -> None:
    generator, _ = generator_returning(completion(usage=SimpleNamespace()))

    result = generator.generate("q", EVIDENCE)

    assert (result.prompt_tokens, result.total_tokens, result.provider_ms) == (None, None, None)


@pytest.mark.parametrize(
    ("error", "code", "status"),
    [
        (groq.APITimeoutError(request=REQUEST), "provider_timeout", 504),
        (groq.APIConnectionError(request=REQUEST), "provider_unreachable", 502),
        (status_error(groq.AuthenticationError, 401), "provider_auth_failed", 503),
        (status_error(groq.PermissionDeniedError, 403), "provider_auth_failed", 503),
        (status_error(groq.NotFoundError, 404), "model_unavailable", 503),
        (status_error(groq.BadRequestError, 400), "provider_error", 502),
        (status_error(groq.InternalServerError, 500), "provider_error", 502),
    ],
)
def test_provider_failures_become_structured_errors(error: Exception, code: str, status: int) -> None:
    generator, client = generator_returning(error)

    with pytest.raises(GenerationError) as raised:
        generator.generate("q", EVIDENCE)

    assert (raised.value.code, raised.value.status_code) == (code, status)
    assert "provider says no" not in raised.value.message  # provider text is never echoed
    assert client.chat.completions.create.call_count == 1  # no retries


def test_rate_limit_passes_retry_after_through() -> None:
    generator, _ = generator_returning(status_error(groq.RateLimitError, 429, {"retry-after": "7"}))

    with pytest.raises(GenerationError) as raised:
        generator.generate("q", EVIDENCE)

    assert (raised.value.code, raised.value.status_code, raised.value.retry_after) == ("rate_limited", 429, "7")


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[], model="m", usage=None),
        SimpleNamespace(model="m"),
        completion(content=None),
        completion(content="   "),
        completion(finish_reason="length"),
        SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")], model="m"),
    ],
)
def test_malformed_responses_are_rejected(response: Any) -> None:
    generator, _ = generator_returning(response)

    with pytest.raises(GenerationError) as raised:
        generator.generate("q", EVIDENCE)

    assert (raised.value.code, raised.value.status_code) == ("malformed_response", 502)


def test_build_generator_requires_credentials(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(GenerationError) as raised:
        build_generator(make_settings())

    assert (raised.value.code, raised.value.status_code) == ("missing_credentials", 503)


def test_build_generator_disables_sdk_retries(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(groq_api_key="test-key", groq_model="qwen/qwen3.8-27b", request_timeout_seconds=5)

    generator = build_generator(settings)

    assert isinstance(generator, GroqGenerator)
    assert generator._client.max_retries == 0
    assert generator._client.timeout == 5
    assert generator._model == "qwen/qwen3.8-27b"