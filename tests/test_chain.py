"""Deterministic tests for envelope validation, retries, logging, and async processing."""

import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from httpx import Request
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from openai import (
    APIConnectionError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
)
from openai.types.chat import ChatCompletion
from tenacity import wait_exponential_jitter

import chain
from schemas import ExtraccionTecnica, NivelDeCriticidad


def _extraction() -> ExtraccionTecnica:
    return ExtraccionTecnica(
        tecnologias=["Python", "PostgreSQL"],
        nivel_de_criticidad="alta",
        resumen_tecnico="Python workers depend on PostgreSQL during critical ingestion.",
    )


def _envelope(
    *,
    parsed: object = None,
    finish_reason: str = "stop",
    parsing_error: Exception | None = None,
) -> dict[str, object]:
    return {
        "raw": AIMessage(content="", response_metadata={"finish_reason": finish_reason}),
        "parsed": parsed,
        "parsing_error": parsing_error,
    }


@pytest.fixture
def instant_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wait_exponential_jitter, "__call__", lambda self, retry_state: 0.0)


def _json_logs(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [json.loads(record.getMessage()) for record in caplog.records]


def test_prompt_uses_only_text_variable() -> None:
    assert chain.PROMPT.input_variables == ["text"]
    messages = chain.PROMPT.format_messages(text="A Python service uses Redis.")
    assert "A Python service uses Redis." in str(messages[-1].content)


def test_envelope_validation_returns_validated_model() -> None:
    result = chain._validate_envelope(_envelope(parsed=_extraction()))

    assert result == _extraction()


def test_envelope_validation_accepts_tool_calls_finish_reason() -> None:
    result = chain._validate_envelope(_envelope(parsed=_extraction(), finish_reason="tool_calls"))

    assert result.nivel_de_criticidad is NivelDeCriticidad.ALTA


def test_envelope_validation_rejects_parsing_failure() -> None:
    with pytest.raises(chain.StructuredOutputValidationError, match="could not be parsed"):
        chain._validate_envelope(
            _envelope(parsed=None, parsing_error=ValueError("malformed provider payload"))
        )


@pytest.mark.parametrize("envelope", [None, {"parsed": None}])
def test_envelope_validation_rejects_invalid_or_missing_raw_message(envelope: object) -> None:
    with pytest.raises(chain.StructuredOutputValidationError, match="envelope"):
        chain._validate_envelope(envelope)


def test_envelope_validation_rejects_length_finish_reason_before_parsed_data() -> None:
    with pytest.raises(chain.IncompleteModelOutputError, match="incomplete"):
        chain._validate_envelope(_envelope(parsed=_extraction(), finish_reason="length"))


def test_envelope_validation_accepts_completed_responses_status() -> None:
    raw = AIMessage(content="", response_metadata={"status": "completed"})
    result = chain._validate_envelope({"raw": raw, "parsed": _extraction(), "parsing_error": None})

    assert result == _extraction()


def test_envelope_validation_rejects_incomplete_responses_status() -> None:
    raw = AIMessage(content="", response_metadata={"status": "incomplete"})
    with pytest.raises(chain.IncompleteModelOutputError, match="incomplete"):
        chain._validate_envelope({"raw": raw, "parsed": _extraction(), "parsing_error": None})


def test_envelope_validation_rejects_missing_completion_signals() -> None:
    envelope = {
        "raw": AIMessage(content="", response_metadata={}),
        "parsed": _extraction(),
        "parsing_error": None,
    }

    with pytest.raises(chain.IncompleteModelOutputError, match="incomplete"):
        chain._validate_envelope(envelope)


def test_envelope_validation_rejects_content_filter_finish_reason() -> None:
    with pytest.raises(chain.IncompleteModelOutputError, match="incomplete"):
        chain._validate_envelope(_envelope(parsed=_extraction(), finish_reason="content_filter"))


def test_envelope_validation_rejects_missing_parsed_output() -> None:
    with pytest.raises(chain.StructuredOutputValidationError, match="did not contain parsed data"):
        chain._validate_envelope(_envelope(parsed=None))


def test_envelope_validation_revalidates_untrusted_parsed_output() -> None:
    invalid = {
        "tecnologias": [],
        "nivel_de_criticidad": "urgent",
        "resumen_tecnico": "short",
    }

    with pytest.raises(chain.StructuredOutputValidationError, match="failed schema validation"):
        chain._validate_envelope(_envelope(parsed=invalid))


@pytest.mark.asyncio
async def test_malformed_and_incomplete_outputs_retry_then_recover(instant_retry: None) -> None:
    responses: Iterator[dict[str, object]] = iter(
        [
            _envelope(parsing_error=ValueError("invalid JSON")),
            _envelope(parsed=_extraction(), finish_reason="length"),
            _envelope(parsed=_extraction()),
        ]
    )
    attempt_count = 0

    async def fake_structured_model(_: object) -> object:
        nonlocal attempt_count
        attempt_count += 1
        return next(responses)

    extractor = chain._build_resilient_extractor(RunnableLambda(fake_structured_model))
    result = await extractor.ainvoke(object())

    assert result == _extraction()
    assert attempt_count == 3


@pytest.mark.asyncio
async def test_retry_exhaustion_uses_exact_attempt_count(instant_retry: None) -> None:
    attempt_count = 0

    async def always_incomplete(_: object) -> object:
        nonlocal attempt_count
        attempt_count += 1
        return _envelope(parsed=_extraction(), finish_reason="length")

    extractor = chain._build_resilient_extractor(RunnableLambda(always_incomplete))

    with pytest.raises(chain.IncompleteModelOutputError):
        await extractor.ainvoke(object())

    assert attempt_count == 3


@pytest.mark.parametrize(
    "parse_error",
    [
        LengthFinishReasonError(completion=ChatCompletion.model_construct(usage=None)),
        ContentFilterFinishReasonError(),
    ],
    ids=["length", "content-filter"],
)
@pytest.mark.asyncio
async def test_pre_envelope_parse_exception_retries_then_recovers(
    instant_retry: None, parse_error: Exception
) -> None:
    responses = iter([parse_error, _envelope(parsed=_extraction())])

    async def parse_failure_then_valid(_: object) -> object:
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    extractor = chain._build_resilient_extractor(RunnableLambda(parse_failure_then_valid))
    assert await extractor.ainvoke(object()) == _extraction()


def test_configure_json_logging_installs_safe_cli_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chain.LOGGER, "handlers", [])
    monkeypatch.setattr(chain.LOGGER, "propagate", True)
    monkeypatch.setattr(chain.LOGGER, "level", logging.NOTSET)

    chain.configure_json_logging(logging.WARNING)

    assert len(chain.LOGGER.handlers) == 1
    assert chain.LOGGER.handlers[0].formatter is not None
    assert chain.LOGGER.level == logging.WARNING
    assert chain.LOGGER.propagate is False


def test_create_model_uses_environment_and_disables_nested_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_chat_openai(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setenv("OPENAI_MODEL", "test-low-cost-model")
    monkeypatch.setattr(chain, "ChatOpenAI", fake_chat_openai)

    result = chain._create_model()

    assert result is sentinel
    assert captured == {
        "model": "test-low-cost-model",
        "temperature": 0,
        "timeout": 30.0,
        "max_retries": 0,
    }


@pytest.mark.asyncio
async def test_build_chain_requests_inspectable_native_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options: dict[str, object] = {}

    class FakeModel:
        def with_structured_output(
            self, schema: object, **kwargs: object
        ) -> RunnableLambda[Any, Any]:
            options["schema"] = schema
            options.update(kwargs)
            return RunnableLambda(lambda _: _envelope(parsed=_extraction()))

    monkeypatch.setattr(chain, "_create_model", FakeModel)

    result = await chain.build_chain().ainvoke({"text": "Python writes to PostgreSQL."})

    assert result == _extraction()
    assert options == {
        "schema": ExtraccionTecnica,
        "method": "json_schema",
        "include_raw": True,
    }


@pytest.mark.asyncio
async def test_process_text_invokes_chain_asynchronously(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[dict[str, str]] = []

    async def fake_chain(payload: dict[str, str]) -> ExtraccionTecnica:
        received.append(payload)
        return _extraction()

    monkeypatch.setattr(chain, "build_chain", lambda: RunnableLambda(fake_chain))

    result = await chain.process_text("Python workers write to PostgreSQL.")

    assert result == _extraction()
    assert received == [{"text": "Python workers write to PostgreSQL."}]


@pytest.mark.asyncio
async def test_blank_input_fails_before_chain_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    chain_built = False

    def forbidden_build() -> RunnableLambda[Any, Any]:
        nonlocal chain_built
        chain_built = True
        raise AssertionError("provider chain must not be built")

    monkeypatch.setattr(chain, "build_chain", forbidden_build)

    with pytest.raises(ValueError, match="must not be blank"):
        await chain.process_text("  \n ")

    assert chain_built is False


@pytest.mark.asyncio
async def test_processing_logs_start_success_and_safe_context(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=chain.LOGGER.name)
    monkeypatch.setattr(
        chain,
        "build_chain",
        lambda: RunnableLambda(lambda _: _extraction()),
    )

    await chain.process_text("A Python worker uses PostgreSQL.")
    logs = _json_logs(caplog)

    assert [entry["event"] for entry in logs] == [
        "extraction.processing_started",
        "extraction.succeeded",
    ]
    assert all(entry["correlation_id"] != "unscoped" for entry in logs)
    assert all(isinstance(entry["duration_ms"], float) for entry in logs)


@pytest.mark.asyncio
async def test_validation_and_retry_events_are_observable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    instant_retry: None,
) -> None:
    caplog.set_level(logging.INFO, logger=chain.LOGGER.name)
    responses = iter(
        [
            _envelope(parsing_error=ValueError("invalid JSON")),
            _envelope(parsed=_extraction()),
        ]
    )

    async def fake_structured_model(_: object) -> object:
        return next(responses)

    extractor = chain._build_resilient_extractor(RunnableLambda(fake_structured_model))
    monkeypatch.setattr(chain, "build_chain", lambda: chain.PROMPT | extractor)

    await chain.process_text("Python is connected to PostgreSQL.")
    events = [entry["event"] for entry in _json_logs(caplog)]

    assert "extraction.validation_failure" in events
    assert "extraction.retry" in events
    assert "extraction.succeeded" in events


@pytest.mark.asyncio
async def test_transient_provider_failure_is_logged_and_retried(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    instant_retry: None,
) -> None:
    caplog.set_level(logging.INFO, logger=chain.LOGGER.name)
    attempt_count = 0

    async def transient_then_valid(_: object) -> object:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise APIConnectionError(request=Request("POST", "https://provider.invalid"))
        return _envelope(parsed=_extraction())

    extractor = chain._build_resilient_extractor(RunnableLambda(transient_then_valid))
    monkeypatch.setattr(chain, "build_chain", lambda: chain.PROMPT | extractor)

    result = await chain.process_text("Python sends records to PostgreSQL.")
    events = [entry["event"] for entry in _json_logs(caplog)]

    assert result == _extraction()
    assert attempt_count == 2
    assert "extraction.provider_transient_failure" in events
    assert "extraction.retry" in events


@pytest.mark.asyncio
async def test_raw_text_and_api_keys_are_never_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_text = "Sensitive architecture text with a private hostname"
    api_key = "test-api-key-never-log"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    caplog.set_level(logging.INFO, logger=chain.LOGGER.name)

    async def failing_chain(_: object) -> ExtraccionTecnica:
        raise RuntimeError(raw_text + api_key)

    monkeypatch.setattr(chain, "build_chain", lambda: RunnableLambda(failing_chain))

    with pytest.raises(RuntimeError):
        await chain.process_text(raw_text)

    rendered_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert raw_text not in rendered_logs
    assert api_key not in rendered_logs
    assert "extraction.terminal_error" in rendered_logs


@pytest.mark.asyncio
async def test_ambiguous_incident_text_flows_through_prompt_and_validation() -> None:
    ambiguous_text = (
        "During a Kubernetes rollout, Go workers reported Kafka timeouts while the "
        "PostgreSQL replica lagged. Metrics recovered after a network policy rollback, "
        "but the evidence does not identify which dependency initiated the incident."
    )
    observed_prompt = ""
    expected = ExtraccionTecnica(
        tecnologias=["Kubernetes", "Go", "Kafka", "PostgreSQL"],
        nivel_de_criticidad="alta",
        resumen_tecnico="A rollout coincided with messaging, database, and network symptoms.",
    )

    async def fake_structured_model(prompt_value: Any) -> object:
        nonlocal observed_prompt
        observed_prompt = str(prompt_value.to_messages()[-1].content)
        return _envelope(parsed=expected)

    pipeline = chain.PROMPT | chain._build_resilient_extractor(
        RunnableLambda(fake_structured_model)
    )
    result = await pipeline.ainvoke({"text": ambiguous_text})

    assert ambiguous_text in observed_prompt
    assert result == expected
