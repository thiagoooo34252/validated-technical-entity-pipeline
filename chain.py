"""Resilient LCEL pipeline for validated technical entity extraction."""

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import NoReturn, cast
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    ContentFilterFinishReasonError,
    InternalServerError,
    LengthFinishReasonError,
    RateLimitError,
)
from pydantic import ValidationError

from schemas import ExtraccionTecnica

LOGGER = logging.getLogger("technical_entity_pipeline")

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un analista de arquitectura de software. Extrae solamente tecnologias "
            "mencionadas o inequívocamente identificables en el texto. Evalua la criticidad "
            "tecnica como baja, media o alta y redacta un resumen tecnico conciso en el idioma "
            "del texto de entrada. No inventes datos y cumple exactamente el esquema solicitado.",
        ),
        ("human", "Texto tecnico para analizar:\n{text}"),
    ]
)

_COMPLETE_FINISH_REASONS = frozenset({"stop", "tool_calls"})
_MAX_ATTEMPTS = 3
_MODEL_TIMEOUT_SECONDS = 30.0


class RetryableExtractionError(RuntimeError):
    """Base error for provider responses that may succeed on a fresh attempt."""


class IncompleteModelOutputError(RetryableExtractionError):
    """Raised when the provider reports truncated, filtered, or otherwise incomplete output."""


class StructuredOutputValidationError(RetryableExtractionError):
    """Raised when a complete provider response cannot produce the required schema."""


_TRANSIENT_PROVIDER_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)
_RETRYABLE_EXCEPTIONS = (
    RetryableExtractionError,
    LengthFinishReasonError,
    ContentFilterFinishReasonError,
    *_TRANSIENT_PROVIDER_EXCEPTIONS,
)


@dataclass(slots=True)
class _ProcessingContext:
    correlation_id: str
    started_at: float
    provider_attempt: int = 0


_PROCESSING_CONTEXT: ContextVar[_ProcessingContext | None] = ContextVar(
    "technical_entity_processing_context",
    default=None,
)


def configure_json_logging(level: int = logging.INFO) -> None:
    """Configure a compact standard-library JSON log handler for command-line usage."""
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False


def _log_event(level: int, event: str, **safe_context: object) -> None:
    context = _PROCESSING_CONTEXT.get()
    duration_ms = 0.0 if context is None else (time.perf_counter() - context.started_at) * 1000
    payload: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "level": logging.getLevelName(level).lower(),
        "event": event,
        "correlation_id": "unscoped" if context is None else context.correlation_id,
        "duration_ms": round(duration_ms, 2),
    }
    payload.update(safe_context)
    LOGGER.log(level, json.dumps(payload, separators=(",", ":"), sort_keys=True))


def _begin_provider_attempt() -> int:
    context = _PROCESSING_CONTEXT.get()
    if context is None:
        return 1
    context.provider_attempt += 1
    return context.provider_attempt


def _current_attempt() -> int:
    context = _PROCESSING_CONTEXT.get()
    if context is None or context.provider_attempt == 0:
        return 1
    return context.provider_attempt


def _log_retry(attempt: int, failure_type: str) -> None:
    if attempt < _MAX_ATTEMPTS:
        _log_event(
            logging.INFO,
            "extraction.retry",
            attempt=attempt,
            next_attempt=attempt + 1,
            failure_type=failure_type,
        )


def _raise_retryable(
    error: RetryableExtractionError,
    *,
    event: str,
    attempt: int,
    **safe_context: object,
) -> NoReturn:
    _log_event(
        logging.WARNING,
        event,
        attempt=attempt,
        error_type=type(error).__name__,
        **safe_context,
    )
    _log_retry(attempt, type(error).__name__)
    raise error


def _validate_envelope(envelope: object) -> ExtraccionTecnica:
    """Validate raw completion state before accepting or converting structured data."""
    attempt = _current_attempt()
    if not isinstance(envelope, Mapping):
        _raise_retryable(
            StructuredOutputValidationError("structured output envelope is invalid"),
            event="extraction.validation_failure",
            attempt=attempt,
            failure_reason="invalid_envelope",
        )

    typed_envelope = cast(Mapping[str, object], envelope)
    raw = typed_envelope.get("raw")
    if not isinstance(raw, AIMessage):
        _raise_retryable(
            StructuredOutputValidationError("structured output envelope is missing an AIMessage"),
            event="extraction.validation_failure",
            attempt=attempt,
            failure_reason="missing_raw_message",
        )

    finish_reason_value = raw.response_metadata.get("finish_reason")
    finish_reason = finish_reason_value if isinstance(finish_reason_value, str) else "missing"
    response_status = raw.response_metadata.get("status")
    is_complete = (
        finish_reason in _COMPLETE_FINISH_REASONS
        if finish_reason_value is not None
        else response_status == "completed"
        and raw.response_metadata.get("incomplete_details") is None
    )
    if not is_complete:
        _raise_retryable(
            IncompleteModelOutputError("provider output is incomplete"),
            event="extraction.incomplete_output",
            attempt=attempt,
            finish_reason=finish_reason,
        )

    parsing_error = typed_envelope.get("parsing_error")
    if parsing_error is not None:
        _raise_retryable(
            StructuredOutputValidationError("provider output could not be parsed"),
            event="extraction.validation_failure",
            attempt=attempt,
            failure_reason="parsing_error",
            parser_error_type=type(parsing_error).__name__,
        )

    parsed = typed_envelope.get("parsed")
    if parsed is None:
        _raise_retryable(
            StructuredOutputValidationError("provider output did not contain parsed data"),
            event="extraction.validation_failure",
            attempt=attempt,
            failure_reason="missing_parsed_output",
        )

    if isinstance(parsed, ExtraccionTecnica):
        return parsed

    try:
        return ExtraccionTecnica.model_validate(parsed)
    except ValidationError as error:
        _raise_retryable(
            StructuredOutputValidationError("parsed output failed schema validation"),
            event="extraction.validation_failure",
            attempt=attempt,
            failure_reason="invalid_parsed_output",
            validation_error_count=error.error_count(),
        )


def _build_resilient_extractor(
    raw_structured_model: Runnable[object, object],
) -> Runnable[object, ExtraccionTecnica]:
    """Wrap one structured provider attempt and its validation in a single retry owner."""

    async def invoke_structured_model(input_value: object, config: RunnableConfig) -> object:
        attempt = _begin_provider_attempt()
        try:
            return await raw_structured_model.ainvoke(input_value, config=config)
        except _TRANSIENT_PROVIDER_EXCEPTIONS as error:
            _log_event(
                logging.WARNING,
                "extraction.provider_transient_failure",
                attempt=attempt,
                error_type=type(error).__name__,
            )
            _log_retry(attempt, type(error).__name__)
            raise

    structured_model = RunnableLambda(invoke_structured_model)
    validation = RunnableLambda(_validate_envelope)
    attempt = structured_model | validation
    return attempt.with_retry(
        retry_if_exception_type=_RETRYABLE_EXCEPTIONS,
        stop_after_attempt=_MAX_ATTEMPTS,
        wait_exponential_jitter=True,
    )


def _create_model() -> ChatOpenAI:
    """Create the provider client lazily from environment-backed configuration."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        timeout=_MODEL_TIMEOUT_SECONDS,
        max_retries=0,
    )


def build_chain() -> Runnable[dict[str, str], ExtraccionTecnica]:
    """Build the prompt and resilient structured extractor without import-time provider setup."""
    model = _create_model()
    with_structured_output = cast(
        Callable[..., Runnable[object, object]],
        model.with_structured_output,  # pyright: ignore[reportUnknownMemberType]
    )
    structured_model = with_structured_output(
        ExtraccionTecnica,
        method="json_schema",
        include_raw=True,
    )
    resilient_extractor = _build_resilient_extractor(structured_model)
    return cast(Runnable[dict[str, str], ExtraccionTecnica], PROMPT | resilient_extractor)


async def process_text(text: str) -> ExtraccionTecnica:
    """Extract and validate technical entities from raw text asynchronously.

    Args:
        text: Non-blank raw technical text. The content is never written to logs.

    Returns:
        A fully validated ``ExtraccionTecnica`` instance.

    Raises:
        ValueError: If ``text`` is blank, before a provider client or request is created.
        RetryableExtractionError: If all three structured-output attempts are malformed or
            incomplete.
        OpenAIError: If the provider fails with a terminal or exhausted transient error.
    """
    context = _ProcessingContext(correlation_id=uuid4().hex, started_at=time.perf_counter())
    token = _PROCESSING_CONTEXT.set(context)
    try:
        _log_event(logging.INFO, "extraction.processing_started", input_character_count=len(text))
        if not text.strip():
            _log_event(logging.WARNING, "extraction.input_validation_failure", reason="blank_input")
            raise ValueError("text must not be blank")

        result = await build_chain().ainvoke({"text": text})
        _log_event(
            logging.INFO,
            "extraction.succeeded",
            technology_count=len(result.tecnologias),
            criticality=result.nivel_de_criticidad.value,
        )
        return result
    except Exception as error:
        _log_event(
            logging.ERROR,
            "extraction.terminal_error",
            error_type=type(error).__name__,
        )
        raise
    finally:
        _PROCESSING_CONTEXT.reset(token)
