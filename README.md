# Validated Technical Entity Pipeline

A small production-oriented Python 3.12 pipeline that receives raw technical text and returns
a validated object. It combines OpenAI native structured output, LangChain Expression Language
(LCEL), explicit completion-state validation, bounded retries, and Pydantic domain constraints.

The two primary implementation files are immediately available at the repository root:

- [`schemas.py`](schemas.py): domain enum, normalization, constraints, and output model.
- [`chain.py`](chain.py): prompt, lazy model configuration, envelope validation, retry policy,
  structured JSON logging, and the async `process_text` API.

## Architecture

```text
raw text
   |
ChatPromptTemplate
   |
ChatOpenAI.with_structured_output(include_raw=True)
   |
finish_reason + parsing + Pydantic validation
   |
ExtraccionTecnica
```

`chain.py` composes `PROMPT | resilient_extractor`. The resilient extractor wraps one complete
provider attempt: native structured generation followed by raw-envelope validation. Its LCEL
`.with_retry()` policy re-executes both steps up to three times with exponential jitter.

Retry ownership is deliberately singular. `ChatOpenAI` uses `max_retries=0`; the outer LCEL
wrapper retries only malformed/incomplete structured results and transient connection, timeout,
rate-limit, or server failures. Authentication and bad-request failures are not broadly retried.

## Output Contract

`ExtraccionTecnica` has exactly three public fields:

- `tecnologias: list[str]`: non-empty, normalized, deduplicated in first-seen order.
- `nivel_de_criticidad`: `baja`, `media`, or `alta`.
- `resumen_tecnico: str`: stripped, non-empty, and length-bounded.

### Exact Example

Input:

```text
Durante el despliegue, los workers de Python perdieron conexión con PostgreSQL y la API
quedó indisponible durante doce minutos.
```

Output shape:

```json
{
  "tecnologias": [
    "Python",
    "PostgreSQL"
  ],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "Los workers de Python perdieron conexión con PostgreSQL y provocaron una indisponibilidad temporal de la API."
}
```

## Setup

Prerequisites: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --all-groups
cp .env.example .env
```

Set `OPENAI_API_KEY` only in the local `.env`. The default model is `gpt-4o-mini`; override it
with `OPENAI_MODEL` when required. `.env` and related local environment files are ignored.

## Run

```bash
uv run python demo.py
```

`demo.py` uses `asyncio.run`, loads `.env` without overriding existing environment variables,
submits one realistic ambiguous incident description, and prints `model_dump_json(indent=2)`.
Without an API key it exits before constructing or contacting a provider and states that no
request was made.

Library use is asynchronous:

```python
import asyncio

from chain import process_text


async def main() -> None:
    result = await process_text("A Python API writes operational state to PostgreSQL.")
    print(result.model_dump_json(indent=2))


asyncio.run(main())
```

## Resilience

`with_structured_output(ExtraccionTecnica, method="json_schema", include_raw=True)` exposes the
raw `AIMessage`, parsed value, and parsing error. Validation checks the raw
response metadata before converting parsed data:

- Chat Completions `finish_reason` remains authoritative; `stop` and `tool_calls` are complete.
- Without it, Responses metadata is complete only for `status="completed"` with no
  `incomplete_details`; incomplete, failed, non-terminal, and unsupported states are rejected.
- Parsing errors, missing parsed output, and Pydantic-invalid parsed output are retryable.
- OpenAI length and content-filter parse exceptions raised before an envelope are retryable.
- After three failed attempts, the typed retryable exception reaches the caller.

Because the retry wrapper surrounds generation and validation together, a rejected envelope
causes a fresh model call. Tests use local Runnables and `AIMessage` envelopes to prove recovery
and exact three-attempt exhaustion without network access.

## Structured Logging

The standard-library logger emits compact JSON event records containing:

- stable event names;
- a per-call correlation ID;
- elapsed `duration_ms`;
- attempt and failure categories where applicable;
- safe counts and criticality on success.

Events cover processing start, input/structured validation failures, incomplete output, retry,
success, and terminal failure. Raw input, provider payloads, exception messages, API keys, and
other secrets are intentionally excluded.

## Verification

All tests are deterministic and keyless. Disable dotenv explicitly when reproducing CI:

```bash
PYTHON_DOTENV_DISABLED=1 OPENAI_API_KEY= uv lock --check
PYTHON_DOTENV_DISABLED=1 OPENAI_API_KEY= uv run pytest --cov=chain --cov=schemas --cov-report=term-missing --cov-fail-under=90
PYTHON_DOTENV_DISABLED=1 OPENAI_API_KEY= uv run pytest tests/test_chain.py -q
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run python -m build
git diff --check
```

## Assignment Checklist

- Exact Spanish schema fields and enum values with Pydantic validation.
- Modular `ChatPromptTemplate` and declarative LCEL composition.
- Lazy, environment-backed `ChatOpenAI` using native JSON Schema output.
- Raw completion-state inspection before parsed output acceptance.
- Bounded retries around provider generation plus validation.
- Async `process_text(text: str)` using `.ainvoke({"text": text})`.
- Structured, correlation-aware, secret-safe JSON logs.
- Key-gated async demo and deterministic provider-free tests.
- Python 3.12 packaging, locked dependencies, Ruff, Pyright, pytest, coverage, and CI.

## Scope And Non-Goals

This repository intentionally contains no PDF/report generation, fallback model, LangSmith
integration, live-provider test, deployment configuration, persistence layer, or GitHub
publication workflow. Entity extraction quality still depends on the selected model and prompt;
the pipeline guarantees output shape and retry behavior, not factual correctness beyond the
provided source text.
