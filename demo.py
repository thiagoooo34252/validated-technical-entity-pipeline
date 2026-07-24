"""Key-gated asynchronous demonstration of the validated extraction pipeline."""

import asyncio
import os

from dotenv import load_dotenv

from chain import configure_json_logging, process_text

EXAMPLE_TEXT = """
After a Kubernetes node rotation, the Python ingestion workers kept reconnecting to
PostgreSQL while Redis latency alarms oscillated. The API recovered after twelve minutes,
but the logs do not prove whether the connection pool or the network policy was responsible.
""".strip()


async def main() -> None:
    """Load local configuration, run one extraction, and print validated JSON."""
    load_dotenv(override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set; no provider request was made. "
            "Create a local .env from .env.example before running the demo."
        )

    extraction = await process_text(EXAMPLE_TEXT)
    print(extraction.model_dump_json(indent=2))


if __name__ == "__main__":
    configure_json_logging()
    asyncio.run(main())
