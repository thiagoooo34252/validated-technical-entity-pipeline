"""Shared deterministic, keyless test configuration."""

import pytest


@pytest.fixture(autouse=True)
def disable_provider_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from discovering local provider credentials or dotenv files."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
