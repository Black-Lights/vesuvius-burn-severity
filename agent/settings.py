"""Which model to call, read from .env. Standard library only, so the notebook can tell whether a
live run is possible without the agent packages installed."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

# name: (base URL, key variable, default model); None is OpenAI's own endpoint
PROVIDERS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-flash"),  # V4.1 Flash
    "kimi": ("https://api.moonshot.ai/v1", "KIMI_API_KEY", "kimi-k3"),
    "openai": (None, "OPENAI_API_KEY", "gpt-5.4-mini"),
}


def load_env() -> None:
    """Read .env into the environment, without overriding what is already set."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                os.environ.setdefault(name.strip(), value.strip())


def setting(name: str) -> str | None:
    """A value from the environment, without an inline comment; None when empty.

    `LLM_MODEL=   # a note` must read as empty, not as the note, so the comment is cut here.
    """
    value = os.environ.get(name, "").split("#")[0].strip().strip("'\"")
    return value or None


def endpoint(provider: str | None = None) -> tuple[str, str | None, str | None, str | None]:
    """(provider, base URL, key, default model) for the provider named, in LLM_PROVIDER or DeepSeek."""
    load_env()
    provider = provider or setting("LLM_PROVIDER") or "deepseek"
    if provider == "custom":
        return provider, setting("LLM_BASE_URL"), setting("LLM_API_KEY"), None
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {', '.join([*PROVIDERS, 'custom'])}, got {provider!r}")
    base_url, key_var, default = PROVIDERS[provider]
    return provider, base_url, setting(key_var), default


def why_not_live(provider: str | None = None) -> str | None:
    """None when a live run can start; otherwise the reason it cannot."""
    if importlib.util.find_spec("langgraph") is None:
        return "the agent packages are not installed (pip install -r requirements-agent.txt)"
    provider, _, key, _ = endpoint(provider)
    return None if key else f"no API key for {provider} in .env"
