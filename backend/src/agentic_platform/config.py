"""Central config: env loading and per-provider settings.

Loading .env here (once, at import) means every module gets the same environment
without each one calling load_dotenv. Provider metadata lives here so the LLM factory
stays a thin dispatch over data, not a pile of hardcoded strings.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

# Provider used by default when a brief doesn't name one.
DEFAULT_PROVIDER = "google_genai"

# Default model per provider, used by the meta-planner when a brief doesn't name one.
DEFAULT_MODELS: dict[str, str] = {
    "moonshot": "kimi-k3",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
    "google_genai": "gemini-3.1-pro-preview",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1",
}

# Env var(s) that may hold the API key for each provider, tried in order. The first
# non-empty one wins. google_genai accepts GEMINI_API_KEY as well as the SDK's native
# GOOGLE_API_KEY, since users commonly have the key under either name.
PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "moonshot": ("MOONSHOT_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google_genai": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    # ollama runs locally and needs no key.
}

MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
