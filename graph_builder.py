"""LLM factory and (Phase 5) the AgentPlan -> LangGraph compiler.

`get_llm` is the single point where a provider string becomes a chat model. Adding a
provider means adding a branch here (and an entry in config), never touching the
compiler or the planner. Provider-specific SDK errors (missing key, bad model) are
translated into typed platform errors at this boundary so upstream code handles one
uniform error surface.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from agent_schema import LLMConfig
from config import MOONSHOT_BASE_URL, PROVIDER_ENV_VARS
from errors import MissingProviderKeyError, UnsupportedProviderError

# Providers reachable through LangChain's native init_chat_model. Moonshot is handled
# separately below because it isn't a native init_chat_model provider string — it's an
# OpenAI-compatible endpoint wired via ChatOpenAI against Moonshot's base URL.
_NATIVE_PROVIDERS = {"openai", "anthropic", "groq", "ollama"}


def _require_key(provider: str) -> str:
    env_vars = PROVIDER_ENV_VARS[provider]
    for env_var in env_vars:
        key = os.environ.get(env_var)
        if key:
            return key
    raise MissingProviderKeyError(provider, " or ".join(env_vars))


def get_llm(config: LLMConfig) -> BaseChatModel:
    provider = config.provider

    if provider == "moonshot":
        # Moonshot Kimi: OpenAI-compatible endpoint. extra_params passes through quirks
        # (e.g. reasoning params) without a bespoke schema field per provider.
        from langchain_openai import ChatOpenAI

        key = _require_key(provider)
        return ChatOpenAI(
            model=config.model,
            base_url=MOONSHOT_BASE_URL,
            api_key=key,
            temperature=config.temperature,
            **config.extra_params,
        )

    if provider == "google_genai":
        # The SDK reads GOOGLE_API_KEY from env, but users commonly store the key under
        # GEMINI_API_KEY — resolve it ourselves and pass it explicitly so either works.
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = _require_key(provider)
        return ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=key,
            temperature=config.temperature,
            **config.extra_params,
        )

    if provider in _NATIVE_PROVIDERS:
        from langchain.chat_models import init_chat_model

        if provider in PROVIDER_ENV_VARS:  # ollama has no key requirement
            _require_key(provider)
        return init_chat_model(
            model=config.model,
            model_provider=provider,
            temperature=config.temperature,
            **config.extra_params,
        )

    raise UnsupportedProviderError(provider)
