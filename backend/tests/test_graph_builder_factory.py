import pytest

from agent_schema import LLMConfig
from errors import MissingProviderKeyError
from graph_builder import get_llm


def test_missing_key_raises_typed_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingProviderKeyError) as exc_info:
        get_llm(LLMConfig(provider="google_genai", model="gemini-3.1-pro-preview"))
    assert exc_info.value.provider == "google_genai"


def test_gemini_key_accepted_under_either_env_name(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-construction")
    # Construction must succeed without a network call when a key is present under the
    # GEMINI_API_KEY alias.
    llm = get_llm(LLMConfig(provider="google_genai", model="gemini-3.1-pro-preview"))
    assert llm is not None
