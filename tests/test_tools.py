import pytest

from errors import MissingToolKeyError, UnknownToolError
from tools import default_tool_names, get_tools, lookup


def test_lookup_known_fact():
    assert lookup.invoke({"query": "capital of France"}) == "Paris"


def test_lookup_unknown_fact_is_deterministic_miss():
    result = lookup.invoke({"query": "meaning of life"})
    assert "no match found" in result


def test_get_tools_resolves_known_names():
    resolved = get_tools(["lookup"])
    assert len(resolved) == 1
    assert resolved[0].name == "lookup"


def test_get_tools_rejects_unknown_names():
    with pytest.raises(UnknownToolError) as exc_info:
        get_tools(["lookup", "shell", "nonexistent"])
    assert exc_info.value.names == ["shell", "nonexistent"]


def test_default_tool_names_excludes_side_effect_tools():
    names = default_tool_names()
    assert "lookup" in names
    assert "web_search" not in names


def test_web_search_missing_key_raises(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(MissingToolKeyError) as exc_info:
        get_tools(["web_search"])
    assert exc_info.value.tool_name == "web_search"
    assert exc_info.value.env_var == "TAVILY_API_KEY"


def test_web_search_resolves_with_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-test")
    sentinel = object()
    monkeypatch.setattr(
        "langchain_tavily.TavilySearch", lambda **kwargs: sentinel
    )
    resolved = get_tools(["web_search"])
    assert resolved == [sentinel]
