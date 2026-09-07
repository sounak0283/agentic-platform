import pytest

from errors import UnknownToolError
from tools import get_tools, lookup


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
        get_tools(["lookup", "web_search", "shell"])
    assert exc_info.value.names == ["web_search", "shell"]
