import pytest

import agentic_platform.tools as tools_module
from agentic_platform.errors import MissingToolKeyError, UnknownToolError
from agentic_platform.tools import calculator, default_tool_names, get_tools, lookup


class _FakeResponse:
    """Stands in for a requests.Response so tool tests never touch the network."""

    def __init__(self, json_data=None, content=b"", text="", status_code=200):
        self._json = json_data
        self.content = content
        self.text = text
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected status {self.status_code}")


def _fake_get(monkeypatch, responses):
    """Patch requests.get to return canned responses in order, recording call params."""
    calls = []
    queue = list(responses)

    def fake(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params or {}})
        return queue.pop(0)

    monkeypatch.setattr(tools_module.requests, "get", fake)
    return calls


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
    # Only the two purely-local tools are available without an explicit opt-in.
    assert set(names) == {"lookup", "calculator"}
    for networked in ("web_search", "wikipedia", "arxiv", "pubmed", "weather",
                      "yahoo_finance_news", "wolfram_alpha"):
        assert networked not in names


# ------------------------------------------------------------------------- calculator
def test_calculator_evaluates_expression():
    assert calculator.invoke({"expression": "(12 * 8) / 4"}) == "24.0"


def test_calculator_respects_precedence_and_power():
    assert calculator.invoke({"expression": "2 + 3 * 4 ** 2"}) == "50"


def test_calculator_rejects_code_execution():
    result = calculator.invoke({"expression": "__import__('os').system('echo pwned')"})
    assert "could not evaluate" in result
    assert "Call" in result  # rejected at the AST level, never executed


def test_calculator_rejects_names():
    assert "could not evaluate" in calculator.invoke({"expression": "some_var + 1"})


def test_calculator_handles_division_by_zero():
    assert "division by zero" in calculator.invoke({"expression": "1/0"})


# --------------------------------------------------------------------- networked tools
def test_wikipedia_searches_then_summarises(monkeypatch):
    calls = _fake_get(
        monkeypatch,
        [
            _FakeResponse({"query": {"search": [{"title": "Alan Turing"}]}}),
            _FakeResponse({"extract": "British mathematician."}),
        ],
    )
    result = get_tools(["wikipedia"])[0].invoke({"query": "turing"})
    assert result == "Alan Turing: British mathematician."
    assert calls[0]["params"]["srsearch"] == "turing"


def test_wikipedia_no_results(monkeypatch):
    _fake_get(monkeypatch, [_FakeResponse({"query": {"search": []}})])
    result = get_tools(["wikipedia"])[0].invoke({"query": "asdfqwerty"})
    assert "no Wikipedia article found" in result


_ARXIV_ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>A Test Paper</title>
    <summary>An abstract about testing.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
  </entry>
</feed>"""


def test_arxiv_parses_atom_feed(monkeypatch):
    _fake_get(monkeypatch, [_FakeResponse(content=_ARXIV_ATOM)])
    result = get_tools(["arxiv"])[0].invoke({"query": "testing"})
    assert "A Test Paper" in result
    assert "Ada Lovelace" in result
    assert "2024-01-15" in result


def test_pubmed_searches_then_summarises(monkeypatch):
    calls = _fake_get(
        monkeypatch,
        [
            _FakeResponse({"esearchresult": {"idlist": ["12345"]}}),
            _FakeResponse(
                {"result": {"12345": {"title": "A Study", "source": "Nature",
                                      "pubdate": "2024"}}}
            ),
        ],
    )
    result = get_tools(["pubmed"])[0].invoke({"query": "vaccines"})
    assert "PMID 12345: A Study" in result
    assert "Nature" in result
    assert "api_key" not in calls[0]["params"]  # optional key omitted when unset


def test_pubmed_passes_optional_api_key(monkeypatch):
    monkeypatch.setenv("PUBMED_API_KEY", "ncbi-key")
    calls = _fake_get(
        monkeypatch,
        [
            _FakeResponse({"esearchresult": {"idlist": ["1"]}}),
            _FakeResponse({"result": {"1": {"title": "T"}}}),
        ],
    )
    get_tools(["pubmed"])[0].invoke({"query": "x"})
    assert calls[0]["params"]["api_key"] == "ncbi-key"


def test_weather_geocodes_then_fetches_forecast(monkeypatch):
    _fake_get(
        monkeypatch,
        [
            _FakeResponse(
                {"results": [{"latitude": 22.5, "longitude": 88.3,
                              "name": "Kolkata", "country": "India"}]}
            ),
            _FakeResponse(
                {"current_weather": {"temperature": 28.3, "windspeed": 7.4,
                                     "time": "2026-09-10T09:45"}}
            ),
        ],
    )
    result = get_tools(["weather"])[0].invoke({"location": "Kolkata"})
    assert "Kolkata, India" in result
    assert "28.3" in result


def test_weather_unknown_location(monkeypatch):
    _fake_get(monkeypatch, [_FakeResponse({"results": []})])
    assert "could not find a location" in get_tools(["weather"])[0].invoke(
        {"location": "zzzz"}
    )


def test_yahoo_finance_news_formats_headlines(monkeypatch):
    class _FakeTicker:
        def __init__(self, symbol):
            self.news = [
                {"content": {"title": "Stock rises",
                             "provider": {"displayName": "Reuters"}}}
            ]

    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    result = get_tools(["yahoo_finance_news"])[0].invoke({"ticker": "AAPL"})
    assert "Stock rises" in result
    assert "Reuters" in result


def test_wolfram_alpha_missing_key_raises(monkeypatch):
    monkeypatch.delenv("WOLFRAM_ALPHA_APPID", raising=False)
    with pytest.raises(MissingToolKeyError) as exc_info:
        get_tools(["wolfram_alpha"])
    assert exc_info.value.tool_name == "wolfram_alpha"
    assert exc_info.value.env_var == "WOLFRAM_ALPHA_APPID"


def test_wolfram_alpha_returns_short_answer(monkeypatch):
    monkeypatch.setenv("WOLFRAM_ALPHA_APPID", "fake-appid")
    calls = _fake_get(monkeypatch, [_FakeResponse(text="4")])
    result = get_tools(["wolfram_alpha"])[0].invoke({"query": "2+2"})
    assert result == "4"
    assert calls[0]["params"]["appid"] == "fake-appid"


def test_wolfram_alpha_handles_no_short_answer(monkeypatch):
    monkeypatch.setenv("WOLFRAM_ALPHA_APPID", "fake-appid")
    _fake_get(monkeypatch, [_FakeResponse(text="No short answer", status_code=501)])
    result = get_tools(["wolfram_alpha"])[0].invoke({"query": "meaning of life"})
    assert "could not produce a short answer" in result


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
