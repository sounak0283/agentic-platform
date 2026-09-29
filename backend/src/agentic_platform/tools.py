"""Closed, vetted tool registry.

Agents select tools by name from `TOOL_REGISTRY`; nothing in the compiler or graph
nodes ever grants network, filesystem, or shell access implicitly. `TOOL_REGISTRY` maps
each name to a *factory* rather than a pre-built tool instance, so importing this module
never fails just because a tool's API key isn't set — only resolving that specific tool
does (mirrors `graph_builder.get_llm`'s `_require_key` pattern). Real tools get added
here behind the same `get_tools` interface — no compiler changes required.

The network-backed tools call each service's public REST API directly rather than going
through `langchain-community`, which was archived (read-only, no future fixes) in 2026 —
its Wikipedia integration is already broken and its Arxiv one only works pinned to an old
`arxiv` release. A direct call to a documented, stable endpoint is the boring option here.
"""

from __future__ import annotations

import ast
import os
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Callable

import requests
from langchain_core.tools import BaseTool, tool

from .errors import MissingToolKeyError, UnknownToolError

# Wikipedia's API policy expects requests to identify themselves; the rest of the
# services are fine with anything, so one constant covers every outbound call.
_USER_AGENT = "agentic-platform/0.1 (multi-agent tool registry)"
_HEADERS = {"User-Agent": _USER_AGENT}
# Every third-party call is bounded so a slow service can't hang an agent node past the
# run bounds the platform enforces.
_HTTP_TIMEOUT_S = 10.0

_MOCK_FACTS = {
    "capital of france": "Paris",
    "speed of light": "299,792,458 m/s",
    "largest planet": "Jupiter",
}


@tool
def lookup(query: str) -> str:
    """Look up a short factual answer for `query` from a small fixed reference table.

    Deterministic and offline — used as the MVP's original tool so the platform's
    tool-calling and error-recovery paths can be exercised without any external
    service or API key.
    """
    answer = _MOCK_FACTS.get(query.strip().lower())
    if answer is None:
        return f"no match found for '{query}'"
    return answer


_CALC_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_CALC_UNARY_OPS = (ast.UAdd, ast.USub)


def _eval_arithmetic(node: ast.AST) -> float:
    """Evaluate an arithmetic AST, rejecting anything that isn't pure math.

    Deliberately not `eval()`: the input comes from an LLM acting on user text, so names,
    calls, attributes and subscripts must be impossible, not merely discouraged.
    """
    if isinstance(node, ast.Expression):
        return _eval_arithmetic(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"only numbers are allowed, got {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, _CALC_BIN_OPS):
        left, right = _eval_arithmetic(node.left), _eval_arithmetic(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        return left**right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _CALC_UNARY_OPS):
        operand = _eval_arithmetic(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result.

    Supports + - * / // % ** and parentheses over plain numbers, e.g. "(12 * 8) / 4".
    Does not support variables, functions, or units — pass numbers only.
    """
    try:
        parsed = ast.parse(expression, mode="eval")
        return str(_eval_arithmetic(parsed))
    except (ValueError, SyntaxError, TypeError) as e:
        return f"could not evaluate '{expression}': {e}"
    except ZeroDivisionError:
        return f"could not evaluate '{expression}': division by zero"


@tool
def wikipedia(query: str) -> str:
    """Search Wikipedia and return a summary of the best-matching article.

    Useful for background on people, places, organisations, events, and general
    encyclopedic facts.
    """
    search = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json",
        },
        headers=_HEADERS,
        timeout=_HTTP_TIMEOUT_S,
    ).json()
    hits = search.get("query", {}).get("search", [])
    if not hits:
        return f"no Wikipedia article found for '{query}'"

    title = hits[0]["title"]
    summary = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}",
        headers=_HEADERS,
        timeout=_HTTP_TIMEOUT_S,
    ).json()
    extract = summary.get("extract")
    if not extract:
        return f"found the article '{title}' but it has no summary text"
    return f"{title}: {extract}"


_ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


@tool
def arxiv(query: str) -> str:
    """Search arXiv for scientific preprints and return the top matches.

    Covers physics, mathematics, computer science, quantitative biology and finance,
    statistics, and economics. Returns title, authors, date, and abstract snippet.
    """
    response = requests.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "start": 0, "max_results": 3},
        headers=_HEADERS,
        timeout=_HTTP_TIMEOUT_S,
    )
    entries = ET.fromstring(response.content).findall("a:entry", _ARXIV_NS)
    if not entries:
        return f"no arXiv papers found for '{query}'"

    results = []
    for entry in entries:
        title = (entry.findtext("a:title", default="", namespaces=_ARXIV_NS) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=_ARXIV_NS) or "").strip()
        published = entry.findtext("a:published", default="", namespaces=_ARXIV_NS) or ""
        authors = [
            (a.findtext("a:name", default="", namespaces=_ARXIV_NS) or "").strip()
            for a in entry.findall("a:author", _ARXIV_NS)
        ]
        results.append(
            f"Title: {title}\nAuthors: {', '.join(authors)}\n"
            f"Published: {published[:10]}\nAbstract: {summary[:500]}"
        )
    return "\n\n".join(results)


@tool
def pubmed(query: str) -> str:
    """Search PubMed for biomedical and life-sciences literature.

    Useful for questions about medicine, clinical research, biology, and public health.
    Returns the top matching article titles with journal and publication date.
    """
    # NCBI raises the anonymous rate limit if a free API key is supplied; it stays
    # entirely optional, so the tool works with no configuration at all.
    common = {"db": "pubmed", "retmode": "json"}
    api_key = os.environ.get("PUBMED_API_KEY")
    if api_key:
        common["api_key"] = api_key

    search = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={**common, "term": query, "retmax": 3},
        headers=_HEADERS,
        timeout=_HTTP_TIMEOUT_S,
    ).json()
    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return f"no PubMed articles found for '{query}'"

    summaries = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={**common, "id": ",".join(ids)},
        headers=_HEADERS,
        timeout=_HTTP_TIMEOUT_S,
    ).json().get("result", {})

    results = []
    for pmid in ids:
        record = summaries.get(pmid)
        if not record:
            continue
        results.append(
            f"PMID {pmid}: {record.get('title', '(no title)')}\n"
            f"  Source: {record.get('source', 'unknown')} "
            f"({record.get('pubdate', 'unknown date')})"
        )
    return "\n".join(results) if results else f"no PubMed details found for '{query}'"


@tool
def weather(location: str) -> str:
    """Get current weather conditions for a place, given its name (e.g. "Kolkata").

    Returns temperature in Celsius, wind speed, and the observation time.
    """
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1},
        headers=_HEADERS,
        timeout=_HTTP_TIMEOUT_S,
    ).json()
    places = geo.get("results") or []
    if not places:
        return f"could not find a location named '{location}'"

    place = places[0]
    forecast = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current_weather": "true",
        },
        headers=_HEADERS,
        timeout=_HTTP_TIMEOUT_S,
    ).json()
    current = forecast.get("current_weather")
    if not current:
        return f"no current weather available for '{location}'"

    where = ", ".join(p for p in (place.get("name"), place.get("country")) if p)
    return (
        f"{where}: {current['temperature']}°C, wind {current['windspeed']} km/h "
        f"(observed {current.get('time', 'unknown time')})"
    )


@tool
def yahoo_finance_news(ticker: str) -> str:
    """Get recent news headlines for a public company, by stock ticker symbol.

    The input must be a ticker (e.g. "AAPL", "TSLA", "INFY.NS"), not a company name.
    """
    import yfinance

    articles = yfinance.Ticker(ticker).news or []
    if not articles:
        return f"no recent news found for ticker '{ticker}'"

    headlines = []
    for article in articles[:5]:
        # yfinance has moved the headline fields between top-level and a "content"
        # sub-dict across releases; read whichever shape is present.
        content = article.get("content", article)
        title = content.get("title") or "(untitled)"
        publisher = content.get("provider", {}).get("displayName") or content.get(
            "publisher", "unknown source"
        )
        headlines.append(f"- {title} ({publisher})")
    return f"Recent news for {ticker}:\n" + "\n".join(headlines)


def _build_web_search() -> BaseTool:
    from langchain_tavily import TavilySearch

    if not os.environ.get("TAVILY_API_KEY"):
        raise MissingToolKeyError("web_search", "TAVILY_API_KEY")
    return TavilySearch(max_results=5)


def _build_wolfram_alpha() -> BaseTool:
    appid = os.environ.get("WOLFRAM_ALPHA_APPID")
    if not appid:
        raise MissingToolKeyError("wolfram_alpha", "WOLFRAM_ALPHA_APPID")

    @tool
    def wolfram_alpha(query: str) -> str:
        """Answer a computational, mathematical, scientific, or factual question.

        Handles symbolic maths, unit conversions, statistics, and curated data
        (e.g. "integrate x^2 sin x", "population of Japan in 1990").
        """
        response = requests.get(
            "https://api.wolframalpha.com/v1/result",
            params={"appid": appid, "i": query},
            headers=_HEADERS,
            timeout=_HTTP_TIMEOUT_S,
        )
        if response.status_code == 501:
            # Wolfram's documented "no short answer available" status, not a failure.
            return f"Wolfram Alpha could not produce a short answer for '{query}'"
        response.raise_for_status()
        return response.text

    return wolfram_alpha


# Tools with real-world side effects — network egress to a third party, whether or not
# they need an API key. The planner never auto-attaches these: see `default_tool_names()`
# and its use in app.py::create_project. A caller must name one explicitly in
# `available_tools` to opt a project into it.
SIDE_EFFECT_TOOLS: frozenset[str] = frozenset(
    {
        "web_search",
        "wikipedia",
        "arxiv",
        "pubmed",
        "weather",
        "yahoo_finance_news",
        "wolfram_alpha",
    }
)

TOOL_REGISTRY: dict[str, Callable[[], BaseTool]] = {
    "lookup": lambda: lookup,
    "calculator": lambda: calculator,
    "web_search": _build_web_search,
    "wikipedia": lambda: wikipedia,
    "arxiv": lambda: arxiv,
    "pubmed": lambda: pubmed,
    "weather": lambda: weather,
    "yahoo_finance_news": lambda: yahoo_finance_news,
    "wolfram_alpha": _build_wolfram_alpha,
}


def default_tool_names() -> list[str]:
    """Tool names safe to expose to a project by default: the full registry minus any
    side-effecting tool, which requires explicit per-project opt-in."""
    return [n for n in TOOL_REGISTRY if n not in SIDE_EFFECT_TOOLS]


def get_tools(names: list[str]) -> list[BaseTool]:
    """Resolve tool names against the registry.

    Raises UnknownToolError (naming every unresolved name, not just the first) instead
    of silently dropping a tool an agent was configured to use. Each resolved tool is
    built by calling its factory, so a missing API key surfaces as a clear
    MissingToolKeyError at resolution time (compile time, via graph_builder.compile_graph)
    rather than mid-run.
    """
    unknown = [n for n in names if n not in TOOL_REGISTRY]
    if unknown:
        raise UnknownToolError(unknown)
    return [TOOL_REGISTRY[n]() for n in names]
