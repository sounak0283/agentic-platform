"""Closed, vetted tool registry.

Agents select tools by name from `TOOL_REGISTRY`; nothing in the compiler or graph
nodes ever grants network, filesystem, or shell access implicitly. `TOOL_REGISTRY` maps
each name to a *factory* rather than a pre-built tool instance, so importing this module
never fails just because a tool's API key isn't set — only resolving that specific tool
does (mirrors `graph_builder.get_llm`'s `_require_key` pattern). Real tools get added
here behind the same `get_tools` interface — no compiler changes required.
"""

from __future__ import annotations

import os
from typing import Callable

from langchain_core.tools import BaseTool, tool

from errors import MissingToolKeyError, UnknownToolError

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


def _build_web_search() -> BaseTool:
    from langchain_tavily import TavilySearch

    if not os.environ.get("TAVILY_API_KEY"):
        raise MissingToolKeyError("web_search", "TAVILY_API_KEY")
    return TavilySearch(max_results=5)


# Tools with real-world side effects (network egress, third-party API calls). The
# planner never auto-attaches these — see `default_tool_names()` and its use in
# app.py::create_project. A caller must name one explicitly in `available_tools` to
# opt a project into it.
SIDE_EFFECT_TOOLS: frozenset[str] = frozenset({"web_search"})

TOOL_REGISTRY: dict[str, Callable[[], BaseTool]] = {
    "lookup": lambda: lookup,
    "web_search": _build_web_search,
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
