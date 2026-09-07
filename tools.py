"""Closed, vetted tool registry.

Agents select tools by name from `TOOL_REGISTRY`; nothing in the compiler or graph
nodes ever grants network, filesystem, or shell access implicitly. MVP ships one
deterministic mock tool so the compiler and its error-recovery loop can be built and
tested without any external API key. Real tools (e.g. web search) get added here later
behind the same `get_tools` interface — no compiler changes required.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from errors import UnknownToolError

_MOCK_FACTS = {
    "capital of france": "Paris",
    "speed of light": "299,792,458 m/s",
    "largest planet": "Jupiter",
}


@tool
def lookup(query: str) -> str:
    """Look up a short factual answer for `query` from a small fixed reference table.

    Deterministic and offline — used as the MVP's only tool so the platform's
    tool-calling and error-recovery paths can be exercised without any external
    service or API key.
    """
    answer = _MOCK_FACTS.get(query.strip().lower())
    if answer is None:
        return f"no match found for '{query}'"
    return answer


TOOL_REGISTRY: dict[str, BaseTool] = {
    "lookup": lookup,
}


def get_tools(names: list[str]) -> list[BaseTool]:
    """Resolve tool names against the registry.

    Raises UnknownToolError (naming every unresolved name, not just the first) instead
    of silently dropping a tool an agent was configured to use.
    """
    unknown = [n for n in names if n not in TOOL_REGISTRY]
    if unknown:
        raise UnknownToolError(unknown)
    return [TOOL_REGISTRY[n] for n in names]
