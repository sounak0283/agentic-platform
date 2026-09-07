"""LLM factory and (Phase 5) the AgentPlan -> LangGraph compiler.

`get_llm` is the single point where a provider string becomes a chat model. Adding a
provider means adding a branch here (and an entry in config), never touching the
compiler or the planner. Provider-specific SDK errors (missing key, bad model) are
translated into typed platform errors at this boundary so upstream code handles one
uniform error surface.
"""

from __future__ import annotations

import operator
import os
import time
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph

from agent_schema import AgentOutput, AgentPlan, AgentSpec, LLMConfig, RunBounds
from config import MOONSHOT_BASE_URL, PROVIDER_ENV_VARS
from errors import (
    AgentExecutionError,
    MissingProviderKeyError,
    UnsupportedProviderError,
)
from tools import get_tools

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


# ---------------------------------------------------------------------------
# Graph compiler
# ---------------------------------------------------------------------------

_NODE_BACKOFF_S = 0.5


class GraphState(TypedDict):
    """Shared state threaded through the compiled graph.

    `outputs` and `errors` use merge reducers so each node contributes its own slice
    without clobbering siblings' contributions. `halted` short-circuits the remaining
    sequential chain once an agent has exhausted its retries.
    """

    input: dict[str, Any]
    outputs: Annotated[dict[str, Any], lambda a, b: {**a, **b}]
    errors: Annotated[list[dict[str, Any]], operator.add]
    halted: bool


def _text(content: Any) -> str:
    """Flatten message content to text, handling providers (e.g. Gemini 3.x) that return
    a list of content blocks rather than a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _build_context(state: GraphState, spec: AgentSpec) -> str:
    parts = [f"TASK INPUT:\n{state['input']}"]
    if spec.depends_on:
        parts.append("\nOUTPUT FROM AGENTS YOU DEPEND ON:")
        for dep in spec.depends_on:
            dep_out = state["outputs"].get(dep)
            if dep_out is not None:
                parts.append(f"  [{dep}]: {dep_out.get('content', dep_out)}")
    return "\n".join(parts)


def _run_agent_once(
    llm: BaseChatModel,
    spec: AgentSpec,
    base_messages: list[AnyMessage],
    bounds: RunBounds,
) -> AgentOutput:
    """One full attempt for an agent node: bounded tool loop, then structured output.

    Any exception here (tool error, schema-validation failure) propagates to the caller's
    retry loop, which feeds the error text back into the agent's own context.
    """
    messages = list(base_messages)
    tools = get_tools(spec.tools)

    if tools:
        bound = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}
        for _ in range(bounds.max_steps):
            ai = bound.invoke(messages)
            messages.append(ai)
            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                break
            for tc in tool_calls:
                tool = tool_map[tc["name"]]  # names validated at compile time
                result = tool.invoke(tc["args"])  # may raise -> caught by retry loop
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

    structured = llm.with_structured_output(AgentOutput)
    messages.append(
        HumanMessage(content="Now produce your final structured answer for this task.")
    )
    return structured.invoke(messages)


def _make_node(spec: AgentSpec, bounds: RunBounds):
    def node(state: GraphState) -> dict[str, Any]:
        llm = get_llm(spec.llm)
        base_messages: list[AnyMessage] = [
            SystemMessage(content=spec.system_prompt),
            HumanMessage(content=_build_context(state, spec)),
        ]

        last_error: str | None = None
        for attempt in range(1, bounds.max_retries + 1):
            attempt_messages = list(base_messages)
            if last_error is not None:
                # Feed the actual error back into THIS agent's context, not a generic
                # "try again" — the recovery signal the SYSTEM_PROMPT loop requires.
                attempt_messages.append(
                    HumanMessage(
                        content=(
                            "Your previous attempt failed with this error. Correct it "
                            f"and try again:\n{last_error}"
                        )
                    )
                )
            try:
                output = _run_agent_once(llm, spec, attempt_messages, bounds)
                return {"outputs": {spec.id: output.model_dump()}}
            except Exception as e:  # noqa: BLE001 — deliberate: recover from any node failure
                last_error = f"{type(e).__name__}: {e}"
                if attempt < bounds.max_retries:
                    time.sleep(_NODE_BACKOFF_S)

        # Ceiling exhausted: record a structured failure and halt the chain rather than
        # raising, so one agent's failure never crashes the whole graph run silently.
        failure = AgentExecutionError(
            spec.id,
            f"agent '{spec.id}' failed after {bounds.max_retries} attempts: {last_error}",
            attempts=bounds.max_retries,
        )
        return {"errors": [failure.to_dict()], "halted": True}

    return node


def _topological_order(agents: list[AgentSpec]) -> list[AgentSpec]:
    by_id = {a.id: a for a in agents}
    indegree = {a.id: 0 for a in agents}
    dependents: dict[str, list[str]] = {a.id: [] for a in agents}
    for a in agents:
        for dep in a.depends_on:
            indegree[a.id] += 1
            dependents[dep].append(a.id)

    # Preserve the planner's agent ordering among ready nodes for a stable, readable chain.
    ready = [a.id for a in agents if indegree[a.id] == 0]
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for nxt in dependents[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    return [by_id[i] for i in order]


def compile_graph(plan: AgentPlan):
    """Compile a validated AgentPlan into a runnable LangGraph graph.

    MVP wires the `sequential` pattern only: agents run one at a time in dependency
    order, each seeing its predecessors' outputs. The state shape (per-agent `outputs`,
    structured `errors`, `halted`) is intentionally pattern-agnostic so parallel and
    supervisor wiring can be added later without reworking the node or state design.
    """
    if plan.orchestration_pattern != "sequential":
        raise NotImplementedError(
            f"orchestration pattern '{plan.orchestration_pattern}' is not yet supported "
            "(MVP ships sequential only)"
        )

    # Fail fast at compile time if any agent references an unknown tool, rather than
    # discovering it mid-run.
    for spec in plan.agents:
        get_tools(spec.tools)

    ordered = _topological_order(plan.agents)
    builder = StateGraph(GraphState)
    for spec in ordered:
        builder.add_node(spec.id, _make_node(spec, plan.bounds))

    builder.add_edge(START, ordered[0].id)
    for i, spec in enumerate(ordered):
        is_last = i == len(ordered) - 1
        next_node = END if is_last else ordered[i + 1].id

        def route(state: GraphState, _next=next_node):
            return END if state.get("halted") else _next

        builder.add_conditional_edges(spec.id, route, {END: END} | (
            {} if is_last else {next_node: next_node}
        ))

    return builder.compile()


def initial_state(task_input: dict[str, Any]) -> GraphState:
    return {"input": task_input, "outputs": {}, "errors": [], "halted": False}


def run_single_agent(
    plan: AgentPlan,
    agent_id: str,
    task_input: dict[str, Any],
    upstream_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one agent node in isolation (the per-agent endpoint path).

    `upstream_outputs` lets the caller supply the outputs this agent's `depends_on`
    would normally provide, without running the whole graph.
    """
    spec = next((a for a in plan.agents if a.id == agent_id), None)
    if spec is None:
        raise KeyError(agent_id)
    node = _make_node(spec, plan.bounds)
    state = initial_state(task_input)
    state["outputs"] = dict(upstream_outputs or {})
    return node(state)


def run_graph(graph, task_input: dict[str, Any], bounds: RunBounds) -> dict[str, Any]:
    """Invoke a compiled graph with a wall-clock timeout bound (principle: every run is
    bounded, not left to model good behavior)."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            graph.invoke,
            initial_state(task_input),
            {"recursion_limit": 50},
        )
        try:
            return future.result(timeout=bounds.timeout_s)
        except concurrent.futures.TimeoutError as e:
            raise TimeoutError(
                f"graph run exceeded {bounds.timeout_s}s timeout"
            ) from e
