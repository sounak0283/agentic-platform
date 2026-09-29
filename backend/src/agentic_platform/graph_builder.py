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
from pydantic import BaseModel, Field

from .agent_schema import AgentOutput, AgentPlan, AgentSpec, LLMConfig, RunBounds
from .config import MOONSHOT_BASE_URL, PROVIDER_ENV_VARS
from .errors import (
    AgentExecutionError,
    MissingProviderKeyError,
    UnsupportedProviderError,
)
from .tools import get_tools

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
    without clobbering siblings' contributions. `halted` uses an OR reducer so multiple
    parallel branches can flag failure in the same superstep without a write conflict; it
    short-circuits the remaining sequential chain once an agent exhausts its retries.
    `next_agent`/`supervisor_steps` are only used by the supervisor pattern (single
    routing node writing them, so no reducer needed).
    """

    input: dict[str, Any]
    outputs: Annotated[dict[str, Any], lambda a, b: {**a, **b}]
    errors: Annotated[list[dict[str, Any]], operator.add]
    halted: Annotated[bool, operator.or_]
    next_agent: str
    supervisor_steps: int


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


def _build_context(
    state: GraphState, spec: AgentSpec, include_all_outputs: bool = False
) -> str:
    parts = [f"TASK INPUT:\n{state['input']}"]
    # Sequential/parallel agents see their declared dependencies; supervisor workers have
    # no depends_on wiring, so they see everything produced so far instead.
    relevant = list(state["outputs"].keys()) if include_all_outputs else spec.depends_on
    shown = [d for d in relevant if d in state["outputs"]]
    if shown:
        header = "\nOUTPUT FROM PRIOR AGENTS:" if include_all_outputs else (
            "\nOUTPUT FROM AGENTS YOU DEPEND ON:"
        )
        parts.append(header)
        for dep in shown:
            dep_out = state["outputs"][dep]
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


def _make_node(spec: AgentSpec, bounds: RunBounds, include_all_outputs: bool = False):
    def node(state: GraphState) -> dict[str, Any]:
        # Dependency gating (the parallel/DAG failure-isolation mechanism): if any agent
        # this node depends on failed and produced no output, skip rather than run on
        # missing input. In sequential mode the halted short-circuit means we never even
        # reach a node with a failed dependency, so this is a no-op there.
        missing = [d for d in spec.depends_on if d not in state["outputs"]]
        if missing:
            skipped = AgentExecutionError(
                spec.id,
                f"agent '{spec.id}' skipped: upstream dependencies did not produce "
                f"output: {missing}",
                attempts=0,
            )
            return {"errors": [skipped.to_dict()], "halted": True}

        llm = get_llm(spec.llm)
        base_messages: list[AnyMessage] = [
            SystemMessage(content=spec.system_prompt),
            HumanMessage(content=_build_context(state, spec, include_all_outputs)),
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


class _SupervisorDecision(BaseModel):
    """Structured routing decision the supervisor emits each turn."""

    next_agent: str = Field(
        description="id of the worker to run next, or 'FINISH' when the task is complete"
    )
    reason: str = Field(default="", description="brief justification for the choice")


def compile_graph(plan: AgentPlan):
    """Compile a validated AgentPlan into a runnable LangGraph graph.

    Dispatches on orchestration pattern. All three share the same node implementation,
    state shape, and error-recovery loop — only the wiring differs:
      - sequential: a dependency-ordered chain, halting the rest on a node failure.
      - parallel:   the real dependency DAG, so independent agents run concurrently and
                    merge at agents that depend on them; failures are isolated per branch.
      - supervisor: a router node picks the next worker at runtime, looping until it
                    signals FINISH or a bounded iteration cap is hit.
    """
    # Fail fast at compile time if any agent references an unknown tool, rather than
    # discovering it mid-run.
    for spec in plan.agents:
        get_tools(spec.tools)

    if plan.orchestration_pattern == "sequential":
        return _compile_sequential(plan)
    if plan.orchestration_pattern == "parallel":
        return _compile_parallel(plan)
    if plan.orchestration_pattern == "supervisor":
        return _compile_supervisor(plan)
    raise NotImplementedError(
        f"orchestration pattern '{plan.orchestration_pattern}' is not supported"
    )


def _compile_sequential(plan: AgentPlan):
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

        builder.add_conditional_edges(
            spec.id, route, {END: END} | ({} if is_last else {next_node: next_node})
        )

    return builder.compile()


def _compile_parallel(plan: AgentPlan):
    """Wire the dependency DAG directly: agents with no deps fan out from START and run
    concurrently; agents with deps run once their deps complete (the merge/synthesis
    step). Failure isolation is per-branch via the node's dependency gating, not a global
    halt — an independent worker failing doesn't stop its siblings."""
    has_dependents: set[str] = set()
    for spec in plan.agents:
        has_dependents.update(spec.depends_on)

    builder = StateGraph(GraphState)
    for spec in plan.agents:
        builder.add_node(spec.id, _make_node(spec, plan.bounds))

    for spec in plan.agents:
        if not spec.depends_on:
            builder.add_edge(START, spec.id)
        else:
            for dep in spec.depends_on:
                builder.add_edge(dep, spec.id)
        if spec.id not in has_dependents:
            builder.add_edge(spec.id, END)

    return builder.compile()


def _make_supervisor(sup_spec: AgentSpec, worker_ids: list[str], cap: int):
    routing_instructions = (
        "\n\nYou are the ROUTER for a team of worker agents. Based on the task and the "
        "outputs produced so far, choose the single worker that should act next, or "
        "return 'FINISH' when the task is complete. You must pick from exactly these "
        f"worker ids: {worker_ids}. Do not invent ids."
    )

    def node(state: GraphState) -> dict[str, Any]:
        steps = state.get("supervisor_steps", 0) + 1
        if steps > cap:
            # Bounded: never let the router loop forever (principle: every run is bounded).
            note = AgentExecutionError(
                sup_spec.id,
                f"supervisor stopped after hitting its {cap}-iteration cap",
                attempts=cap,
            )
            return {"supervisor_steps": steps, "next_agent": "FINISH", "errors": [note.to_dict()]}

        llm = get_llm(sup_spec.llm)
        messages: list[AnyMessage] = [
            SystemMessage(content=sup_spec.system_prompt + routing_instructions),
            HumanMessage(content=_build_context(state, sup_spec, include_all_outputs=True)),
        ]
        try:
            decision = llm.with_structured_output(_SupervisorDecision).invoke(messages)
            nxt = decision.next_agent.strip()
            if nxt not in worker_ids and nxt != "FINISH":
                nxt = "FINISH"  # invalid routing target -> finish safely rather than loop
        except Exception as e:  # noqa: BLE001 — a router failure must end the run cleanly
            note = AgentExecutionError(
                sup_spec.id, f"supervisor routing failed: {e}", attempts=1
            )
            return {"supervisor_steps": steps, "next_agent": "FINISH", "errors": [note.to_dict()]}

        return {"supervisor_steps": steps, "next_agent": nxt}

    return node


def _compile_supervisor(plan: AgentPlan):
    workers = [a for a in plan.agents if a.id != plan.supervisor_id]
    worker_ids = [w.id for w in workers]
    sup_spec = next(a for a in plan.agents if a.id == plan.supervisor_id)
    cap = max(plan.bounds.max_steps, 2 * len(workers) + 2)

    builder = StateGraph(GraphState)
    builder.add_node(sup_spec.id, _make_supervisor(sup_spec, worker_ids, cap))
    for w in workers:
        # Supervisor workers see all accumulated outputs (they carry no depends_on wiring).
        builder.add_node(w.id, _make_node(w, plan.bounds, include_all_outputs=True))
        builder.add_edge(w.id, sup_spec.id)  # control returns to the router after each worker

    builder.add_edge(START, sup_spec.id)

    def route(state: GraphState):
        nxt = state.get("next_agent")
        return END if nxt in (None, "FINISH") else nxt

    builder.add_conditional_edges(
        sup_spec.id, route, {wid: wid for wid in worker_ids} | {END: END}
    )

    return builder.compile()


def initial_state(task_input: dict[str, Any]) -> GraphState:
    return {
        "input": task_input,
        "outputs": {},
        "errors": [],
        "halted": False,
        "next_agent": "",
        "supervisor_steps": 0,
    }


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
