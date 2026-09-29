from langchain_core.tools import tool

import agentic_platform.graph_builder as gb
import agentic_platform.tools as tools_module
from agentic_platform.agent_schema import AgentOutput, AgentPlan, AgentSpec, LLMConfig, RunBounds
from agentic_platform.graph_builder import _SupervisorDecision, compile_graph, initial_state


def _spec(id_, model, depends_on=None):
    return AgentSpec(
        id=id_,
        role="worker",
        system_prompt=f"agent {id_}",
        llm=LLMConfig(provider="google_genai", model=model),
        depends_on=depends_on or [],
    )


class _Structured:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)

    def invoke(self, messages):
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeLLM:
    def __init__(self, structured):
        self._structured = structured

    def with_structured_output(self, schema):
        return self._structured

    def bind_tools(self, tools):  # not exercised by these no-tool agents
        return self


def _patch(monkeypatch, by_model):
    monkeypatch.setattr(gb, "get_llm", lambda cfg: _FakeLLM(by_model[cfg.model]))
    monkeypatch.setattr(gb.time, "sleep", lambda s: None)


def test_failed_agent_halts_chain_and_records_structured_failure(monkeypatch):
    # agent 'a' always fails; 'b' depends on it and must never run.
    b_structured = _Structured([AgentOutput(content="should not appear")])
    _patch(
        monkeypatch,
        {
            "fail-model": _Structured([ValueError("boom")]),
            "ok-model": b_structured,
        },
    )
    plan = AgentPlan(
        agents=[_spec("a", "fail-model"), _spec("b", "ok-model", depends_on=["a"])],
        orchestration_pattern="sequential",
        bounds=RunBounds(max_retries=3),
    )
    graph = compile_graph(plan)
    result = graph.invoke(initial_state({"q": "x"}))

    assert result["halted"] is True
    assert len(result["errors"]) == 1
    assert result["errors"][0]["agent_id"] == "a"
    assert result["errors"][0]["attempts"] == 3
    assert "b" not in result["outputs"]  # downstream agent did not run


def test_agent_recovers_after_transient_failure(monkeypatch):
    # 'a' fails once, then succeeds — the retry loop should feed the error back and pass.
    _patch(
        monkeypatch,
        {"flaky": _Structured([ValueError("transient"), AgentOutput(content="recovered")])},
    )
    plan = AgentPlan(
        agents=[_spec("a", "flaky")],
        orchestration_pattern="sequential",
        bounds=RunBounds(max_retries=3),
    )
    graph = compile_graph(plan)
    result = graph.invoke(initial_state({"q": "x"}))

    assert result["halted"] is False
    assert result["errors"] == []
    assert result["outputs"]["a"]["content"] == "recovered"


# --------------------------------------------------------------------------- parallel
def test_parallel_independent_agents_all_run(monkeypatch):
    _patch(
        monkeypatch,
        {
            "m-a": _Structured([AgentOutput(content="from a")]),
            "m-b": _Structured([AgentOutput(content="from b")]),
        },
    )
    plan = AgentPlan(
        agents=[_spec("a", "m-a"), _spec("b", "m-b")],
        orchestration_pattern="parallel",
    )
    result = compile_graph(plan).invoke(initial_state({"q": "x"}))

    assert result["halted"] is False
    assert result["outputs"]["a"]["content"] == "from a"
    assert result["outputs"]["b"]["content"] == "from b"


def test_parallel_failure_isolated_and_merge_node_gated(monkeypatch):
    # 'a' fails, 'b' (independent) succeeds, 'synth' depends on both -> must be skipped
    # because a produced no output, while b's result is preserved.
    _patch(
        monkeypatch,
        {
            "m-fail": _Structured([ValueError("boom")]),
            "m-ok": _Structured([AgentOutput(content="from b")]),
            "m-synth": _Structured([AgentOutput(content="should not appear")]),
        },
    )
    plan = AgentPlan(
        agents=[
            _spec("a", "m-fail"),
            _spec("b", "m-ok"),
            _spec("synth", "m-synth", depends_on=["a", "b"]),
        ],
        orchestration_pattern="parallel",
        bounds=RunBounds(max_retries=2),
    )
    result = compile_graph(plan).invoke(initial_state({"q": "x"}))

    assert result["outputs"]["b"]["content"] == "from b"  # sibling unaffected
    assert "a" not in result["outputs"]
    assert "synth" not in result["outputs"]  # merge node gated on missing dependency
    failed_ids = {e["agent_id"] for e in result["errors"]}
    assert {"a", "synth"} <= failed_ids


# ------------------------------------------------------------------------- supervisor
def test_supervisor_routes_then_finishes(monkeypatch):
    # Supervisor routes to 'w' on its first turn, then FINISH on its second.
    _patch(
        monkeypatch,
        {
            "m-sup": _Structured(
                [
                    _SupervisorDecision(next_agent="w", reason="do work"),
                    _SupervisorDecision(next_agent="FINISH", reason="done"),
                ]
            ),
            "m-w": _Structured([AgentOutput(content="work done")]),
        },
    )
    plan = AgentPlan(
        agents=[_spec("sup", "m-sup"), _spec("w", "m-w")],
        orchestration_pattern="supervisor",
        supervisor_id="sup",
    )
    result = compile_graph(plan).invoke(initial_state({"q": "x"}))

    assert result["outputs"]["w"]["content"] == "work done"
    assert result["next_agent"] == "FINISH"
    assert result["supervisor_steps"] == 2


# ------------------------------------------------------------------------------ tools
@tool
def _fake_web_search(query: str) -> str:
    """Deterministic stand-in for the real web_search tool, so this test never hits
    Tavily's API."""
    return f"search result for {query}"


class _FakeAIMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class _FakeToolCallLLM:
    """Fakes one round of tool-calling then a structured final answer, exercising the
    tool loop in `_run_agent_once` end to end through the compiled graph."""

    def __init__(self, structured):
        self._structured = structured
        self._calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self._calls += 1
        if self._calls == 1:
            return _FakeAIMessage(
                tool_calls=[{"name": "web_search", "args": {"query": "test"}, "id": "call-1"}]
            )
        return _FakeAIMessage(tool_calls=[])

    def with_structured_output(self, schema):
        return self._structured


def test_agent_with_tools_calls_tool_then_produces_structured_output(monkeypatch):
    monkeypatch.setitem(tools_module.TOOL_REGISTRY, "web_search", lambda: _fake_web_search)
    fake_llm = _FakeToolCallLLM(_Structured([AgentOutput(content="done")]))
    monkeypatch.setattr(gb, "get_llm", lambda cfg: fake_llm)
    monkeypatch.setattr(gb.time, "sleep", lambda s: None)

    spec = AgentSpec(
        id="a",
        role="worker",
        system_prompt="agent a",
        llm=LLMConfig(provider="google_genai", model="tool-model"),
        tools=["web_search"],
    )
    plan = AgentPlan(agents=[spec], orchestration_pattern="sequential")
    result = compile_graph(plan).invoke(initial_state({"q": "x"}))

    assert result["halted"] is False
    assert result["errors"] == []
    assert result["outputs"]["a"]["content"] == "done"


def test_supervisor_invalid_route_target_finishes_safely(monkeypatch):
    # Supervisor names a non-existent worker -> route coerced to FINISH, no crash/loop.
    _patch(
        monkeypatch,
        {
            "m-sup": _Structured([_SupervisorDecision(next_agent="ghost")]),
            "m-w": _Structured([AgentOutput(content="unused")]),
        },
    )
    plan = AgentPlan(
        agents=[_spec("sup", "m-sup"), _spec("w", "m-w")],
        orchestration_pattern="supervisor",
        supervisor_id="sup",
    )
    result = compile_graph(plan).invoke(initial_state({"q": "x"}))

    assert result["next_agent"] == "FINISH"
    assert "w" not in result["outputs"]
