import graph_builder as gb
from agent_schema import AgentOutput, AgentPlan, AgentSpec, LLMConfig, RunBounds
from graph_builder import compile_graph, initial_state


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
