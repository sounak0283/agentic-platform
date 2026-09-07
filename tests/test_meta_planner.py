import pytest

import meta_planner_prompt as mp
from agent_schema import AgentPlan, AgentSpec, LLMConfig
from errors import PlannerError


def _valid_plan():
    return AgentPlan(
        agents=[
            AgentSpec(
                id="only",
                role="worker",
                system_prompt="do it",
                llm=LLMConfig(provider="google_genai", model="gemini-3.1-pro-preview"),
                tools=["lookup"],
            )
        ],
        orchestration_pattern="sequential",
    )


class _FakeStructured:
    """Stand-in for llm.with_structured_output(AgentPlan): scripted per-call outcomes."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeLLM:
    def __init__(self, structured):
        self._structured = structured

    def with_structured_output(self, schema):
        return self._structured


def _patch_llm(monkeypatch, outcomes):
    structured = _FakeStructured(outcomes)
    monkeypatch.setattr(mp, "get_llm", lambda cfg: _FakeLLM(structured))
    monkeypatch.setattr(mp.time, "sleep", lambda s: None)  # no real backoff in tests
    return structured


def test_recovers_after_one_invalid_attempt(monkeypatch):
    # First attempt returns a plan using a tool not in the allow-list -> semantic ValueError;
    # second attempt returns a valid plan. The loop should feed the error back and succeed.
    bad_plan = AgentPlan(
        agents=[
            AgentSpec(
                id="only",
                role="worker",
                system_prompt="do it",
                llm=LLMConfig(provider="google_genai", model="gemini-3.1-pro-preview"),
                tools=["ghost_tool"],
            )
        ],
        orchestration_pattern="sequential",
    )
    structured = _patch_llm(monkeypatch, [bad_plan, _valid_plan()])

    plan = mp.plan_project(
        user_brief="x",
        requested_agent_count=1,
        available_tools=["lookup"],
        available_llms=[{"provider": "google_genai", "model": "gemini-3.1-pro-preview"}],
    )
    assert plan.agents[0].tools == ["lookup"]
    assert structured.calls == 2


def test_raises_planner_error_after_ceiling(monkeypatch):
    structured = _patch_llm(monkeypatch, [ValueError("bad")] * 3)

    with pytest.raises(PlannerError) as exc_info:
        mp.plan_project(
            user_brief="x",
            requested_agent_count=1,
            available_tools=["lookup"],
            available_llms=[{"provider": "google_genai", "model": "gemini-3.1-pro-preview"}],
        )
    assert exc_info.value.attempts == 3
    assert structured.calls == 3
