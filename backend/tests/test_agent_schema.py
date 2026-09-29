import pytest
from pydantic import ValidationError

from agent_schema import AgentPlan, AgentSpec, LLMConfig


def _spec(id_, depends_on=None, **kw):
    return AgentSpec(
        id=id_,
        role="tester",
        system_prompt="do the thing",
        llm=LLMConfig(provider="moonshot", model="kimi-k2-0711-preview"),
        depends_on=depends_on or [],
        **kw,
    )


def test_valid_sequential_plan_parses():
    plan = AgentPlan(
        agents=[_spec("research"), _spec("draft", depends_on=["research"])],
        orchestration_pattern="sequential",
    )
    assert len(plan.agents) == 2
    assert plan.bounds.max_retries == 3


def test_dangling_depends_on_rejected():
    with pytest.raises(ValidationError, match="unknown agent id"):
        AgentPlan(
            agents=[_spec("draft", depends_on=["ghost"])],
            orchestration_pattern="sequential",
        )


def test_self_dependency_rejected():
    with pytest.raises(ValidationError, match="cannot depend on itself"):
        AgentPlan(
            agents=[_spec("draft", depends_on=["draft"])],
            orchestration_pattern="sequential",
        )


def test_duplicate_agent_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate agent ids"):
        AgentPlan(
            agents=[_spec("a"), _spec("a")],
            orchestration_pattern="sequential",
        )


def test_cycle_rejected_for_sequential():
    with pytest.raises(ValidationError, match="cycle"):
        AgentPlan(
            agents=[_spec("a", depends_on=["b"]), _spec("b", depends_on=["a"])],
            orchestration_pattern="sequential",
        )


def test_empty_plan_rejected():
    with pytest.raises(ValidationError, match="at least one agent"):
        AgentPlan(agents=[], orchestration_pattern="sequential")


def test_unknown_provider_rejected():
    with pytest.raises(ValidationError):
        LLMConfig(provider="not_a_real_provider", model="x")


def test_supervisor_pattern_allows_no_acyclic_check():
    # Supervisor routing decides at runtime; depends_on cycles aren't meaningful there.
    plan = AgentPlan(
        agents=[_spec("sup"), _spec("a"), _spec("b")],
        orchestration_pattern="supervisor",
        supervisor_id="sup",
    )
    assert plan.orchestration_pattern == "supervisor"


def test_supervisor_requires_supervisor_id():
    with pytest.raises(ValidationError, match="requires supervisor_id"):
        AgentPlan(agents=[_spec("a"), _spec("b")], orchestration_pattern="supervisor")


def test_supervisor_id_must_reference_real_agent():
    with pytest.raises(ValidationError, match="not a known agent id"):
        AgentPlan(
            agents=[_spec("a"), _spec("b")],
            orchestration_pattern="supervisor",
            supervisor_id="ghost",
        )


def test_supervisor_needs_at_least_one_worker():
    with pytest.raises(ValidationError, match="at least one worker"):
        AgentPlan(
            agents=[_spec("sup")],
            orchestration_pattern="supervisor",
            supervisor_id="sup",
        )


def test_supervisor_id_rejected_for_non_supervisor_pattern():
    with pytest.raises(ValidationError, match="only be set when"):
        AgentPlan(
            agents=[_spec("a")],
            orchestration_pattern="sequential",
            supervisor_id="a",
        )
