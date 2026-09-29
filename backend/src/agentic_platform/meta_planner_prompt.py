"""Meta-planner: natural-language brief -> validated AgentPlan.

This is the only component allowed to turn free text into a plan, and it emits nothing
but structured data (an AgentPlan). Even though the LLM is asked for structured output,
we re-validate server-side (principle: never trust the model's claim that its output is
well-formed) and apply the runtime error-recovery loop to the planner itself: on a
validation failure, feed the actual error back to the model and retry up to a small
ceiling before raising a typed PlannerError.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from .agent_schema import AgentPlan, LLMConfig
from .config import DEFAULT_MODELS, DEFAULT_PROVIDER
from .errors import PlannerError
from .graph_builder import get_llm
from .tools import TOOL_REGISTRY

_PLANNER_RETRY_CEILING = 3
_PLANNER_BACKOFF_S = 1.0

_SYSTEM_PROMPT = """\
You are a meta-planner for a multi-agent platform. Given a task description and a desired
agent count, you design a team of AI agents and return it as a structured AgentPlan.

Rules you MUST follow:
- Produce exactly the requested number of agents unless the task clearly needs fewer;
  never exceed it.
- Each agent needs a unique `id` (short snake_case), a `role`, and a concrete
  `system_prompt` telling that agent exactly what to do with its input.
- `tools` for each agent must be chosen ONLY from the available tool names given below.
  Use an empty list if an agent needs no tools. Never invent tool names.
- `llm` for each agent must be chosen ONLY from the available LLM options given below
  (copy provider + model exactly).
- Wire agents with `depends_on` (list of other agent ids whose output this agent needs).
- Pick `orchestration_pattern`:
    * "sequential" — a strict pipeline where each agent feeds the next. Use for simple
      linear tasks. Wire the order with depends_on.
    * "parallel" — independent agents work on the same input concurrently, then one
      merge/synthesis agent combines them. Give the independent agents an empty
      depends_on, and give the merge agent depends_on listing all of them.
    * "supervisor" — a router agent decides at runtime which worker acts next, looping
      until done. Use when the task needs conditional branching or dynamic ordering.
- depends_on must not contain cycles for sequential/parallel patterns.
- If and ONLY if you choose "supervisor", also set `supervisor_id` to the id of the
  router agent (it must be one of the agents), and leave the worker agents' depends_on
  empty (the supervisor decides ordering at runtime). For sequential/parallel, leave
  `supervisor_id` null.
"""


def _build_user_message(
    user_brief: str,
    requested_agent_count: int,
    available_tools: list[str],
    available_llms: list[dict[str, Any]],
    correction: str | None,
) -> str:
    parts = [
        f"TASK DESCRIPTION:\n{user_brief}",
        f"\nDESIRED AGENT COUNT: {requested_agent_count}",
        f"\nAVAILABLE TOOL NAMES: {available_tools or '(none)'}",
        "\nAVAILABLE LLM OPTIONS (choose provider+model from these only):",
    ]
    for opt in available_llms:
        parts.append(f"  - provider={opt['provider']}, model={opt['model']}")
    if correction:
        parts.append(
            "\nYOUR PREVIOUS ATTEMPT WAS REJECTED. Fix exactly this problem and return a "
            f"corrected AgentPlan:\n{correction}"
        )
    return "\n".join(parts)


def _semantic_check(
    plan: AgentPlan,
    available_tools: list[str],
    available_llms: list[dict[str, Any]],
) -> None:
    """Checks the schema alone can't express, raised as ValueError to feed the retry loop."""
    tool_set = set(available_tools)
    allowed_llms = {(o["provider"], o["model"]) for o in available_llms}
    for agent in plan.agents:
        unknown_tools = [t for t in agent.tools if t not in tool_set]
        if unknown_tools:
            raise ValueError(
                f"agent '{agent.id}' uses tools not in the available set: {unknown_tools}. "
                f"Allowed: {sorted(tool_set)}"
            )
        choice = (agent.llm.provider, agent.llm.model)
        if choice not in allowed_llms:
            raise ValueError(
                f"agent '{agent.id}' uses llm {choice} which is not an available option. "
                f"Allowed: {sorted(allowed_llms)}"
            )


def plan_project(
    user_brief: str,
    requested_agent_count: int,
    available_tools: list[str] | None = None,
    available_llms: list[dict[str, Any]] | None = None,
    planner_llm: LLMConfig | None = None,
) -> AgentPlan:
    """Turn a natural-language brief into a validated AgentPlan.

    Retries the structured-output call on validation failure, feeding the actual error
    back to the model each time, then raises PlannerError if the ceiling is exhausted.
    """
    available_tools = available_tools if available_tools is not None else list(TOOL_REGISTRY)
    available_llms = available_llms or [
        {"provider": DEFAULT_PROVIDER, "model": DEFAULT_MODELS[DEFAULT_PROVIDER]}
    ]
    planner_llm = planner_llm or LLMConfig(
        provider=DEFAULT_PROVIDER, model=DEFAULT_MODELS[DEFAULT_PROVIDER]
    )

    llm = get_llm(planner_llm)
    structured = llm.with_structured_output(AgentPlan)

    correction: str | None = None
    last_error: str | None = None

    for attempt in range(1, _PLANNER_RETRY_CEILING + 1):
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=_build_user_message(
                    user_brief,
                    requested_agent_count,
                    available_tools,
                    available_llms,
                    correction,
                )
            ),
        ]
        try:
            plan: AgentPlan = structured.invoke(messages)
            # Re-validate server-side: the structured-output call parses into AgentPlan
            # (running its model validators), but semantic constraints beyond the schema
            # (tool/LLM allow-lists) are enforced here explicitly.
            _semantic_check(plan, available_tools, available_llms)
            return plan
        except (ValidationError, ValueError) as e:
            last_error = str(e)
            correction = last_error
            if attempt < _PLANNER_RETRY_CEILING:
                time.sleep(_PLANNER_BACKOFF_S)

    raise PlannerError(
        f"planner failed to produce a valid AgentPlan after {_PLANNER_RETRY_CEILING} attempts",
        attempts=_PLANNER_RETRY_CEILING,
        last_error=last_error,
    )
