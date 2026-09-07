"""Pydantic contracts for the agent planning/compilation pipeline.

The meta-planner LLM is only ever allowed to emit `AgentPlan` (structured data);
`graph_builder.compile_graph` is the only thing that turns that data into a runnable
graph. Nothing here imports the tool registry or an LLM client — validating that tool
names and provider strings actually resolve is a semantic check layered on top in
`meta_planner_prompt.plan_project` and again defensively in `graph_builder.compile_graph`,
not baked into these base models. That keeps this module a pure, dependency-free contract.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ProviderName = Literal["openai", "anthropic", "google_genai", "groq", "moonshot", "ollama"]
OrchestrationPattern = Literal["sequential", "parallel", "supervisor"]


class LLMConfig(BaseModel):
    provider: ProviderName
    model: str
    temperature: float = 0.0
    # Escape hatch for provider quirks that don't fit this generic schema (e.g. Moonshot
    # Kimi's always-on reasoning / locked sampling params) — never a bespoke field per
    # provider, per the platform's multi-provider factory principle.
    extra_params: dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    """Generic structured output every compiled agent node returns.

    MVP ships one shape for all agents; `AgentSpec.output_schema` is a forward-compatible
    hook for per-agent schemas registered by name, without changing this contract.
    """

    content: str
    data: dict[str, Any] = Field(default_factory=dict)


class AgentSpec(BaseModel):
    id: str
    role: str
    system_prompt: str
    llm: LLMConfig
    tools: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    output_schema: str = "default"

    @field_validator("id", "role", "system_prompt")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v


class RunBounds(BaseModel):
    max_steps: int = 8
    # Whole-graph wall-clock cap. Frontier reasoning models (e.g. Gemini 3.1 Pro) can take
    # tens of seconds per agent, and a multi-agent run chains several such calls, so the
    # default is generous; tighten it per project for latency-sensitive deployments.
    timeout_s: float = 300.0
    max_retries: int = 3


class AgentPlan(BaseModel):
    agents: list[AgentSpec]
    orchestration_pattern: OrchestrationPattern
    # Required for the supervisor pattern: the id of the agent that routes at runtime.
    # Must be None for sequential/parallel.
    supervisor_id: str | None = None
    bounds: RunBounds = Field(default_factory=RunBounds)

    @model_validator(mode="after")
    def _validate_graph(self) -> "AgentPlan":
        if not self.agents:
            raise ValueError("a plan must contain at least one agent")

        ids = [a.id for a in self.agents]
        seen: set[str] = set()
        duplicates = {i for i in ids if i in seen or seen.add(i)}
        if duplicates:
            raise ValueError(f"duplicate agent ids: {sorted(duplicates)}")

        id_set = set(ids)
        for agent in self.agents:
            if agent.id in agent.depends_on:
                raise ValueError(f"agent '{agent.id}' cannot depend on itself")
            unknown = [d for d in agent.depends_on if d not in id_set]
            if unknown:
                raise ValueError(
                    f"agent '{agent.id}' depends_on unknown agent id(s): {unknown}"
                )

        if self.orchestration_pattern in ("sequential", "parallel"):
            _assert_acyclic(self.agents)

        if self.orchestration_pattern == "supervisor":
            if self.supervisor_id is None:
                raise ValueError("supervisor pattern requires supervisor_id to be set")
            if self.supervisor_id not in id_set:
                raise ValueError(
                    f"supervisor_id '{self.supervisor_id}' is not a known agent id"
                )
            workers = [a for a in self.agents if a.id != self.supervisor_id]
            if not workers:
                raise ValueError(
                    "supervisor pattern requires at least one worker agent besides the "
                    "supervisor"
                )
        elif self.supervisor_id is not None:
            raise ValueError(
                "supervisor_id may only be set when orchestration_pattern is 'supervisor'"
            )

        return self


def _assert_acyclic(agents: list[AgentSpec]) -> None:
    graph = {a.id: a.depends_on for a in agents}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, path: list[str]) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle = " -> ".join([*path[path.index(node) :], node])
            raise ValueError(f"dependency cycle detected: {cycle}")
        visiting.add(node)
        for dep in graph[node]:
            visit(dep, [*path, node])
        visiting.discard(node)
        visited.add(node)

    for agent_id in graph:
        visit(agent_id, [])
