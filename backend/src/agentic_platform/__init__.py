"""Agentic Platform: natural-language brief in, compiled multi-agent system out.

The public surface is deliberately small — the planner turns a brief into a validated
`AgentPlan`, the compiler turns that plan into a runnable graph, and nothing else needs
to be imported to drive the pipeline.
"""

from .agent_schema import AgentOutput, AgentPlan, AgentSpec, LLMConfig, RunBounds
from .graph_builder import compile_graph, get_llm, run_graph, run_single_agent
from .meta_planner_prompt import plan_project
from .tools import SIDE_EFFECT_TOOLS, default_tool_names, get_tools

__all__ = [
    "AgentOutput",
    "AgentPlan",
    "AgentSpec",
    "LLMConfig",
    "RunBounds",
    "SIDE_EFFECT_TOOLS",
    "compile_graph",
    "default_tool_names",
    "get_llm",
    "get_tools",
    "plan_project",
    "run_graph",
    "run_single_agent",
]
