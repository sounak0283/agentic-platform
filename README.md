# Agentic Platform

**Natural-language multi-agent builder.** Describe a task in plain English, say how many
agents you want and what each should do, and get back live, callable API endpoints for a
compiled multi-agent system — ready to wire into your own website, app, or pipeline.

> Full architecture, the reasoning behind every dependency, error model, and deployment
> design: see [systemDescription.md](systemDescription.md). This file is a quick-start.

---

## Overview

### Problem

Building a multi-agent AI system today means hand-writing orchestration logic, wiring up
tools, choosing and configuring models, and standing up infrastructure to serve it — even
for people who know exactly *what* they want the agents to do but don't want to own the
engineering behind *how* it runs.

### What it does

1. The user describes their task and the agent team they want, in natural language.
2. The planning layer turns that description into a structured, validated plan: how many
   agents, each agent's job, how they're wired together (sequential, parallel, or
   supervised/routed), what tools each needs, and which LLM powers each one.
3. The user reviews and can adjust the plan before anything is built.
4. The platform compiles the plan into a running multi-agent graph and serves it —
   locally via FastAPI today, on Amazon Bedrock AgentCore Runtime for managed deployment.
5. The user takes the resulting endpoints and builds whatever they want on top, without
   touching orchestration code themselves.

### Non-goals (for now)

- Not a general no-code app builder — the output is API endpoints, not a UI.
- Not a training or fine-tuning platform — it composes and serves existing foundation models.
- Not aiming for full arbitrary-code agents — agents are LLM calls plus a fixed, vetted
  set of tools, never sandboxed code-execution environments.

---

## Project structure

```
.
├── agent_schema.py         # Pydantic contracts: LLMConfig, AgentSpec, AgentPlan
├── meta_planner_prompt.py  # Meta-planner: brief -> validated AgentPlan
├── graph_builder.py        # LLM factory + AgentPlan -> LangGraph compiler
├── tools.py                # Closed, vetted tool registry
├── config.py                # Env loading, provider/model defaults
├── errors.py                # Typed platform errors
├── app.py                   # FastAPI deployment layer (local/MVP serving)
├── tests/                   # Offline, fully mocked test suite
├── scripts/smoke_e2e.py     # Real-API end-to-end smoke script
├── README.md                 # This file
├── systemDescription.md      # Full architecture, flow diagrams, dependency rationale
├── CLAUDE.md                 # Operating contract for whoever builds this
└── SYSTEM_PROMPT.md          # Original builder system prompt
```

---

## Getting started

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies (creates .venv from pyproject.toml / uv.lock)
uv sync

# 2. Configure environment
cp .env.example .env
# Fill in at least one LLM provider key (the default provider is Gemini, so
# GOOGLE_API_KEY or GEMINI_API_KEY is the easiest to start with).
# TAVILY_API_KEY is only needed if a project opts into the web_search tool.

# 3. Run the API locally
uv run uvicorn app:app --reload
# -> http://127.0.0.1:8000/docs for interactive OpenAPI docs
```

Try it end to end with curl:

```bash
# Plan a project from a natural-language brief
curl -s -X POST localhost:8000/projects \
  -H "Content-Type: application/json" \
  -d '{"brief": "Answer a factual question, then write a one-sentence summary.", "agent_count": 2}'
# -> {"project_id": "...", "plan": {...}, "side_effect_tools_enabled": []}

# Compile the plan into a runnable graph
curl -s -X POST localhost:8000/projects/<project_id>/compile

# Run it
curl -s -X POST localhost:8000/projects/<project_id>/run \
  -H "Content-Type: application/json" \
  -d '{"input": {"question": "What is the speed of light?"}}'
```

### Tests

```bash
uv run pytest                       # offline, fully mocked — no API keys or network needed
uv run python scripts/smoke_e2e.py  # real LLM + tool calls end to end (needs .env keys)
```

---

## Orchestration patterns

Chosen automatically by the planner per project, never hardcoded to one:

| Pattern | When it's used |
|---|---|
| **Sequential** | Strict pipeline — each agent's output feeds the next (research → draft → edit). |
| **Parallel** | Independent agents run concurrently on the same input and merge at a synthesis step. |
| **Supervisor** | A router agent decides at runtime which agent acts next, looping until it signals completion. |

See [systemDescription.md](systemDescription.md) for how each pattern actually compiles
to a LangGraph graph.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| **MVP** | Meta-planner + schema, compiler (all 3 patterns), FastAPI serving, real `web_search` tool + opt-in enforcement | **Done** |
| **V1** | Persistence (Postgres/Redis); per-project API keys; plan-review/edit UX; synthesis (aggregate) endpoint; package a compiled project as a Bedrock AgentCore runtime | Not started |
| **V2** | Dedicated AgentCore runtime per heavy/isolated tenant; AgentCore Observability + LangSmith; expanded tool registry; rate limiting and billing | Not started |

---

## Security notes (summary)

- No model-generated code is ever executed — only model-generated configuration,
  interpreted by a fixed compiler.
- Every agent run is bounded — enforced max steps, timeouts, and error retries.
- Tools with real-world side effects require explicit per-project opt-in; the planner
  never auto-attaches them.

Full security model, secrets handling, and the AgentCore deployment design:
see [systemDescription.md](systemDescription.md).
