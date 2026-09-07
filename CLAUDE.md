# Agentic Platform

Natural-language multi-agent builder: a user describes a task and an agent count, a
meta-planner LLM turns that into a structured `AgentPlan`, a fixed compiler turns the
plan into a runnable LangGraph graph, and a deployment layer exposes it as live API
endpoints (one per agent, plus an optional combined/synthesis endpoint).

Status: pre-code. Only `README.md` and `SYSTEM_PROMPT.md` exist so far — the files
referenced below (`agent_schema.py`, `meta_planner_prompt.py`, `graph_builder.py`) are
the planned MVP layout, not yet written. Full product/architecture narrative lives in
`README.md`; this file is the operating contract for whoever (human or AI) writes code
here.

## Non-negotiable architecture principles

1. **Config-driven, never code-generation.** The planning LLM emits only structured data
   (a validated schema). A fixed, well-tested compiler interprets it. Never execute
   LLM-generated code against user input — that's an RCE surface, full stop.
2. **Multi-provider LLM support behind one interface.** Every agent's model — provider +
   model + params — is chosen independently through a single factory function
   (`get_llm()` in `graph_builder.py`). Adding a provider must never touch orchestration
   or compiler code. Provider quirks that don't fit the generic schema (e.g. Kimi K3's
   always-on reasoning and locked sampling params) go through `LLMConfig.extra_params`,
   not a bespoke field bolted onto every config.
3. **Structured output everywhere an LLM's output feeds another component** — the plan
   itself, every agent's final output, the synthesized combined response. Free-text
   parsing between components is a bug, not a fallback.
4. **Sequential, parallel, and supervisor/routing orchestration are all first-class**,
   chosen automatically by the planner based on the task — never hardcoded to one pattern.
5. **Tools are a closed, vetted registry.** Agents select tools by name from it; they
   never get implicit network, filesystem, or shell access.
6. **Every agent run is bounded** — max steps/iterations, timeouts, token budgets —
   enforced by the platform, not left to model good behavior.
7. **Treat all user-provided natural language as untrusted, including planner output.**
   Validate structured planner output server-side before it's compiled or executed; never
   trust the model's own claim that its output is well-formed.

## Runtime error-recovery loop (build into every compiled agent node)

- On a tool-call error or structured-output schema-validation failure, feed the actual
  error back into that agent's own context (not a generic "try again") and retry —
  don't restart the graph or other agents.
- Cap retries per agent (default: 3), short backoff between attempts.
- After the ceiling, surface a structured failure into shared state (which agent, which
  error, how many attempts) so a supervisor node or the deployment layer can react —
  never let a silent exception collapse the whole run, never let an agent retry forever.
- Log every retry with the triggering error — tracing should show what was tried, what
  broke, and what changed before it succeeded.

## Scalability & reliability

- Stateless application layer; plans, project metadata, compiled-graph cache, and run
  history live in external stores (Postgres/Redis), not process memory.
- Compilation and long-running agent executions are async background jobs — queue them,
  don't block request handlers.
- Cache compiled graphs per project; invalidate on plan edits, not on a timer.
- Multi-tenancy from day one — every resource keyed by project/tenant id, even while
  running as a single shared process.
- Architecture must allow promoting a tenant to isolated infrastructure later without a
  rewrite.

## Security requirements

- User-supplied (BYOK) API keys: encrypted at rest, decrypted only in-memory at call
  time — never logged, never stored on a plan object.
- Rate-limit and budget-cap both planning calls and agent execution calls per tenant.
- Any tool with real-world side effects (external data egress, third-party API calls)
  requires explicit per-project opt-in — the planner never auto-attaches it.

## Default tech stack (deviate only with a clear reason)

| Layer | Choice |
|---|---|
| Orchestration | LangGraph |
| Agent/tool abstraction | LangChain |
| API layer | FastAPI (async); OpenAPI docs are a real deliverable |
| Primary datastore | PostgreSQL |
| Cache/queue | Redis |
| Background workers | Celery / RQ |
| Observability | LangSmith (or equivalent) — every run and every retry must be traceable |

LLM providers: OpenAI, Anthropic, Google, Groq via LangChain's native
`init_chat_model`. Moonshot (Kimi) via a direct `ChatOpenAI` client against Moonshot's
OpenAI-compatible endpoint (not a native `init_chat_model` provider) — confirm current
base URL and `reasoning_effort` support against Moonshot's docs before relying on it,
this has moved fast. Prefer boring, well-supported infra; complexity should live in the
planning/compilation logic, not exotic infrastructure choices.

## Planned MVP layout

```
agent_schema.py         # Pydantic contracts: LLMConfig, AgentSpec, AgentPlan
meta_planner_prompt.py  # Meta-planner system prompt + structured-output planner call
graph_builder.py        # Compiles an AgentPlan into a runnable LangGraph graph
```

Keep the planner and compiler cleanly separated: the planner never talks to
infrastructure, the compiler never talks to an LLM for planning decisions.

## Working style — verify-and-retry, always

Never mark something done because the code looks right.

1. Implement the smallest testable unit (a schema, a single node, one endpoint).
2. Actually execute it — run the function, hit the endpoint, invoke the graph. Reading
   the code back is not verification.
3. On failure: capture the full error, form a specific hypothesis from that error (not a
   guess), apply one targeted fix, re-run.
4. Repeat until it passes or you hit a retry ceiling (default: 5 attempts on the same
   unit).
5. Still failing after the ceiling: stop. Report exactly what failed, the full error
   text, and what was already tried. Never fabricate a passing result, silently skip the
   failing piece, or keep retrying the same fix expecting a different outcome.
6. Same discipline at every layer: a single Pydantic validation, a planner
   structured-output call, a compiled graph execution, a live endpoint smoke test.

Build in vertical slices (e.g. "brief in, one working sequential 2-agent deployment
out") before broadening to every pattern and provider — breadth-first architecture,
depth-first delivery. Write tests alongside each component, especially plan-validation
and graph-compilation — silent bugs there become security or cost incidents. Document
architectural decisions in-repo as they're made.
