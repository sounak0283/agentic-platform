"""End-to-end smoke test against a real LLM provider.

Runs the full product path — brief -> plan -> compile -> run -> single-agent invoke —
through the actual FastAPI stack, making real model calls. Kept out of the default
pytest run (which is fully mocked) so routine testing never burns API budget. Run it
before calling the MVP "working":

    uv run python scripts/smoke_e2e.py

Exits non-zero on any failure.
"""

from __future__ import annotations

import os
import sys

from fastapi.testclient import TestClient

import agentic_platform.app as app_module
from agentic_platform.agent_schema import AgentPlan, AgentSpec, LLMConfig
from agentic_platform.config import DEFAULT_MODELS, DEFAULT_PROVIDER
from agentic_platform.graph_builder import compile_graph, initial_state

_LLM = LLMConfig(provider=DEFAULT_PROVIDER, model=DEFAULT_MODELS[DEFAULT_PROVIDER])


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _spec(id_, role, prompt, deps=None, tools=None):
    return AgentSpec(
        id=id_,
        role=role,
        system_prompt=prompt,
        llm=_LLM,
        depends_on=deps or [],
        tools=tools or [],
    )


def _check_run(label: str, plan: AgentPlan, task_input: dict) -> None:
    result = compile_graph(plan).invoke(initial_state(task_input))
    if result["errors"]:
        _fail(f"{label} produced errors: {result['errors']}")
    if not result["outputs"]:
        _fail(f"{label} produced no outputs")
    print(f"      {label}: {list(result['outputs'])} outputs, halted={result['halted']}")


def main() -> None:
    client = TestClient(app_module.app)

    # 1. Plan from a natural-language brief.
    r = client.post(
        "/projects",
        json={
            "brief": "Answer a factual question, then write a one-sentence summary.",
            "agent_count": 2,
        },
    )
    if r.status_code != 201:
        _fail(f"create project returned {r.status_code}: {r.text}")
    project_id = r.json()["project_id"]
    agents = [a["id"] for a in r.json()["plan"]["agents"]]
    print(f"[1] planned project {project_id} with agents {agents}")

    # 2. Compile.
    r = client.post(f"/projects/{project_id}/compile")
    if r.status_code != 200:
        _fail(f"compile returned {r.status_code}: {r.text}")
    print("[2] compiled")

    # 3. Run the full graph.
    r = client.post(
        f"/projects/{project_id}/run",
        json={"input": {"question": "What is the speed of light?"}},
    )
    if r.status_code != 200:
        _fail(f"run returned {r.status_code}: {r.text}")
    body = r.json()
    if body["halted"] or body["errors"]:
        _fail(f"run halted/errored: {body['errors']}")
    if not body["outputs"]:
        _fail("run produced no outputs")
    print(f"[3] ran full graph; {len(body['outputs'])} agent outputs")
    for aid, out in body["outputs"].items():
        print(f"      [{aid}] {out.get('content', '')[:120]}")

    # 4. Invoke a single agent directly.
    first_agent = agents[0]
    r = client.post(
        f"/projects/{project_id}/agents/{first_agent}/invoke",
        json={"input": {"question": "What is the capital of France?"}},
    )
    if r.status_code != 200:
        _fail(f"single-agent invoke returned {r.status_code}: {r.text}")
    result = r.json()["result"]
    if result.get("errors"):
        _fail(f"single-agent invoke errored: {result['errors']}")
    print(f"[4] invoked agent '{first_agent}' directly")

    # 5. Parallel pattern: two independent analysts merged by a synthesizer.
    print("[5] parallel pattern")
    parallel = AgentPlan(
        agents=[
            _spec("pros", "Pros", "List two concise advantages of the subject in the input."),
            _spec("cons", "Cons", "List two concise disadvantages of the subject in the input."),
            _spec(
                "synth",
                "Synthesizer",
                "Given the pros and cons from prior agents, write one balanced verdict.",
                deps=["pros", "cons"],
            ),
        ],
        orchestration_pattern="parallel",
    )
    _check_run("parallel", parallel, {"subject": "remote work"})

    # 6. Supervisor pattern: a router coordinates a math helper and a writer.
    print("[6] supervisor pattern")
    supervisor = AgentPlan(
        agents=[
            _spec("sup", "Supervisor", "Route to the math helper to compute, then the writer to phrase the answer, then FINISH."),
            _spec("math", "Math", "Perform the arithmetic in the input and state the numeric result."),
            _spec("writer", "Writer", "Using prior outputs, write one friendly sentence with the final answer."),
        ],
        orchestration_pattern="supervisor",
        supervisor_id="sup",
    )
    _check_run("supervisor", supervisor, {"question": "What is 12 times 8?"})

    # 7. Real web_search tool, gated on TAVILY_API_KEY being set — a project must
    # explicitly opt in by naming it in available_tools (never auto-attached).
    if os.environ.get("TAVILY_API_KEY"):
        print("[7] web_search tool (real Tavily call)")
        search_plan = AgentPlan(
            agents=[
                _spec(
                    "researcher",
                    "Researcher",
                    "Use the web_search tool to find one current fact about the input "
                    "topic, then report it in one sentence.",
                    tools=["web_search"],
                )
            ],
            orchestration_pattern="sequential",
        )
        _check_run("web_search", search_plan, {"topic": "the current version of Python"})
    else:
        print("[7] SKIPPED web_search step: TAVILY_API_KEY not set in environment")

    # 8. Keyless tools (calculator + wikipedia) through the full stack — no API key
    # needed, so this step always runs.
    print("[8] keyless tools (calculator + real Wikipedia call)")
    tool_plan = AgentPlan(
        agents=[
            _spec(
                "researcher",
                "Researcher",
                "Use the wikipedia tool to look up the subject in the input, then state "
                "one factual sentence about it.",
                tools=["wikipedia"],
            ),
            _spec(
                "math",
                "Math",
                "Use the calculator tool to evaluate the expression in the input and "
                "state the result.",
                tools=["calculator"],
                deps=["researcher"],
            ),
        ],
        orchestration_pattern="sequential",
    )
    _check_run("keyless tools", tool_plan, {"subject": "Alan Turing", "expression": "(12 * 8) / 4"})

    # 9. WolframAlpha, gated on its app id being set.
    if os.environ.get("WOLFRAM_ALPHA_APPID"):
        print("[9] wolfram_alpha tool (real call)")
        wolfram_plan = AgentPlan(
            agents=[
                _spec(
                    "solver",
                    "Solver",
                    "Use the wolfram_alpha tool to answer the question in the input.",
                    tools=["wolfram_alpha"],
                )
            ],
            orchestration_pattern="sequential",
        )
        _check_run("wolfram_alpha", wolfram_plan, {"question": "What is 17 squared?"})
    else:
        print("[9] SKIPPED wolfram_alpha step: WOLFRAM_ALPHA_APPID not set in environment")

    print("\nOK: end-to-end smoke passed (sequential + parallel + supervisor + tools).")


if __name__ == "__main__":
    main()
