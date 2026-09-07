"""End-to-end smoke test against a real LLM provider.

Runs the full product path — brief -> plan -> compile -> run -> single-agent invoke —
through the actual FastAPI stack, making real model calls. Kept out of the default
pytest run (which is fully mocked) so routine testing never burns API budget. Run it
before calling the MVP "working":

    uv run python scripts/smoke_e2e.py

Exits non-zero on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import app as app_module


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


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

    print("\nOK: end-to-end smoke passed.")


if __name__ == "__main__":
    main()
