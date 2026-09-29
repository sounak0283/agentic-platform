import app as app_module
from agent_schema import AgentPlan, AgentSpec, LLMConfig
from fastapi.testclient import TestClient


def _plan():
    return AgentPlan(
        agents=[
            AgentSpec(
                id="a",
                role="worker",
                system_prompt="do it",
                llm=LLMConfig(provider="google_genai", model="gemini-3.1-pro-preview"),
            )
        ],
        orchestration_pattern="sequential",
    )


def _client(monkeypatch):
    # Stub the planner so API tests never hit a real model.
    monkeypatch.setattr(app_module, "plan_project", lambda **kw: _plan())
    app_module._PROJECTS.clear()
    app_module._COMPILED.clear()
    return TestClient(app_module.app)


def test_health(monkeypatch):
    c = _client(monkeypatch)
    assert c.get("/health").json() == {"status": "ok"}


def test_create_returns_plan(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/projects", json={"brief": "x", "agent_count": 1})
    assert r.status_code == 201
    assert r.json()["plan"]["agents"][0]["id"] == "a"


def test_get_unknown_project_is_404(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/projects/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error_type"] == "ProjectNotFound"


def test_run_before_compile_is_409(monkeypatch):
    c = _client(monkeypatch)
    pid = c.post("/projects", json={"brief": "x", "agent_count": 1}).json()["project_id"]
    r = c.post(f"/projects/{pid}/run", json={"input": {"q": "x"}})
    assert r.status_code == 409
    assert r.json()["error_type"] == "NotCompiled"


def test_agent_count_out_of_range_rejected(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/projects", json={"brief": "x", "agent_count": 0})
    assert r.status_code == 422


def test_edit_plan_invalidates_compiled_graph(monkeypatch):
    c = _client(monkeypatch)
    pid = c.post("/projects", json={"brief": "x", "agent_count": 1}).json()["project_id"]
    c.post(f"/projects/{pid}/compile")
    assert pid in app_module._COMPILED
    edited = _plan().model_dump()
    r = c.patch(f"/projects/{pid}/plan", json=edited)
    assert r.status_code == 200
    assert pid not in app_module._COMPILED  # stale graph dropped on edit


def test_invoke_unknown_agent_is_404(monkeypatch):
    c = _client(monkeypatch)
    pid = c.post("/projects", json={"brief": "x", "agent_count": 1}).json()["project_id"]
    r = c.post(
        f"/projects/{pid}/agents/ghost/invoke", json={"input": {"q": "x"}}
    )
    assert r.status_code == 404
    assert r.json()["error_type"] == "AgentNotFound"


def test_openapi_docs_available(monkeypatch):
    c = _client(monkeypatch)
    assert c.get("/openapi.json").status_code == 200
    assert c.get("/docs").status_code == 200


def test_create_project_default_excludes_side_effect_tools(monkeypatch):
    c = _client(monkeypatch)
    seen_kwargs = {}

    def spy(**kw):
        seen_kwargs.update(kw)
        return _plan()

    monkeypatch.setattr(app_module, "plan_project", spy)
    r = c.post("/projects", json={"brief": "x", "agent_count": 1})
    assert r.status_code == 201
    assert "web_search" not in seen_kwargs["available_tools"]
    assert "lookup" in seen_kwargs["available_tools"]
    assert r.json()["side_effect_tools_enabled"] == []


def test_create_project_explicit_web_search_is_opt_in(monkeypatch):
    c = _client(monkeypatch)
    seen_kwargs = {}

    def spy(**kw):
        seen_kwargs.update(kw)
        return _plan()

    monkeypatch.setattr(app_module, "plan_project", spy)
    r = c.post(
        "/projects",
        json={"brief": "x", "agent_count": 1, "available_tools": ["lookup", "web_search"]},
    )
    assert r.status_code == 201
    assert seen_kwargs["available_tools"] == ["lookup", "web_search"]
    assert r.json()["side_effect_tools_enabled"] == ["web_search"]

    pid = r.json()["project_id"]
    assert c.get(f"/projects/{pid}").json()["side_effect_tools_enabled"] == ["web_search"]
