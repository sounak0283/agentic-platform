"""FastAPI deployment layer.

Exposes the pipeline as HTTP: brief -> plan -> (edit) -> compile -> invoke/run. State is
in-memory for the MVP (Postgres/Redis are V1 per the roadmap); every project is keyed by
id so the move to external stores and multi-tenancy is additive, not a rewrite. Every
typed platform error is translated to a stable JSON error shape by a single handler per
type, so clients never see a raw stack trace.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from agent_schema import AgentPlan
from config import DEFAULT_MODELS, DEFAULT_PROVIDER
from errors import (
    MissingProviderKeyError,
    PlannerError,
    PlatformError,
    UnknownToolError,
    UnsupportedProviderError,
)
from graph_builder import compile_graph, run_graph, run_single_agent
from meta_planner_prompt import plan_project
from tools import TOOL_REGISTRY

app = FastAPI(
    title="Agentic Platform",
    description="Describe a multi-agent task in natural language; get live, callable "
    "endpoints for a compiled multi-agent system.",
    version="0.1.0",
)

# In-memory stores (MVP). Keyed by project id so persistence/multi-tenancy is additive.
_PROJECTS: dict[str, dict[str, Any]] = {}
_COMPILED: dict[str, Any] = {}


# --------------------------------------------------------------------------- models
class CreateProjectRequest(BaseModel):
    brief: str
    agent_count: int = Field(gt=0, le=10)
    available_tools: list[str] | None = None
    available_llms: list[dict[str, str]] | None = None


class RunRequest(BaseModel):
    input: dict[str, Any]


class AgentInvokeRequest(BaseModel):
    input: dict[str, Any]
    upstream_outputs: dict[str, Any] = Field(default_factory=dict)


class ProjectNotFoundError(PlatformError):
    def __init__(self, project_id: str):
        self.project_id = project_id
        super().__init__(f"project not found: {project_id}")


class NotCompiledError(PlatformError):
    def __init__(self, project_id: str):
        self.project_id = project_id
        super().__init__(f"project '{project_id}' has not been compiled yet")


def _get_project(project_id: str) -> dict[str, Any]:
    project = _PROJECTS.get(project_id)
    if project is None:
        raise ProjectNotFoundError(project_id)
    return project


# --------------------------------------------------------------------------- routes
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/projects", status_code=201)
def create_project(req: CreateProjectRequest) -> dict[str, Any]:
    available_llms = req.available_llms or [
        {"provider": DEFAULT_PROVIDER, "model": DEFAULT_MODELS[DEFAULT_PROVIDER]}
    ]
    available_tools = req.available_tools if req.available_tools is not None else list(TOOL_REGISTRY)
    plan = plan_project(
        user_brief=req.brief,
        requested_agent_count=req.agent_count,
        available_tools=available_tools,
        available_llms=available_llms,
    )
    project_id = uuid.uuid4().hex
    _PROJECTS[project_id] = {"id": project_id, "brief": req.brief, "plan": plan}
    return {"project_id": project_id, "plan": plan.model_dump()}


@app.get("/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    project = _get_project(project_id)
    return {
        "project_id": project_id,
        "brief": project["brief"],
        "plan": project["plan"].model_dump(),
        "compiled": project_id in _COMPILED,
    }


@app.patch("/projects/{project_id}/plan")
def edit_plan(project_id: str, plan: AgentPlan) -> dict[str, Any]:
    # AgentPlan validators run on parse, so an invalid edit is rejected before it lands.
    project = _get_project(project_id)
    project["plan"] = plan
    _COMPILED.pop(project_id, None)  # invalidate stale compiled graph on plan edit
    return {"project_id": project_id, "plan": plan.model_dump()}


@app.post("/projects/{project_id}/compile")
def compile_project(project_id: str) -> dict[str, Any]:
    project = _get_project(project_id)
    _COMPILED[project_id] = compile_graph(project["plan"])
    return {"project_id": project_id, "status": "compiled"}


@app.post("/projects/{project_id}/run")
def run_project(project_id: str, req: RunRequest) -> dict[str, Any]:
    project = _get_project(project_id)
    graph = _COMPILED.get(project_id)
    if graph is None:
        raise NotCompiledError(project_id)
    result = run_graph(graph, req.input, project["plan"].bounds)
    return {
        "project_id": project_id,
        "outputs": result["outputs"],
        "errors": result["errors"],
        "halted": result["halted"],
    }


@app.post("/projects/{project_id}/agents/{agent_id}/invoke")
def invoke_agent(project_id: str, agent_id: str, req: AgentInvokeRequest) -> dict[str, Any]:
    project = _get_project(project_id)
    try:
        result = run_single_agent(
            project["plan"], agent_id, req.input, req.upstream_outputs
        )
    except KeyError:
        return JSONResponse(
            status_code=404,
            content={
                "error_type": "AgentNotFound",
                "message": f"agent '{agent_id}' not in project '{project_id}'",
                "details": None,
            },
        )
    return {"project_id": project_id, "agent_id": agent_id, "result": result}


# --------------------------------------------------------------------------- errors
def _error_response(status: int, error_type: str, message: str, details: Any = None):
    return JSONResponse(
        status_code=status,
        content={"error_type": error_type, "message": message, "details": details},
    )


@app.exception_handler(ProjectNotFoundError)
def _handle_not_found(_req: Request, exc: ProjectNotFoundError):
    return _error_response(404, "ProjectNotFound", str(exc))


@app.exception_handler(NotCompiledError)
def _handle_not_compiled(_req: Request, exc: NotCompiledError):
    return _error_response(409, "NotCompiled", str(exc))


@app.exception_handler(UnknownToolError)
def _handle_unknown_tool(_req: Request, exc: UnknownToolError):
    return _error_response(400, "UnknownTool", str(exc), {"names": exc.names})


@app.exception_handler(UnsupportedProviderError)
def _handle_unsupported_provider(_req: Request, exc: UnsupportedProviderError):
    return _error_response(400, "UnsupportedProvider", str(exc))


@app.exception_handler(MissingProviderKeyError)
def _handle_missing_key(_req: Request, exc: MissingProviderKeyError):
    # Server misconfiguration, not client error — but never echo the key or its value.
    return _error_response(500, "MissingProviderKey", str(exc))


@app.exception_handler(PlannerError)
def _handle_planner(_req: Request, exc: PlannerError):
    return _error_response(
        422, "PlannerFailed", str(exc), {"attempts": exc.attempts, "last_error": exc.last_error}
    )


@app.exception_handler(ValidationError)
def _handle_validation(_req: Request, exc: ValidationError):
    return _error_response(422, "ValidationError", "invalid plan or request", exc.errors())


@app.exception_handler(PlatformError)
def _handle_platform(_req: Request, exc: PlatformError):
    return _error_response(400, type(exc).__name__, str(exc))
