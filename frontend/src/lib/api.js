// Thin client over the platform API. Every failure is normalised into an ApiError
// carrying the backend's typed error shape ({error_type, message, details}), so the UI
// can show what actually went wrong instead of a generic "request failed".

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

export class ApiError extends Error {
  constructor({ status, errorType, message, details }) {
    super(message);
    this.status = status;
    this.errorType = errorType;
    this.details = details;
  }
}

async function request(path, { method = "GET", body } = {}) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (cause) {
    throw new ApiError({
      status: 0,
      errorType: "NetworkError",
      message: "Could not reach the API. Is the server running on port 8000?",
      details: String(cause),
    });
  }

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    throw new ApiError({
      status: response.status,
      errorType: payload?.error_type ?? "HttpError",
      message: payload?.message ?? `Request failed with status ${response.status}`,
      details: payload?.details ?? null,
    });
  }
  return payload;
}

export const api = {
  health: () => request("/health"),
  tools: () => request("/tools"),
  providers: () => request("/providers"),

  createProject: ({ brief, agentCount, availableTools, availableLlms }) =>
    request("/projects", {
      method: "POST",
      body: {
        brief,
        agent_count: agentCount,
        ...(availableTools ? { available_tools: availableTools } : {}),
        ...(availableLlms ? { available_llms: availableLlms } : {}),
      },
    }),

  getProject: (id) => request(`/projects/${id}`),
  updatePlan: (id, plan) => request(`/projects/${id}/plan`, { method: "PATCH", body: plan }),
  compile: (id) => request(`/projects/${id}/compile`, { method: "POST" }),
  run: (id, input) => request(`/projects/${id}/run`, { method: "POST", body: { input } }),
  invokeAgent: (id, agentId, input, upstreamOutputs = {}) =>
    request(`/projects/${id}/agents/${agentId}/invoke`, {
      method: "POST",
      body: { input, upstream_outputs: upstreamOutputs },
    }),
};
