import { useState } from "react";
import { Button, ErrorCallout, MetaRow, Panel, Pill } from "./ui.jsx";

const PATTERN_TONE = { sequential: "accent", parallel: "accent", supervisor: "accent" };

function AgentCard({ agent, isSupervisor }) {
  return (
    <article className="agent">
      <header className="agent__head">
        <span className="agent__id">{agent.id}</span>
        {isSupervisor && <Pill tone="accent">router</Pill>}
        <span className="agent__role">{agent.role}</span>
      </header>
      <div className="agent__body">
        <p className="agent__prompt">{agent.system_prompt}</p>
        <div className="agent__meta">
          <Pill>{agent.llm.model}</Pill>
          {agent.tools.map((t) => (
            <Pill key={t} tone="warn">
              {t}
            </Pill>
          ))}
          {agent.depends_on.length > 0 && (
            <Pill>after: {agent.depends_on.join(", ")}</Pill>
          )}
        </div>
      </div>
    </article>
  );
}

export function PlanPanel({ project, onSave, onCompile, busy, error, compiled }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [parseError, setParseError] = useState(null);

  if (!project) {
    return (
      <Panel title="Agent plan">
        <div className="empty">
          No plan yet — describe a task and generate a plan to see the agent team here.
        </div>
      </Panel>
    );
  }

  const { plan } = project;

  function startEditing() {
    setDraft(JSON.stringify(plan, null, 2));
    setParseError(null);
    setEditing(true);
  }

  function save() {
    let parsed;
    try {
      parsed = JSON.parse(draft);
    } catch (e) {
      setParseError(`Invalid JSON: ${e.message}`);
      return;
    }
    setParseError(null);
    onSave(parsed, () => setEditing(false));
  }

  return (
    <Panel
      title="Agent plan"
      actions={
        editing ? (
          <div className="row">
            <Button size="sm" onClick={() => setEditing(false)}>
              Cancel
            </Button>
            <Button size="sm" variant="primary" loading={busy} onClick={save}>
              Save plan
            </Button>
          </div>
        ) : (
          <div className="row">
            <Button size="sm" onClick={startEditing}>
              Edit JSON
            </Button>
            <Button size="sm" variant="primary" loading={busy} onClick={onCompile}>
              {compiled ? "Recompile" : "Compile"}
            </Button>
          </div>
        )
      }
    >
      {editing ? (
        <>
          <textarea
            className="mono"
            style={{ minHeight: 420 }}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
          <div className="field__hint">
            Edited plans are re-validated server-side — cycles, unknown agent ids, and
            unknown tools are rejected before anything compiles.
          </div>
          {parseError && (
            <div style={{ marginTop: 12 }}>
              <ErrorCallout error={{ errorType: "Invalid JSON", message: parseError }} />
            </div>
          )}
        </>
      ) : (
        <>
          <div className="metalist" style={{ marginBottom: 16 }}>
            <MetaRow label="Project">
              <span className="mono">{project.project_id}</span>
            </MetaRow>
            <MetaRow label="Pattern">
              <Pill tone={PATTERN_TONE[plan.orchestration_pattern]}>
                {plan.orchestration_pattern}
              </Pill>
              {plan.supervisor_id && <Pill>router: {plan.supervisor_id}</Pill>}
            </MetaRow>
            <MetaRow label="Bounds">
              <Pill>{plan.bounds.max_steps} steps</Pill>
              <Pill>{plan.bounds.max_retries} retries</Pill>
              <Pill>{plan.bounds.timeout_s}s timeout</Pill>
            </MetaRow>
            <MetaRow label="External">
              {project.side_effect_tools_enabled?.length ? (
                project.side_effect_tools_enabled.map((t) => (
                  <Pill key={t} tone="warn">
                    {t}
                  </Pill>
                ))
              ) : (
                <span style={{ color: "var(--text-subtle)" }}>None enabled</span>
              )}
            </MetaRow>
            <MetaRow label="Status">
              {compiled ? (
                <Pill tone="success" dot>
                  Compiled
                </Pill>
              ) : (
                <Pill dot>Not compiled</Pill>
              )}
            </MetaRow>
          </div>

          <div className="agentgrid">
            {plan.agents.map((agent) => (
              <AgentCard
                key={agent.id}
                agent={agent}
                isSupervisor={agent.id === plan.supervisor_id}
              />
            ))}
          </div>
        </>
      )}

      {error && (
        <div style={{ marginTop: 12 }}>
          <ErrorCallout error={error} />
        </div>
      )}
    </Panel>
  );
}
