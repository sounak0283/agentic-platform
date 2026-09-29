import { useState } from "react";
import { Button, Callout, ErrorCallout, Field, Panel, Pill } from "./ui.jsx";

export function DefinePanel({ tools, providers, onSubmit, busy, error, disabled }) {
  const [brief, setBrief] = useState("");
  const [agentCount, setAgentCount] = useState(2);
  const [selectedTools, setSelectedTools] = useState(null); // null = platform default
  const [provider, setProvider] = useState("");

  // Until the operator touches the tool list, mirror the platform default so the UI
  // shows the same opt-in posture the API applies server-side.
  const effectiveTools = selectedTools ?? tools.default ?? [];
  const optedIn = effectiveTools.filter(
    (name) => tools.tools?.find((t) => t.name === name)?.side_effect,
  );

  function toggleTool(name) {
    const next = new Set(effectiveTools);
    next.has(name) ? next.delete(name) : next.add(name);
    setSelectedTools([...next]);
  }

  function submit(event) {
    event.preventDefault();
    const chosen = providers.find((p) => `${p.provider}/${p.model}` === provider);
    onSubmit({
      brief: brief.trim(),
      agentCount: Number(agentCount),
      availableTools: effectiveTools,
      availableLlms: chosen
        ? [{ provider: chosen.provider, model: chosen.model }]
        : null,
    });
  }

  return (
    <form onSubmit={submit}>
      <Panel
        title="Define project"
        footer={
          <Button
            type="submit"
            variant="primary"
            block
            loading={busy}
            disabled={disabled || !brief.trim()}
          >
            {busy ? "Planning…" : "Generate plan"}
          </Button>
        }
      >
        <Field
          label="Task brief"
          hint="Describe the work in plain language. The meta-planner turns this into a validated agent plan."
        >
          <textarea
            value={brief}
            onChange={(e) => setBrief(e.target.value)}
            placeholder="e.g. Research a topic, draft a summary, then fact-check it."
            disabled={disabled}
          />
        </Field>

        <Field label="Agent count" hint="1–10. The planner may use fewer if the task is simpler.">
          <input
            type="number"
            min="1"
            max="10"
            value={agentCount}
            onChange={(e) => setAgentCount(e.target.value)}
            disabled={disabled}
          />
        </Field>

        <Field label="Model" hint="Applies to every agent unless the plan is edited.">
          <select
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
            disabled={disabled}
          >
            <option value="">Platform default</option>
            {providers.map((p) => (
              <option key={`${p.provider}/${p.model}`} value={`${p.provider}/${p.model}`}>
                {p.provider} · {p.model}
                {p.default ? " (default)" : ""}
              </option>
            ))}
          </select>
        </Field>

        <Field
          label="Available tools"
          hint="Agents can only use tools enabled here. The planner never adds one that isn't on this list."
        >
          <div className="toolgrid">
            {(tools.tools ?? []).map((tool) => (
              <label key={tool.name} className="tooloption">
                <input
                  type="checkbox"
                  checked={effectiveTools.includes(tool.name)}
                  onChange={() => toggleTool(tool.name)}
                  disabled={disabled}
                />
                <span className="tooloption__name">{tool.name}</span>
                {tool.side_effect && <Pill tone="warn">external</Pill>}
              </label>
            ))}
          </div>
        </Field>

        {optedIn.length > 0 && (
          <Callout tone="warn" title="External access opted in">
            Sends data to third-party services: {optedIn.join(", ")}. Enabled per project,
            never attached automatically.
          </Callout>
        )}

        <div style={{ marginTop: 12 }}>
          <ErrorCallout error={error} />
        </div>
      </Panel>
    </form>
  );
}
