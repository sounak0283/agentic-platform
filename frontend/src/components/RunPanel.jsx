import { useState } from "react";
import { Button, Callout, ErrorCallout, Pill } from "./ui.jsx";

const SAMPLE = '{\n  "question": "What is the speed of light?"\n}';

function Results({ result }) {
  if (!result) return null;

  const outputs = result.outputs ?? {};
  const errors = result.errors ?? [];
  const entries = Object.entries(outputs);

  return (
    <div style={{ marginTop: 16 }}>
      <div className="row" style={{ marginBottom: 12 }}>
        <Pill tone={result.halted ? "danger" : "success"} dot>
          {result.halted ? "Halted" : "Completed"}
        </Pill>
        <Pill>{entries.length} agent outputs</Pill>
        {errors.length > 0 && <Pill tone="danger">{errors.length} errors</Pill>}
      </div>

      {errors.map((err, i) => (
        <div key={i} style={{ marginBottom: 10 }}>
          <Callout tone="danger" title={`${err.agent_id} — failed after ${err.attempts} attempts`}>
            {err.error}
          </Callout>
        </div>
      ))}

      {entries.map(([agentId, output]) => (
        <div key={agentId} className="result">
          <header className="result__head">
            <span className="agent__id">{agentId}</span>
          </header>
          <div className="result__body">{output.content}</div>
          {output.data && Object.keys(output.data).length > 0 && (
            <pre className="result__data">{JSON.stringify(output.data, null, 2)}</pre>
          )}
        </div>
      ))}

      {entries.length === 0 && errors.length === 0 && (
        <div className="empty">The run produced no outputs.</div>
      )}
    </div>
  );
}

export function RunPanel({ project, compiled, onRun, onInvoke, busy, error, result }) {
  const [tab, setTab] = useState("graph");
  const [input, setInput] = useState(SAMPLE);
  const [agentId, setAgentId] = useState("");
  const [parseError, setParseError] = useState(null);

  const agents = project?.plan?.agents ?? [];
  const selectedAgent = agentId || agents[0]?.id || "";

  function execute() {
    let parsed;
    try {
      parsed = JSON.parse(input);
    } catch (e) {
      setParseError(`Invalid JSON: ${e.message}`);
      return;
    }
    setParseError(null);
    tab === "graph" ? onRun(parsed) : onInvoke(selectedAgent, parsed);
  }

  const blocked = tab === "graph" && !compiled;

  return (
    <section className="panel">
      <header className="panel__header">
        <h2 className="panel__title">Execute</h2>
      </header>

      <div className="tabs">
        <button
          className={`tab ${tab === "graph" ? "tab--active" : ""}`}
          onClick={() => setTab("graph")}
        >
          Full graph
        </button>
        <button
          className={`tab ${tab === "agent" ? "tab--active" : ""}`}
          onClick={() => setTab("agent")}
        >
          Single agent
        </button>
      </div>

      <div className="panel__body">
        {tab === "agent" && (
          <label className="field">
            <span className="field__label">Agent</span>
            <select value={selectedAgent} onChange={(e) => setAgentId(e.target.value)}>
              {agents.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.id} — {a.role}
                </option>
              ))}
            </select>
          </label>
        )}

        <label className="field">
          <span className="field__label">Task input (JSON)</span>
          <textarea
            className="mono"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            style={{ minHeight: 130 }}
          />
          <span className="field__hint">
            {tab === "graph"
              ? "Passed to every agent as the shared task input."
              : "Runs one agent in isolation, without the rest of the graph."}
          </span>
        </label>

        {blocked && (
          <Callout tone="warn">Compile the plan before running the full graph.</Callout>
        )}

        {parseError && (
          <div style={{ marginTop: 12 }}>
            <ErrorCallout error={{ errorType: "Invalid JSON", message: parseError }} />
          </div>
        )}

        {error && (
          <div style={{ marginTop: 12 }}>
            <ErrorCallout error={error} />
          </div>
        )}

        <Results result={result} />
      </div>

      <div className="panel__footer">
        <Button
          variant="primary"
          loading={busy}
          disabled={!project || blocked}
          onClick={execute}
        >
          {busy ? "Running…" : tab === "graph" ? "Run graph" : "Invoke agent"}
        </Button>
        <span className="field__hint" style={{ marginTop: 0 }}>
          Runs are bounded by the plan's step, retry, and timeout limits.
        </span>
      </div>
    </section>
  );
}
