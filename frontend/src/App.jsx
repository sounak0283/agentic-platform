import { useCallback, useEffect, useState } from "react";
import { api } from "./lib/api.js";
import { DefinePanel } from "./components/DefinePanel.jsx";
import { PlanPanel } from "./components/PlanPanel.jsx";
import { RunPanel } from "./components/RunPanel.jsx";
import { Stepper } from "./components/Stepper.jsx";
import { TopBar } from "./components/TopBar.jsx";
import { Button, Panel } from "./components/ui.jsx";

export default function App() {
  const [health, setHealth] = useState("checking");
  const [tools, setTools] = useState({ tools: [], default: [] });
  const [providers, setProviders] = useState([]);

  const [project, setProject] = useState(null);
  const [compiled, setCompiled] = useState(false);
  const [result, setResult] = useState(null);

  // One busy flag per concern, so a slow run never greys out the whole console.
  const [busy, setBusy] = useState({});
  const [errors, setErrors] = useState({});

  const setBusyFor = (key, value) => setBusy((b) => ({ ...b, [key]: value }));
  const setErrorFor = (key, value) => setErrors((e) => ({ ...e, [key]: value }));

  useEffect(() => {
    api
      .health()
      .then(() => setHealth("online"))
      .catch(() => setHealth("offline"));
    api.tools().then(setTools).catch(() => {});
    api
      .providers()
      .then((r) => setProviders(r.providers))
      .catch(() => {});
  }, []);

  const call = useCallback(async (key, fn) => {
    setBusyFor(key, true);
    setErrorFor(key, null);
    try {
      return await fn();
    } catch (error) {
      setErrorFor(key, error);
      return null;
    } finally {
      setBusyFor(key, false);
    }
  }, []);

  async function createProject(form) {
    const created = await call("define", () => api.createProject(form));
    if (!created) return;
    setProject(created);
    setCompiled(false);
    setResult(null);
  }

  async function savePlan(plan, done) {
    const updated = await call("plan", () => api.updatePlan(project.project_id, plan));
    if (!updated) return;
    // A plan edit invalidates the compiled graph server-side; mirror that here.
    setProject((p) => ({ ...p, plan: updated.plan }));
    setCompiled(false);
    done();
  }

  async function compile() {
    const ok = await call("plan", () => api.compile(project.project_id));
    if (ok) setCompiled(true);
  }

  async function runGraph(input) {
    setResult(null);
    const res = await call("run", () => api.run(project.project_id, input));
    if (res) setResult(res);
  }

  async function invokeAgent(agentId, input) {
    setResult(null);
    const res = await call("run", () => api.invokeAgent(project.project_id, agentId, input));
    if (!res) return;
    // Normalise the single-agent shape into the same {outputs, errors, halted} view.
    setResult({
      outputs: res.result.outputs ?? {},
      errors: res.result.errors ?? [],
      halted: Boolean(res.result.halted),
    });
  }

  function reset() {
    setProject(null);
    setCompiled(false);
    setResult(null);
    setErrors({});
  }

  const step = !project ? 0 : !compiled ? 1 : result ? 3 : 2;

  return (
    <div className="app">
      <TopBar health={health} />
      <main className="workspace">
        <Stepper current={step} />

        <div className="columns">
          <div className="stack">
            <DefinePanel
              tools={tools}
              providers={providers}
              onSubmit={createProject}
              busy={busy.define}
              error={errors.define}
              disabled={health === "offline"}
            />

            {project && (
              <Panel title="Session">
                <Button block onClick={reset}>
                  Start a new project
                </Button>
              </Panel>
            )}
          </div>

          <div className="stack">
            <PlanPanel
              project={project}
              compiled={compiled}
              onSave={savePlan}
              onCompile={compile}
              busy={busy.plan}
              error={errors.plan}
            />
            {project && (
              <RunPanel
                project={project}
                compiled={compiled}
                onRun={runGraph}
                onInvoke={invokeAgent}
                busy={busy.run}
                error={errors.run}
                result={result}
              />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
