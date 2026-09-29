import { Pill } from "./ui.jsx";

const STATUS = {
  checking: { tone: undefined, label: "Checking API" },
  online: { tone: "success", label: "API online" },
  offline: { tone: "danger", label: "API unreachable" },
};

export function TopBar({ health }) {
  const status = STATUS[health] ?? STATUS.checking;
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand__mark">AP</span>
        <span className="brand__name">Agentic Platform</span>
        <span className="brand__sub">Console</span>
      </div>
      <div className="topbar__spacer" />
      <Pill tone={status.tone} dot>
        {status.label}
      </Pill>
      <a
        className="topbar__link"
        href="http://127.0.0.1:8000/docs"
        target="_blank"
        rel="noreferrer"
      >
        API docs
      </a>
    </header>
  );
}
