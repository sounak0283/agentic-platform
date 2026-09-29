const STEPS = ["Define brief", "Review plan", "Compile", "Run"];

export function Stepper({ current }) {
  return (
    <nav className="stepper" aria-label="Progress">
      {STEPS.map((label, i) => {
        const state = i < current ? "done" : i === current ? "active" : "todo";
        return (
          <div key={label} className={`step step--${state}`}>
            <span className="step__index">{state === "done" ? "✓" : i + 1}</span>
            <span className="step__label">{label}</span>
          </div>
        );
      })}
    </nav>
  );
}
