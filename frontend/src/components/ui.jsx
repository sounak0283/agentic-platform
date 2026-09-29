export function Panel({ title, actions, children, footer }) {
  return (
    <section className="panel">
      {(title || actions) && (
        <header className="panel__header">
          {title && <h2 className="panel__title">{title}</h2>}
          <div className="panel__spacer" />
          {actions}
        </header>
      )}
      <div className="panel__body">{children}</div>
      {footer && <div className="panel__footer">{footer}</div>}
    </section>
  );
}

export function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="field__label">{label}</span>
      {children}
      {hint && <span className="field__hint">{hint}</span>}
    </label>
  );
}

export function Button({ variant, size, block, loading, children, ...props }) {
  const classes = [
    "btn",
    variant && `btn--${variant}`,
    size && `btn--${size}`,
    block && "btn--block",
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <button className={classes} disabled={loading || props.disabled} {...props}>
      {loading && <span className="spin" />}
      {children}
    </button>
  );
}

export function Pill({ tone, dot, children }) {
  return (
    <span className={["pill", tone && `pill--${tone}`].filter(Boolean).join(" ")}>
      {dot && <span className="dot" />}
      {children}
    </span>
  );
}

export function Callout({ tone, title, children, detail }) {
  return (
    <div className={["callout", tone && `callout--${tone}`].filter(Boolean).join(" ")}>
      <div className="callout__body">
        {title && <div className="callout__title">{title}</div>}
        {children}
        {detail && <pre className="callout__detail">{detail}</pre>}
      </div>
    </div>
  );
}

export function ErrorCallout({ error }) {
  if (!error) return null;
  const detail =
    error.details == null
      ? null
      : typeof error.details === "string"
        ? error.details
        : JSON.stringify(error.details, null, 2);
  return (
    <Callout tone="danger" title={error.errorType ?? "Error"} detail={detail}>
      {error.message}
    </Callout>
  );
}

export function MetaRow({ label, children }) {
  return (
    <div className="metarow">
      <span className="metarow__key">{label}</span>
      <span className="metarow__val">{children}</span>
    </div>
  );
}
