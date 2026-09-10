interface StatusDotProps {
  ready: boolean;
  label: string;
}

export function StatusDot({ ready, label }: StatusDotProps) {
  return (
    <span className={`status-pill ${ready ? "is-ready" : ""}`}>
      <span className="status-dot" />
      {label}
    </span>
  );
}
