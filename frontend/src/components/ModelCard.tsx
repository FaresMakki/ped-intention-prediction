import type { SourceOutput } from "../types";
import { ProbabilityBars } from "./ProbabilityBars";

interface ModelCardProps {
  index: string;
  title: string;
  subtitle: string;
  output?: SourceOutput;
}

export function ModelCard({ index, title, subtitle, output }: ModelCardProps) {
  return (
    <article className="model-card panel">
      <div className="model-card-heading">
        <span className="model-index">{index}</span>
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <span className={`model-state ${output ? "live" : ""}`}>
          {output ? "Computed" : "Waiting"}
        </span>
      </div>
      {output ? (
        <ProbabilityBars
          probabilities={output.probabilities}
          logits={output.logits}
          compact
        />
      ) : (
        <div className="empty-bars" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
      )}
      <p className="checkpoint-name">
        {output?.checkpoint ?? "Checkpoint ready"}
      </p>
    </article>
  );
}
