import type { ClassValues } from "../types";

const CLASSES = ["No Crossing", "Crossing", "Irrelevant"] as const;

interface ProbabilityBarsProps {
  probabilities: ClassValues;
  logits?: ClassValues;
  compact?: boolean;
}

export function ProbabilityBars({
  probabilities,
  logits,
  compact,
}: ProbabilityBarsProps) {
  return (
    <div className={`probability-bars ${compact ? "compact" : ""}`}>
      {CLASSES.map((name) => {
        const probability = probabilities[name];
        return (
          <div className="probability-row" key={name}>
            <div className="probability-label">
              <span>{name}</span>
              <span>
                {(probability * 100).toFixed(1)}%
                {logits && <small>logit {logits[name].toFixed(2)}</small>}
              </span>
            </div>
            <div className="probability-track">
              <span
                className={name.toLowerCase().replace(" ", "-")}
                style={{
                  width: `${Math.min(100, Math.max(0, probability * 100))}%`,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
