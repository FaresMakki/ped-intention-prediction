import type { ComparisonMethod } from "../types";
import { ProbabilityBars } from "./ProbabilityBars";

interface ComparisonMethodCardProps {
  index: number;
  method: ComparisonMethod;
}

function percentage(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function ComparisonMethodCard({
  index,
  method,
}: ComparisonMethodCardProps) {
  return (
    <article
      className={`comparison-method ${method.is_final ? "is-final" : ""}`}
    >
      <div className="comparison-method-heading">
        <span className="method-rank">
          {String(index + 1).padStart(2, "0")}
        </span>
        <div>
          <div className="method-name-line">
            <h3>{method.name}</h3>
            {method.is_final && (
              <span className="recommended-badge">Final</span>
            )}
          </div>
          <p>{method.architecture}</p>
        </div>
        <div className="method-decision">
          <span>Binary decision</span>
          <strong>{method.binary_label}</strong>
          <small className={method.is_correct ? "correct" : "incorrect"}>
            {method.is_correct ? "Correct" : "Incorrect"}
          </small>
        </div>
      </div>

      <ProbabilityBars
        compact
        logits={method.logits}
        probabilities={method.probabilities}
      />

      <div className="method-evidence">
        <div>
          <span>Sample score</span>
          <strong>{percentage(method.binary_score)}</strong>
        </div>
        <div>
          <span>Test binary F1</span>
          <strong>{percentage(method.metrics.binary_macro_f1)}</strong>
        </div>
        <div>
          <span>Crossing recall</span>
          <strong>{percentage(method.metrics.crossing_recall)}</strong>
        </div>
        <div>
          <span>PR AUC</span>
          <strong>{percentage(method.metrics.pr_auc)}</strong>
        </div>
      </div>

      <footer className="method-card-footer">
        <span>{method.role}</span>
        <span>{method.selection_note}</span>
      </footer>
    </article>
  );
}
