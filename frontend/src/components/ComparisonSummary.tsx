import type { ComparisonResponse, SampleDetail } from "../types";

interface ComparisonSummaryProps {
  comparison: ComparisonResponse | null;
  onCompare: () => void;
  comparing: boolean;
  sample: SampleDetail | null;
}

export function ComparisonSummary({
  comparison,
  onCompare,
  comparing,
  sample,
}: ComparisonSummaryProps) {
  return (
    <section className="comparison-summary panel">
      <div>
        <p className="eyebrow">Decision room</p>
        <h2>Five-model comparison</h2>
        <p className="summary-description">
          Run every checkpoint for every annotated pedestrian, then aggregate
          one global scene decision per method.
        </p>
      </div>

      {comparison ? (
        <div className="vote-panel">
          <span className="vote-kicker">
            {comparison.agreement.unanimous
              ? "Unanimous vote"
              : "Models disagree"}
          </span>
          <strong>
            {comparison.agreement.crossing_votes} Crossing ·{" "}
            {comparison.agreement.not_crossing_votes} Not Crossing
          </strong>
          <div className="vote-meter" aria-label="Crossing model votes">
            <span
              style={{
                width: `${(comparison.agreement.crossing_votes / comparison.agreement.total_methods) * 100}%`,
              }}
            />
          </div>
          <p>
            {comparison.agreement.correct_methods}/
            {comparison.agreement.total_methods} methods match the ground truth.
          </p>
          <div className="global-method-results">
            {comparison.global_methods.map((method) => (
              <div className="global-method-result" key={method.id}>
                <span>{method.short_name}</span>
                <strong>{method.binary_label}</strong>
                <small>
                  {method.crossing_pedestrian_ids.length}/
                  {comparison.pedestrian_count} pedestrians Crossing
                </small>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="comparison-empty">
          <span className="comparison-orbit" aria-hidden="true">
            <i />
          </span>
          <p>The results will appear below as one aligned comparison board.</p>
        </div>
      )}

      <div className="comparison-truth">
        <span>Global ground truth</span>
        <strong>
          {comparison
            ? comparison.global_ground_truth.binary_label
            : sample
              ? sample.binary_target
                ? "Crossing"
                : "Not Crossing"
              : "—"}
        </strong>
      </div>
      <button
        className="primary-action"
        disabled={!sample || comparing}
        onClick={onCompare}
        type="button"
      >
        <span>
          {comparing ? "Running checkpoints…" : "Run all pedestrians"}
        </span>
        <span>→</span>
      </button>
      {comparison && (
        <p className="latency-line">
          Full comparison · {comparison.latency_ms.toFixed(1)} ms
        </p>
      )}
    </section>
  );
}
