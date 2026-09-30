import type { SampleDetail, ScenePredictionResponse } from "../types";
import { ProbabilityBars } from "./ProbabilityBars";

interface PredictionPanelProps {
  sample: SampleDetail | null;
  prediction: ScenePredictionResponse | null;
  predicting: boolean;
  focusedPedestrianId: string;
  onFocusPedestrian: (pedestrianId: string) => void;
  onPredict: () => void;
}

export function PredictionPanel({
  sample,
  prediction,
  predicting,
  focusedPedestrianId,
  onFocusPedestrian,
  onPredict,
}: PredictionPanelProps) {
  const focused =
    prediction?.predictions.find(
      (item) => item.pedestrian_id === focusedPedestrianId,
    ) ?? prediction?.predictions[0];

  return (
    <aside className="prediction-panel panel">
      <div className="prediction-heading">
        <div>
          <p className="eyebrow">Global decision</p>
          <h2>Parallel pedestrian fusion</h2>
        </div>
        <span className="rule-badge">BATCH</span>
      </div>

      {!prediction ? (
        <div className="prediction-empty">
          <div className="radar-icon" aria-hidden="true">
            <span />
          </div>
          <h3>
            {predicting ? "Running all pedestrians…" : "Ready for inference"}
          </h3>
          <p>
            Every annotated pedestrian is evaluated independently in one model
            batch. If any pedestrian predicts Crossing, the global scene result
            is Crossing.
          </p>
        </div>
      ) : (
        <div className="prediction-result">
          <div
            className={`decision-card ${prediction.global_decision.binary_prediction ? "crossing" : ""}`}
          >
            <span className="decision-kicker">Global scene prediction</span>
            <strong>{prediction.global_decision.binary_label}</strong>
            <span className="decision-score">
              {prediction.global_decision.crossing_pedestrian_ids.length} of{" "}
              {prediction.pedestrian_count} pedestrians predicted Crossing
            </span>
          </div>

          <div className="truth-row scene-truth-row">
            <span>
              Global ground truth{" "}
              <strong>
                {prediction.global_decision.ground_truth_binary_label}
              </strong>
            </span>
            <span
              className={
                prediction.global_decision.is_correct ? "correct" : "incorrect"
              }
            >
              {prediction.global_decision.is_correct
                ? "✓ Correct"
                : "× Incorrect"}
            </span>
          </div>

          <div className="person-results" aria-label="Pedestrian predictions">
            {prediction.predictions.map((person, index) => {
              const color =
                sample?.pedestrians.find(
                  (item) => item.pedestrian_id === person.pedestrian_id,
                )?.color ?? "#B1EF68";
              return (
                <button
                  aria-pressed={person.pedestrian_id === focused?.pedestrian_id}
                  className={`person-result ${
                    person.pedestrian_id === focused?.pedestrian_id
                      ? "active"
                      : ""
                  }`}
                  key={person.pedestrian_id}
                  onClick={() => onFocusPedestrian(person.pedestrian_id)}
                  type="button"
                >
                  <span className="person-result-id">
                    <i style={{ background: color }} /> P{index + 1}{" "}
                    {person.pedestrian_id}
                  </span>
                  <strong>{person.fusion.binary_label}</strong>
                  <small>
                    {(person.fusion.binary_score * 100).toFixed(1)}% · truth{" "}
                    {person.ground_truth.binary_label}
                  </small>
                </button>
              );
            })}
          </div>

          {focused && (
            <>
              <div className="focused-person-heading">
                <span>Selected pedestrian</span>
                <strong>{focused.pedestrian_id}</strong>
              </div>
              <ProbabilityBars
                probabilities={focused.fusion.probabilities}
                logits={focused.fusion.logits}
              />
            </>
          )}

          <div className="fusion-formula">
            <span>
              <b>NC</b> 75% trajectory + 25% social
            </span>
            <span>
              <b>C</b> 100% trajectory
            </span>
            <span>
              <b>IRR</b> 80% trajectory + 20% social
            </span>
          </div>

          <p className="latency-line">
            Batched live PyTorch inference · {prediction.latency_ms.toFixed(1)}{" "}
            ms
          </p>
        </div>
      )}

      <button
        className="primary-action"
        disabled={!sample || predicting}
        onClick={onPredict}
        type="button"
      >
        <span>
          {predicting
            ? "Running parallel inference"
            : prediction
              ? "Run scene again"
              : sample && sample.pedestrian_count > 1
                ? `Run all ${sample.pedestrian_count} pedestrians`
                : "Run both models"}
        </span>
        <span aria-hidden="true">{predicting ? "···" : "→"}</span>
      </button>
    </aside>
  );
}
