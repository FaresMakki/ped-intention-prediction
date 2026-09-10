import type { PredictionResponse, SampleDetail } from "../types";
import { ProbabilityBars } from "./ProbabilityBars";

interface PredictionPanelProps {
  sample: SampleDetail | null;
  prediction: PredictionResponse | null;
  predicting: boolean;
  onPredict: () => void;
}

export function PredictionPanel({
  sample,
  prediction,
  predicting,
  onPredict,
}: PredictionPanelProps) {
  return (
    <aside className="prediction-panel panel">
      <div className="prediction-heading">
        <div>
          <p className="eyebrow">Final decision</p>
          <h2>Locked fusion</h2>
        </div>
        <span className="rule-badge">RULE B</span>
      </div>

      {!prediction ? (
        <div className="prediction-empty">
          <div className="radar-icon" aria-hidden="true">
            <span />
          </div>
          <h3>
            {predicting ? "Running both networks…" : "Ready for inference"}
          </h3>
          <p>
            This sends the prepared graph tensors through both copied
            checkpoints, then applies the validation-locked classwise fusion.
          </p>
        </div>
      ) : (
        <div className="prediction-result">
          <div
            className={`decision-card ${prediction.fusion.binary_prediction ? "crossing" : ""}`}
          >
            <span className="decision-kicker">Binary prediction</span>
            <strong>{prediction.fusion.binary_label}</strong>
            <span className="decision-score">
              Decision score {(prediction.fusion.binary_score * 100).toFixed(1)}
              %
            </span>
          </div>

          <div className="truth-row">
            <span>
              Ground truth <strong>{prediction.ground_truth.class_name}</strong>
            </span>
            <span className={prediction.is_correct ? "correct" : "incorrect"}>
              {prediction.is_correct ? "✓ Correct" : "× Incorrect"}
            </span>
          </div>

          <ProbabilityBars
            probabilities={prediction.fusion.probabilities}
            logits={prediction.fusion.logits}
          />

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
            Live PyTorch inference · {prediction.latency_ms.toFixed(1)} ms
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
            ? "Running inference"
            : prediction
              ? "Run again"
              : "Run both models"}
        </span>
        <span aria-hidden="true">{predicting ? "···" : "→"}</span>
      </button>
    </aside>
  );
}
