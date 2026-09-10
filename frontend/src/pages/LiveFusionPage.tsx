import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { AppHeader } from "../components/AppHeader";
import { ModelCard } from "../components/ModelCard";
import { PredictionPanel } from "../components/PredictionPanel";
import { SampleLibrary } from "../components/SampleLibrary";
import { SequenceViewer } from "../components/SequenceViewer";
import { StatusDot } from "../components/StatusDot";
import type {
  BatchPredictionResponse,
  FilterName,
  HealthResponse,
  PredictionResponse,
  SampleDetail,
  SampleSummary,
} from "../types";

export function LiveFusionPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [sample, setSample] = useState<SampleDetail | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [filter, setFilter] = useState<FilterName>("All");
  const [loadingSample, setLoadingSample] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [batching, setBatching] = useState(false);
  const [batch, setBatch] = useState<BatchPredictionResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api.health(), api.samples()])
      .then(([healthResponse, sampleResponse]) => {
        setHealth(healthResponse);
        setSamples(sampleResponse);
        if (sampleResponse.length) setSelectedId(sampleResponse[0].id);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    setLoadingSample(true);
    setPrediction(null);
    api
      .sample(selectedId)
      .then(setSample)
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoadingSample(false));
  }, [selectedId]);

  const selectedSummary = useMemo(
    () => samples.find((item) => item.id === selectedId),
    [samples, selectedId],
  );

  async function runPrediction() {
    if (!selectedId) return;
    setPredicting(true);
    setError("");
    try {
      setPrediction(await api.predict(selectedId));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Prediction failed");
    } finally {
      setPredicting(false);
    }
  }

  async function runAll() {
    setBatching(true);
    setError("");
    try {
      setBatch(await api.predictAll());
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Batch inference failed",
      );
    } finally {
      setBatching(false);
    }
  }

  return (
    <div className="app-shell">
      <AppHeader
        activePage="live"
        subtitle="PIE · Trajectory + Social Fusion"
        actions={
          <>
            {batch && (
              <div className="batch-result">
                <span>Demo set</span>
                <strong>
                  {batch.correct}/{batch.total} correct ·{" "}
                  {(batch.accuracy * 100).toFixed(1)}%
                </strong>
              </div>
            )}
            <button
              className="secondary-action"
              disabled={batching}
              onClick={runAll}
              type="button"
            >
              {batching ? "Running 15…" : "Evaluate all 15"}
            </button>
            <StatusDot
              label={health ? "2 source models online" : "Connecting"}
              ready={health?.models_loaded ?? false}
            />
          </>
        }
      />

      {error && (
        <div className="error-banner" role="alert">
          <span>{error}</span>
          <button onClick={() => setError("")} type="button">
            Dismiss
          </button>
        </div>
      )}

      <main className="workspace">
        <SampleLibrary
          filter={filter}
          onFilterChange={setFilter}
          onSelect={setSelectedId}
          samples={samples}
          selectedId={selectedId}
        />

        <div className="analysis-column">
          <div className="analysis-grid">
            <SequenceViewer loading={loadingSample} sample={sample} />
            <PredictionPanel
              onPredict={runPrediction}
              prediction={prediction}
              predicting={predicting}
              sample={sample}
            />
          </div>

          <section className="model-section" aria-label="Source model outputs">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Inside the pipeline</p>
                <h2>Source model signals</h2>
              </div>
              <p>
                Independent sigmoid outputs · probabilities do not need to sum
                to 100%
              </p>
            </div>
            <div className="model-grid">
              <ModelCard
                index="01"
                output={prediction?.trajectory}
                subtitle="15 × 8 relation-risk graph"
                title="Trajectory context"
              />
              <div className="fusion-connector" aria-hidden="true">
                <span>+</span>
              </div>
              <ModelCard
                index="02"
                output={prediction?.social}
                subtitle="15 × 12 interaction graph"
                title="Social interaction"
              />
            </div>
          </section>

          <footer className="method-footer">
            <span>Selected sample</span>
            <strong>
              {selectedSummary
                ? `${selectedSummary.set_id} / ${selectedSummary.video_id} / ` +
                  selectedSummary.pedestrian_id
                : "Loading…"}
            </strong>
            <span className="footer-divider" />
            <span>Inference contract</span>
            <strong>
              2 checkpoints → classwise logits → fixed binary mapping
            </strong>
          </footer>
        </div>
      </main>
    </div>
  );
}
