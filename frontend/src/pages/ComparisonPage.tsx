import { useEffect, useState } from "react";
import { api } from "../api/client";
import { AppHeader } from "../components/AppHeader";
import { ComparisonMethodCard } from "../components/ComparisonMethodCard";
import { ComparisonSummary } from "../components/ComparisonSummary";
import { SampleLibrary } from "../components/SampleLibrary";
import { SequenceViewer } from "../components/SequenceViewer";
import { StatusDot } from "../components/StatusDot";
import type {
  ComparisonResponse,
  FilterName,
  HealthResponse,
  SampleDetail,
  SampleSummary,
} from "../types";

export function ComparisonPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [sample, setSample] = useState<SampleDetail | null>(null);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [filter, setFilter] = useState<FilterName>("All");
  const [loadingSample, setLoadingSample] = useState(false);
  const [comparing, setComparing] = useState(false);
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
    setComparison(null);
    api
      .sample(selectedId)
      .then(setSample)
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoadingSample(false));
  }, [selectedId]);

  async function runComparison() {
    if (!selectedId) return;
    setComparing(true);
    setError("");
    try {
      setComparison(await api.compare(selectedId));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Comparison failed");
    } finally {
      setComparing(false);
    }
  }

  return (
    <div className="app-shell">
      <AppHeader
        activePage="comparison"
        subtitle="PIE · Comparative Inference"
        actions={
          <StatusDot
            label={health ? "6 checkpoints online" : "Connecting"}
            ready={health?.models_loaded ?? false}
          />
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

      <main className="workspace comparison-workspace">
        <SampleLibrary
          filter={filter}
          onFilterChange={setFilter}
          onSelect={setSelectedId}
          samples={samples}
          selectedId={selectedId}
        />

        <div className="analysis-column comparison-column">
          <header className="comparison-intro">
            <div>
              <p className="eyebrow">
                Same sequence · Different representations
              </p>
              <h1>Compare how each model reads the scene.</h1>
            </div>
            <p>
              Sample outputs are live. Reference metrics come from each
              validation-selected checkpoint on the complete 715-sample test
              set.
            </p>
          </header>

          <div className="comparison-top-grid">
            <SequenceViewer loading={loadingSample} sample={sample} />
            <ComparisonSummary
              comparing={comparing}
              comparison={comparison}
              onCompare={runComparison}
              sample={sample}
            />
          </div>

          <section className="comparison-board" aria-label="Model results">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Comparison board</p>
                <h2>Aligned model decisions</h2>
              </div>
              <p>
                Every method uses the fixed binary mapping at threshold 0.5.
              </p>
            </div>
            {comparison ? (
              <div className="comparison-methods">
                {comparison.methods.map((method, index) => (
                  <ComparisonMethodCard
                    index={index}
                    key={method.id}
                    method={method}
                  />
                ))}
              </div>
            ) : (
              <div className="comparison-board-empty panel">
                <span>05</span>
                <div>
                  <strong>Five inference paths are ready.</strong>
                  <p>
                    Choose a prepared sequence and run the comparison to reveal
                    logits, probabilities, decisions, and test evidence.
                  </p>
                </div>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
