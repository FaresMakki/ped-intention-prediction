import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { AppHeader } from "../components/AppHeader";
import { ModelCard } from "../components/ModelCard";
import { PredictionPanel } from "../components/PredictionPanel";
import { StatusDot } from "../components/StatusDot";
import type {
  HealthResponse,
  PreparedVideoUploadResponse,
  ScenePredictionResponse,
} from "../types";

const EXAMPLES = [
  {
    folder: "01_no_crossing_4ped",
    title: "No Crossing · 4 pedestrians",
    source: "SET03 / VIDEO 0010",
  },
  {
    folder: "02_crossing_4ped",
    title: "Crossing · 4 pedestrians",
    source: "SET03 / VIDEO 0012",
  },
  {
    folder: "03_crossing_4ped",
    title: "Crossing · 4 pedestrians",
    source: "SET03 / VIDEO 0015",
  },
  {
    folder: "04_irrelevant_4ped",
    title: "Irrelevant · 4 pedestrians",
    source: "SET04 / VIDEO 0003",
  },
  {
    folder: "05_no_crossing_4ped",
    title: "No Crossing · 4 pedestrians",
    source: "SET04 / VIDEO 0004",
  },
];

function fileSize(file: File) {
  return `${(file.size / (1024 * 1024)).toFixed(1)} MB`;
}

export function PreparedVideoPage() {
  const videoInput = useRef<HTMLInputElement>(null);
  const packageInput = useRef<HTMLInputElement>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [packageFile, setPackageFile] = useState<File | null>(null);
  const [upload, setUpload] = useState<PreparedVideoUploadResponse | null>(
    null,
  );
  const [prediction, setPrediction] = useState<ScenePredictionResponse | null>(
    null,
  );
  const [focusedPedestrianId, setFocusedPedestrianId] = useState("");
  const [working, setWorking] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const focusedPrediction =
    prediction?.predictions.find(
      (person) => person.pedestrian_id === focusedPedestrianId,
    ) ?? prediction?.predictions[0];

  async function uploadAndRun() {
    if (!videoFile || !packageFile) return;
    setWorking(true);
    setError("");
    setUpload(null);
    setPrediction(null);
    try {
      const uploaded = await api.uploadPreparedVideo(videoFile, packageFile);
      setUpload(uploaded);
      const result = await api.predictScene(uploaded.sample.id);
      setPrediction(result);
      setFocusedPedestrianId(result.predictions[0]?.pedestrian_id ?? "");
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Video analysis failed",
      );
    } finally {
      setWorking(false);
    }
  }

  async function runPrediction() {
    if (!upload) return;
    setPredicting(true);
    setError("");
    try {
      const result = await api.predictScene(upload.sample.id);
      setPrediction(result);
      setFocusedPedestrianId(result.predictions[0]?.pedestrian_id ?? "");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Prediction failed");
    } finally {
      setPredicting(false);
    }
  }

  function reset() {
    setVideoFile(null);
    setPackageFile(null);
    setUpload(null);
    setPrediction(null);
    setFocusedPedestrianId("");
    setError("");
    if (videoInput.current) videoInput.current.value = "";
    if (packageInput.current) packageInput.current.value = "";
  }

  return (
    <div className="app-shell">
      <AppHeader
        activePage="prepared-video"
        subtitle="PIE · Prepared Video Inference"
        actions={
          <StatusDot
            label={health ? "Exact-model pipeline online" : "Connecting"}
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

      <main className="prepared-video-page">
        <header className="prepared-video-intro">
          <div>
            <p className="eyebrow">Experimental presentation input</p>
            <h1>Upload a video with its exact observation data.</h1>
          </div>
          <p>
            The MP4 presents the 15-frame scene. Its matching package supplies
            the annotations, sensor features, masks, and model tensors—no YOLO
            approximation is used.
          </p>
        </header>

        {!upload ? (
          <>
            <section className="prepared-upload-panel panel">
              <div className="prepared-panel-heading">
                <div>
                  <p className="eyebrow">Matched input pair</p>
                  <h2>Select both files from one example folder</h2>
                </div>
                <span className="prepared-badge">15 FRAMES</span>
              </div>

              <div className="prepared-upload-grid">
                <div
                  className={`prepared-file-card ${videoFile ? "ready" : ""}`}
                >
                  <input
                    accept=".mp4,video/mp4"
                    aria-label="Choose prepared MP4 video"
                    disabled={working}
                    onChange={(event) =>
                      setVideoFile(event.target.files?.[0] ?? null)
                    }
                    ref={videoInput}
                    type="file"
                  />
                  <span className="prepared-file-index">01</span>
                  <div>
                    <span className="prepared-file-label">
                      Observation video
                    </span>
                    <strong>{videoFile?.name ?? "clip.mp4"}</strong>
                    <small>
                      {videoFile
                        ? fileSize(videoFile)
                        : "Annotated 15-frame MP4"}
                    </small>
                  </div>
                  <button
                    disabled={working}
                    onClick={() => videoInput.current?.click()}
                    type="button"
                  >
                    {videoFile ? "Change" : "Choose video"}
                  </button>
                </div>

                <div
                  className={`prepared-file-card ${packageFile ? "ready" : ""}`}
                >
                  <input
                    accept=".zip,application/zip"
                    aria-label="Choose companion data package"
                    disabled={working}
                    onChange={(event) =>
                      setPackageFile(event.target.files?.[0] ?? null)
                    }
                    ref={packageInput}
                    type="file"
                  />
                  <span className="prepared-file-index">02</span>
                  <div>
                    <span className="prepared-file-label">Companion data</span>
                    <strong>{packageFile?.name ?? "clip_data.zip"}</strong>
                    <small>
                      {packageFile
                        ? fileSize(packageFile)
                        : "Exact tensors, metadata and annotations"}
                    </small>
                  </div>
                  <button
                    disabled={working}
                    onClick={() => packageInput.current?.click()}
                    type="button"
                  >
                    {packageFile ? "Change" : "Choose data"}
                  </button>
                </div>
              </div>

              <div className="prepared-validation-note">
                <span>SHA-256 pair validation</span>
                <p>
                  The server rejects a video paired with data from a different
                  scene before registering or running it.
                </p>
              </div>

              <button
                className="prepared-run-button primary-action"
                disabled={!videoFile || !packageFile || working}
                onClick={uploadAndRun}
                type="button"
              >
                <span>
                  {working
                    ? "Validating pair and running all pedestrians"
                    : "Upload pair and run the model"}
                </span>
                <span aria-hidden="true">{working ? "···" : "→"}</span>
              </button>
            </section>

            <section className="prepared-examples">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Ready to test</p>
                  <h2>Five matched PIE examples</h2>
                </div>
                <p>Download both files from one row, then upload them above.</p>
              </div>
              <div className="prepared-example-grid">
                {EXAMPLES.map((example, index) => {
                  const root = `/prepared-video-examples/${example.folder}`;
                  return (
                    <article
                      className="prepared-example-card panel"
                      key={example.folder}
                    >
                      <span className="prepared-example-number">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <div>
                        <strong>{example.title}</strong>
                        <small>{example.source}</small>
                      </div>
                      <div className="prepared-example-links">
                        <a download href={`${root}/clip.mp4`}>
                          Video
                        </a>
                        <a download href={`${root}/clip_data.zip`}>
                          Data
                        </a>
                      </div>
                    </article>
                  );
                })}
              </div>
            </section>
          </>
        ) : (
          <>
            <div className="prepared-result-toolbar">
              <div>
                <p className="eyebrow">Validated prepared input</p>
                <h2>{upload.sample.display_name}</h2>
              </div>
              <button
                className="secondary-action"
                onClick={reset}
                type="button"
              >
                Upload another pair
              </button>
            </div>

            <div className="prepared-result-grid">
              <section className="prepared-video-viewer panel">
                <div className="viewer-toolbar">
                  <div>
                    <p className="eyebrow">Observation video</p>
                    <h2>
                      {upload.sample.set_id.toUpperCase()} ·{" "}
                      {upload.sample.video_id.replace("_", " ").toUpperCase()}
                    </h2>
                  </div>
                  <div className="viewer-tags">
                    <span>{upload.video.frame_count} frames</span>
                    <span>{upload.sample.pedestrian_count} pedestrians</span>
                  </div>
                </div>
                <video
                  autoPlay
                  controls
                  loop
                  muted
                  playsInline
                  poster={upload.sample.frames[7]?.url}
                  src={upload.video.url}
                />
                <div className="prepared-video-meta">
                  <span>
                    Display {upload.video.playback_fps.toFixed(0)} FPS · Source{" "}
                    {upload.video.source_fps.toFixed(0)} FPS
                  </span>
                  <span>Verified video/data fingerprint</span>
                </div>
              </section>

              <PredictionPanel
                focusedPedestrianId={focusedPrediction?.pedestrian_id ?? ""}
                onFocusPedestrian={setFocusedPedestrianId}
                onPredict={runPrediction}
                prediction={prediction}
                predicting={predicting || working}
                sample={upload.sample}
              />
            </div>

            <section className="prepared-provenance panel">
              <div>
                <span>Video role</span>
                <strong>15-frame scene presentation</strong>
              </div>
              <div>
                <span>Data role</span>
                <strong>Exact PIE annotations and sensor tensors</strong>
              </div>
              <div>
                <span>Inference</span>
                <strong>Parallel batch + global safety OR</strong>
              </div>
            </section>

            <section
              className="model-section"
              aria-label="Source model outputs"
            >
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Inside the prepared-video pipeline</p>
                  <h2>Source model signals</h2>
                </div>
                <p>Exact checkpoint inputs from the companion data package.</p>
              </div>
              <div className="model-grid">
                <ModelCard
                  index="01"
                  output={focusedPrediction?.trajectory}
                  subtitle="15 × 8 trajectory-context graph"
                  title="Trajectory context"
                />
                <div className="fusion-connector" aria-hidden="true">
                  <span>+</span>
                </div>
                <ModelCard
                  index="02"
                  output={focusedPrediction?.social}
                  subtitle="15 × 12 interaction graph"
                  title="Social interaction"
                />
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}
