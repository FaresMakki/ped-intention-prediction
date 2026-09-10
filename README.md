# PIE Crossing Intelligence Lab

This folder is a self-contained deployment of the final locked **Trajectory +
Social classwise fusion (Rule B)** and a separate research comparison view. It
uses 15 prepared sequences from the PIE test split and performs live checkpoint
inference when the user starts an analysis.

Runtime inference never reads saved logits. Each request loads the aligned graph
or skeleton tensors, executes the copied PyTorch checkpoints, and applies the
same fixed binary mapping used by the experiments.

## Pages

- **Live fusion** (`#/`) preserves the original final-model experience. It shows
  the Trajectory and Social branch outputs and their locked Rule B result.
- **Compare models** (`#/comparison`) runs five methods on the same selected
  sequence: final Trajectory + Social, Relation-Risk, Environment,
  Environment + Skeleton, and Skeleton.

The comparison page separates two kinds of evidence:

- logits, probabilities, and decisions are computed live for the selected
  sequence;
- reference F1, recall, and PR AUC values come from each validation-selected
  checkpoint evaluated on the complete 715-sample test set.

## Deployed checkpoints

- `models/trajectory_model.pt` — trajectory-context graph.
- `models/social_model.pt` — social-interaction graph.
- `models/environment_model.pt` — protected Environment Graph, epoch 10.
- `models/skeleton_model.pt` — adaptive 19-joint PedGraphPlus GCN, epoch 61.
- `models/environment_skeleton_fusion_model.pt` — validation-selected
  uncertainty-gated residual correction, epoch 7.
- `models/relation_risk_model.pt` — Relation-Risk Graph, epoch 14.

The Environment + Skeleton comparison deliberately uses the validation-selected
variant. A test-only descriptive alternative is not deployed because test data
must not be used to choose a model.

## Project structure

- `backend/app/api/` — FastAPI routes.
- `backend/app/ml/` — exact inference architectures, checkpoint loading, and
  fusion functions.
- `backend/app/repositories/` — validated access to prepared sample tensors.
- `backend/app/services/` — final-model inference and comparison orchestration.
- `backend/tests/` — API, fusion, and original-artifact parity tests.
- `frontend/src/pages/` — independent live-fusion and comparison pages.
- `data/` — 15 aligned model-input samples and 225 annotated frames.
- `scripts/prepare_demo_data.py` — reproducible aligned sample exporter.

The demo is balanced: five raw No Crossing, five Crossing, and five Irrelevant
examples. Selection takes the first indexed test sample from five distinct
videos for each raw class and never filters by prediction correctness.

## Final inference path

```text
15-frame PIE sequence
        |
        +-- trajectory graph [15, 8, 64] --> trajectory_model.pt --+
        |                                                         |
        +-- social graph [15, 12, 80] ------> social_model.pt -----+
                                                                  |
                                      Rule B classwise logit fusion
                                      NC: 75% trajectory + 25% social
                                      C:  100% trajectory
                                      IRR: 80% trajectory + 20% social
                                                                  |
                                                 fixed binary mapping
                                                                  |
                                                Crossing / Not Crossing
```

The three class outputs use independent sigmoids, so they are not a softmax
distribution and do not need to sum to 100%.

## Run locally

From PowerShell in this directory:

```powershell
.\setup.ps1
.\run.ps1
```

Open `http://127.0.0.1:8000`. The production command serves the compiled React
application, media, and API from the same FastAPI process.

For frontend hot reload during development:

```powershell
.\run-dev.ps1
```

Vite is then available at `http://127.0.0.1:5173` and proxies API and media
requests to port 8000.

## API

- `GET /api/health` — all checkpoint hashes, epochs, and service readiness.
- `GET /api/samples` — the 15 example summaries.
- `GET /api/samples/{id}` — frames and metadata for one example.
- `POST /api/samples/{id}/predict` — live final-model inference.
- `POST /api/predict-all` — evaluate the final model on all prepared examples.
- `POST /api/comparison/samples/{id}/predict` — run the five-method comparison.
- `GET /openapi.json` — OpenAPI schema.

## Verification

```powershell
..\.venv\Scripts\python.exe -m pytest backend
cd frontend
npm run build
```

The parity tests execute every displayed checkpoint on all 15 examples and
compare deployment logits with the original research artifacts. They also check
the final Rule B predictions and the HTTP response contracts.

## Docker

```powershell
docker build -t pie-crossing-demo .
docker run --rm -p 8000:8000 pie-crossing-demo
```

For a CPU-only image, configure the appropriate PyTorch package index for the
target host if image size matters.

## Railway

Create one Railway service from this repository. Railway detects the root
`Dockerfile`, builds the React frontend, and serves the interface and FastAPI
backend from one container. The container listens on Railway's injected `PORT`
and falls back to port 8000 locally.

After the first successful deployment, open **Settings → Networking** in
Railway and generate a public domain. No frontend API URL or CORS variable is
required because the frontend, API, and media are served from the same origin.
