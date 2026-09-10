import type {
  BatchPredictionResponse,
  ComparisonResponse,
  HealthResponse,
  PredictionResponse,
  SampleDetail,
  SampleSummary,
} from "../types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const message =
      payload?.detail ?? `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  samples: () => request<SampleSummary[]>("/api/samples"),
  sample: (id: string) => request<SampleDetail>(`/api/samples/${id}`),
  predict: (id: string) =>
    request<PredictionResponse>(`/api/samples/${id}/predict`, {
      method: "POST",
    }),
  predictAll: () =>
    request<BatchPredictionResponse>("/api/predict-all", { method: "POST" }),
  compare: (id: string) =>
    request<ComparisonResponse>(`/api/comparison/samples/${id}/predict`, {
      method: "POST",
    }),
};
