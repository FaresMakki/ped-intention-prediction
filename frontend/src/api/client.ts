import type {
  BatchPredictionResponse,
  ComparisonResponse,
  HealthResponse,
  PackageUploadResponse,
  PreparedVideoUploadResponse,
  PredictionResponse,
  ScenePredictionResponse,
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
  predictScene: (id: string) =>
    request<ScenePredictionResponse>(`/api/samples/${id}/predict-scene`, {
      method: "POST",
    }),
  predictAll: () =>
    request<BatchPredictionResponse>("/api/predict-all", { method: "POST" }),
  compare: (id: string) =>
    request<ComparisonResponse>(`/api/comparison/samples/${id}/predict`, {
      method: "POST",
    }),
  uploadPackage: (file: File) =>
    request<PackageUploadResponse>("/api/uploads", {
      method: "POST",
      headers: {
        "Content-Type": "application/zip",
        "X-File-Name": encodeURIComponent(file.name),
      },
      body: file,
    }),
  uploadPreparedVideo: (video: File, packageFile: File) => {
    const body = new FormData();
    body.append("video", video);
    body.append("package", packageFile);
    return request<PreparedVideoUploadResponse>("/api/prepared-video/uploads", {
      method: "POST",
      body,
    });
  },
};
