export type ClassName = "No Crossing" | "Crossing" | "Irrelevant";
export type FilterName = "All" | ClassName;
export type ClassValues = Record<ClassName, number>;

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  sample_count: number;
  device: string;
  models: Record<
    string,
    { file: string; sha256: string; selected_epoch: number }
  >;
}

export interface SampleSummary {
  id: string;
  display_name: string;
  sample_index: number;
  set_id: string;
  video_id: string;
  pedestrian_id: string;
  class_name: ClassName;
  raw_class: number;
  binary_target: number;
  pedestrian_count: number;
  poster_url: string;
  source: "prepared" | "upload";
}

export interface PedestrianFrameBox {
  pedestrian_id: string;
  box: number[] | null;
  color: string;
}

export interface FrameInfo {
  ordinal: number;
  source_frame: number;
  url: string;
  target_box: number[] | null;
  pedestrian_boxes: PedestrianFrameBox[];
}

export interface PedestrianInfo {
  tensor_index: number;
  sample_index: number;
  pedestrian_id: string;
  class_name: ClassName;
  raw_class: number;
  binary_target: number;
  color: string;
}

export interface SampleDetail extends Omit<SampleSummary, "poster_url"> {
  tensor_index: number;
  pedestrians: PedestrianInfo[];
  frames: FrameInfo[];
}

export interface PackageUploadResponse {
  summary: SampleSummary;
  sample: SampleDetail;
}

export interface SourceOutput {
  name: string;
  checkpoint: string;
  logits: ClassValues;
  probabilities: ClassValues;
}

export interface FusionOutput {
  name: string;
  weights: {
    no_crossing: number;
    crossing: number;
    irrelevant: number;
  };
  logits: ClassValues;
  probabilities: ClassValues;
  three_class_prediction: ClassName;
  binary_score: number;
  binary_prediction: number;
  binary_label: "Crossing" | "Not Crossing";
  threshold: number;
}

export interface PredictionResponse {
  sample_id: string;
  sample_index: number;
  pedestrian_id: string;
  ground_truth: {
    raw_class: number;
    class_name: ClassName;
    binary_target: number;
    binary_label: "Crossing" | "Not Crossing";
  };
  trajectory: SourceOutput;
  social: SourceOutput;
  fusion: FusionOutput;
  is_correct: boolean;
  latency_ms: number;
}

export interface ScenePredictionResponse {
  sample_id: string;
  pedestrian_count: number;
  predictions: PredictionResponse[];
  global_decision: {
    binary_prediction: number;
    binary_label: "Crossing" | "Not Crossing";
    binary_score: number;
    threshold: number;
    crossing_pedestrian_ids: string[];
    not_crossing_pedestrian_ids: string[];
    ground_truth_binary_target: number;
    ground_truth_binary_label: "Crossing" | "Not Crossing";
    is_correct: boolean;
  };
  latency_ms: number;
}

export interface BatchPredictionResponse {
  predictions: ScenePredictionResponse[];
  correct: number;
  total: number;
  accuracy: number;
  latency_ms: number;
}

export interface ComparisonMetrics {
  binary_macro_f1: number;
  crossing_recall: number;
  crossing_f1: number;
  three_class_macro_f1: number;
  pr_auc: number;
}

export interface ComparisonMethod {
  id: string;
  name: string;
  short_name: string;
  architecture: string;
  role: string;
  selection_note: string;
  is_final: boolean;
  metrics: ComparisonMetrics;
  checkpoint: string;
  logits: ClassValues;
  probabilities: ClassValues;
  three_class_prediction: ClassName;
  binary_score: number;
  binary_prediction: number;
  binary_label: "Crossing" | "Not Crossing";
  threshold: number;
  is_correct: boolean;
}

export interface ComparisonResponse {
  sample_id: string;
  sample_index: number;
  ground_truth: PredictionResponse["ground_truth"];
  methods: ComparisonMethod[];
  agreement: {
    unanimous: boolean;
    crossing_votes: number;
    not_crossing_votes: number;
    correct_methods: number;
    total_methods: number;
  };
  latency_ms: number;
}
