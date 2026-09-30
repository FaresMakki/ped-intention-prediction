"""Typed HTTP response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    status: str
    models_loaded: bool
    sample_count: int
    device: str
    models: dict[str, Any]


class SampleSummary(ApiModel):
    id: str
    display_name: str
    sample_index: int
    set_id: str
    video_id: str
    pedestrian_id: str
    class_name: str
    raw_class: int
    binary_target: int
    pedestrian_count: int
    poster_url: str
    source: str


class PedestrianFrameBox(ApiModel):
    pedestrian_id: str
    box: list[float] | None
    color: str


class FrameInfo(ApiModel):
    ordinal: int
    source_frame: int
    url: str
    target_box: list[float] | None
    pedestrian_boxes: list[PedestrianFrameBox]


class PedestrianInfo(ApiModel):
    tensor_index: int
    sample_index: int
    pedestrian_id: str
    class_name: str
    raw_class: int
    binary_target: int
    color: str


class SampleDetail(ApiModel):
    id: str
    display_name: str
    tensor_index: int
    sample_index: int
    set_id: str
    video_id: str
    pedestrian_id: str
    class_name: str
    raw_class: int
    binary_target: int
    pedestrian_count: int
    pedestrians: list[PedestrianInfo]
    frames: list[FrameInfo]
    source: str


class PackageUploadResponse(ApiModel):
    summary: SampleSummary
    sample: SampleDetail


class PredictionResponse(ApiModel):
    sample_id: str
    sample_index: int
    pedestrian_id: str
    ground_truth: dict[str, Any]
    trajectory: dict[str, Any]
    social: dict[str, Any]
    fusion: dict[str, Any]
    is_correct: bool
    latency_ms: float


class ScenePredictionResponse(ApiModel):
    sample_id: str
    pedestrian_count: int
    predictions: list[PredictionResponse]
    global_decision: dict[str, Any]
    latency_ms: float


class BatchPredictionResponse(ApiModel):
    predictions: list[ScenePredictionResponse]
    correct: int
    total: int
    accuracy: float
    latency_ms: float


class ComparisonResponse(ApiModel):
    sample_id: str
    pedestrian_count: int
    pedestrians: list[dict[str, Any]]
    global_methods: list[dict[str, Any]]
    global_ground_truth: dict[str, Any]
    agreement: dict[str, Any]
    latency_ms: float
