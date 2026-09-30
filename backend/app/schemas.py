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
    poster_url: str
    source: str


class FrameInfo(ApiModel):
    ordinal: int
    source_frame: int
    url: str
    target_box: list[float] | None


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
    frames: list[FrameInfo]
    source: str


class PackageUploadResponse(ApiModel):
    summary: SampleSummary
    sample: SampleDetail


class PredictionResponse(ApiModel):
    sample_id: str
    sample_index: int
    ground_truth: dict[str, Any]
    trajectory: dict[str, Any]
    social: dict[str, Any]
    fusion: dict[str, Any]
    is_correct: bool
    latency_ms: float


class BatchPredictionResponse(ApiModel):
    predictions: list[PredictionResponse]
    correct: int
    total: int
    accuracy: float
    latency_ms: float


class ComparisonResponse(ApiModel):
    sample_id: str
    sample_index: int
    ground_truth: dict[str, Any]
    methods: list[dict[str, Any]]
    agreement: dict[str, Any]
    latency_ms: float
