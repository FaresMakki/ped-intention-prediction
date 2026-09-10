"""Public API for health, samples, and live checkpoint inference."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas import (
    BatchPredictionResponse,
    ComparisonResponse,
    HealthResponse,
    PredictionResponse,
    SampleDetail,
    SampleSummary,
)
from app.services.comparison_service import ComparisonService
from app.services.inference_service import InferenceService

router = APIRouter(prefix="/api")


def _service(request: Request) -> InferenceService:
    return request.app.state.inference_service


def _comparison_service(request: Request) -> ComparisonService:
    return request.app.state.comparison_service


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> dict:
    service = _service(request)
    return {
        "status": "ready",
        "models_loaded": True,
        "sample_count": len(service.repository.list_samples()),
        "device": str(service.device),
        "models": {
            **service.model_info,
            **_comparison_service(request).model_info,
        },
    }


@router.get("/samples", response_model=list[SampleSummary])
def list_samples(request: Request) -> list[dict]:
    return _service(request).repository.list_samples()


@router.get("/samples/{sample_id}", response_model=SampleDetail)
def sample_detail(sample_id: str, request: Request) -> dict:
    try:
        return _service(request).repository.get_metadata(sample_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/samples/{sample_id}/predict", response_model=PredictionResponse)
def predict(sample_id: str, request: Request) -> dict:
    try:
        return _service(request).predict(sample_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/predict-all", response_model=BatchPredictionResponse)
def predict_all(request: Request) -> dict:
    return _service(request).predict_all()


@router.post(
    "/comparison/samples/{sample_id}/predict",
    response_model=ComparisonResponse,
)
def compare(sample_id: str, request: Request) -> dict:
    try:
        return _comparison_service(request).predict(sample_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
