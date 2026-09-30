"""Public API for health, samples, and live checkpoint inference."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status

from app.schemas import (
    BatchPredictionResponse,
    ComparisonResponse,
    HealthResponse,
    PackageUploadResponse,
    PredictionResponse,
    SampleDetail,
    SampleSummary,
)
from app.services.comparison_service import ComparisonService
from app.services.inference_service import InferenceService
from app.services.package_service import (
    MAX_PACKAGE_BYTES,
    PackageValidationError,
    load_package,
)

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


@router.post(
    "/uploads",
    response_model=PackageUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_package(request: Request) -> dict:
    """Validate and register one versioned application/zip package."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_PACKAGE_BYTES:
                raise HTTPException(status_code=413, detail="Package exceeds 25 MB.")
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid Content-Length.") from error

    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > MAX_PACKAGE_BYTES:
            raise HTTPException(status_code=413, detail="Package exceeds 25 MB.")
    try:
        loaded = load_package(bytes(payload))
    except PackageValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    repository = _service(request).repository
    sample = repository.add_uploaded_sample(
        loaded.metadata,
        loaded.tensors,
        loaded.frames,
    )
    return {"summary": repository.get_summary(sample["id"]), "sample": sample}


@router.get("/uploads/{sample_id}/frames/{ordinal}")
def uploaded_frame(sample_id: str, ordinal: int, request: Request) -> Response:
    try:
        frame = _service(request).repository.get_uploaded_frame(sample_id, ordinal)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return Response(
        content=frame.content,
        media_type=frame.media_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


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
