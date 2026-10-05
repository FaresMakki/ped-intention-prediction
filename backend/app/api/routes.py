"""Public API for health, samples, and live checkpoint inference."""

from __future__ import annotations

from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)

from app.schemas import (
    BatchPredictionResponse,
    ComparisonResponse,
    HealthResponse,
    PackageUploadResponse,
    PredictionResponse,
    PreparedVideoUploadResponse,
    SampleDetail,
    SampleSummary,
    ScenePredictionResponse,
)
from app.services.comparison_service import ComparisonService
from app.services.inference_service import InferenceService
from app.services.package_service import (
    MAX_PACKAGE_BYTES,
    PackageValidationError,
    load_package,
)
from app.services.prepared_video_service import MAX_VIDEO_BYTES, load_prepared_video

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


async def _read_upload(upload: UploadFile, limit: int, label: str) -> bytes:
    payload = bytearray()
    try:
        while chunk := await upload.read(1024 * 1024):
            payload.extend(chunk)
            if len(payload) > limit:
                raise HTTPException(status_code=413, detail=f"{label} exceeds 25 MB.")
    finally:
        await upload.close()
    return bytes(payload)


@router.post(
    "/prepared-video/uploads",
    response_model=PreparedVideoUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_prepared_video(
    request: Request,
    video: Annotated[UploadFile, File(description="Prepared 15-frame MP4")],
    package: Annotated[UploadFile, File(description="Matching companion data ZIP")],
) -> dict:
    """Validate a matched MP4/data pair and register it for live inference."""
    video_file_name = video.filename
    video_payload = await _read_upload(video, MAX_VIDEO_BYTES, "Video")
    package_payload = await _read_upload(package, MAX_PACKAGE_BYTES, "Package")
    try:
        loaded = load_package(package_payload)
        uploaded_video = load_prepared_video(
            video_payload,
            video_file_name,
            loaded.video,
        )
    except PackageValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    repository = _service(request).repository
    sample = repository.add_uploaded_sample(
        loaded.metadata,
        loaded.tensors,
        loaded.frames,
        source="prepared-video",
        video=uploaded_video,
    )
    sample_id = sample["id"]
    return {
        "summary": repository.get_summary(sample_id),
        "sample": sample,
        "video": {
            "file_name": uploaded_video.file_name,
            "url": f"/api/prepared-video/uploads/{sample_id}/video",
            "sha256": uploaded_video.sha256,
            "frame_count": uploaded_video.frame_count,
            "playback_fps": uploaded_video.playback_fps,
            "source_fps": uploaded_video.source_fps,
            "width": uploaded_video.width,
            "height": uploaded_video.height,
        },
    }


@router.get("/prepared-video/uploads/{sample_id}/video")
def uploaded_video(sample_id: str, request: Request) -> Response:
    try:
        video = _service(request).repository.get_uploaded_video(sample_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return Response(
        content=video.content,
        media_type=video.media_type,
        headers={
            "Cache-Control": "private, max-age=3600",
            "Content-Disposition": f'inline; filename="{video.file_name}"',
        },
    )


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


@router.post(
    "/samples/{sample_id}/predict-scene",
    response_model=ScenePredictionResponse,
)
def predict_scene(sample_id: str, request: Request) -> dict:
    """Run one batched forward pass for every annotated pedestrian."""
    try:
        return _service(request).predict_scene(sample_id)
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
