"""FastAPI application factory and static production host."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.repositories.sample_repository import SampleRepository
from app.services.comparison_service import ComparisonService
from app.services.inference_service import InferenceService


@asynccontextmanager
async def lifespan(app: FastAPI):
    repository = SampleRepository(settings.manifest_path, settings.tensors_path)
    app.state.inference_service = InferenceService(
        repository=repository,
        trajectory_checkpoint=settings.trajectory_checkpoint,
        social_checkpoint=settings.social_checkpoint,
        device_name=settings.device,
    )
    app.state.comparison_service = ComparisonService(
        repository=repository,
        primary_service=app.state.inference_service,
        environment_checkpoint=settings.environment_checkpoint,
        skeleton_checkpoint=settings.skeleton_checkpoint,
        fusion_checkpoint=settings.environment_skeleton_checkpoint,
        relation_risk_checkpoint=settings.relation_risk_checkpoint,
        device_name=settings.device,
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="PIE Locked Fusion API",
        version="1.1.0",
        description="Live inference and comparison across selected PIE graph models.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.mount("/media", StaticFiles(directory=settings.media_dir), name="media")
    if settings.frontend_dist.exists():
        app.mount(
            "/",
            StaticFiles(directory=settings.frontend_dist, html=True),
            name="frontend",
        )
    return app


app = create_app()
