"""Validation for MP4 files paired with exact-model PIE data packages."""

from __future__ import annotations

import hashlib
from pathlib import PurePath

from app.repositories.sample_repository import UploadedVideo
from app.services.package_service import PackageValidationError, PackageVideo

MAX_VIDEO_BYTES = 25 * 1024 * 1024


def load_prepared_video(
    payload: bytes,
    file_name: str | None,
    expected: PackageVideo | None,
) -> UploadedVideo:
    """Authenticate one small MP4 against metadata in its companion package."""
    if expected is None:
        raise PackageValidationError(
            "Companion package has no prepared-video metadata. "
            "Use one of the video example data packages."
        )
    if not payload:
        raise PackageValidationError("Uploaded video is empty.")
    if len(payload) > MAX_VIDEO_BYTES:
        raise PackageValidationError("Video exceeds the 25 MB limit.")
    safe_name = PurePath(file_name or "").name
    if safe_name != file_name or safe_name != expected.file_name:
        raise PackageValidationError(
            f"Video file name must be {expected.file_name!r} for this package."
        )
    if len(payload) < 12 or payload[4:8] != b"ftyp":
        raise PackageValidationError("Uploaded video is not a valid MP4 container.")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected.sha256:
        raise PackageValidationError(
            "Video and companion data do not match. Select both files from "
            "the same prepared example folder."
        )
    return UploadedVideo(
        content=payload,
        file_name=expected.file_name,
        sha256=digest,
        frame_count=expected.frame_count,
        playback_fps=expected.playback_fps,
        source_fps=expected.source_fps,
        width=expected.width,
        height=expected.height,
    )
