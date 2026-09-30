"""Validation and loading for portable PIE inference packages."""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.repositories.sample_repository import SampleTensors, UploadedFrame

PACKAGE_FORMAT = "pie-crossing-package"
PACKAGE_VERSION = 1
FRAME_COUNT = 15
MAX_PACKAGE_BYTES = 25 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_TENSOR_ARCHIVE_BYTES = 4 * 1024 * 1024

EXPECTED_SHAPES = {
    "trajectory_nodes": (15, 8, 64),
    "trajectory_adjacency": (15, 8, 8),
    "trajectory_mask": (15, 8),
    "social_nodes": (15, 12, 80),
    "social_adjacency": (15, 12, 12),
    "social_mask": (15, 12),
    "environment_nodes": (15, 12, 37),
    "environment_adjacency": (15, 12, 12),
    "environment_mask": (15, 12),
    "skeleton_input": (15, 19, 4),
    "skeleton_keypoint_mask": (15, 19),
    "skeleton_frame_mask": (15,),
    "relation_risk_nodes": (15, 8, 64),
    "relation_risk_adjacency": (15, 8, 8),
    "relation_risk_mask": (15, 8),
}
MASK_ARRAYS = {
    "trajectory_mask",
    "social_mask",
    "environment_mask",
    "skeleton_keypoint_mask",
    "skeleton_frame_mask",
    "relation_risk_mask",
}
ADJACENCY_ARRAYS = {
    "trajectory_adjacency",
    "social_adjacency",
    "environment_adjacency",
    "relation_risk_adjacency",
}
CLASS_BY_RAW = {-1: "Irrelevant", 0: "No Crossing", 1: "Crossing"}


class PackageValidationError(ValueError):
    """Raised when an uploaded archive violates the versioned contract."""


class PackageFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordinal: int = Field(ge=1, le=FRAME_COUNT)
    source_frame: int = Field(ge=0)
    file: str = Field(min_length=1, max_length=160)
    target_box: list[float] | None

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}
        ):
            raise ValueError("frame file must be a safe JPEG or PNG path")
        return value

    @field_validator("target_box")
    @classmethod
    def validate_target_box(
        cls, value: list[float] | None
    ) -> list[float] | None:
        if value is None:
            return value
        if len(value) != 4 or not all(math.isfinite(item) for item in value):
            raise ValueError("target_box must contain four finite coordinates")
        x1, y1, x2, y2 = value
        if x2 <= x1 or y2 <= y1:
            raise ValueError("target_box must have positive width and height")
        return value


class PackageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    format_name: Literal[PACKAGE_FORMAT] = Field(alias="format")
    version: Literal[PACKAGE_VERSION]
    frame_count: Literal[FRAME_COUNT]
    tensors_file: Literal["tensors.npz"]
    display_name: str = Field(min_length=1, max_length=80)
    sample_index: int = Field(ge=0)
    set_id: str = Field(pattern=r"^set\d{2}$")
    video_id: str = Field(pattern=r"^video_\d{4}$")
    pedestrian_id: str = Field(min_length=1, max_length=64)
    class_name: Literal["No Crossing", "Crossing", "Irrelevant"]
    raw_class: Literal[-1, 0, 1]
    binary_target: Literal[0, 1]
    frames: list[PackageFrame] = Field(min_length=FRAME_COUNT, max_length=FRAME_COUNT)


@dataclass(frozen=True)
class LoadedPackage:
    metadata: dict
    tensors: SampleTensors
    frames: dict[int, UploadedFrame]


def _safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    files: dict[str, zipfile.ZipInfo] = {}
    total_size = 0
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise PackageValidationError("Package contains an unsafe file path.")
        if info.filename in files:
            raise PackageValidationError("Package contains duplicate file names.")
        if info.flag_bits & 0x1:
            raise PackageValidationError("Encrypted ZIP entries are not supported.")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise PackageValidationError("Unsupported ZIP compression method.")
        total_size += info.file_size
        if total_size > MAX_UNCOMPRESSED_BYTES:
            raise PackageValidationError("Uncompressed package is too large.")
        files[info.filename] = info
    return files


def _load_manifest(archive: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo]):
    try:
        info = members["manifest.json"]
    except KeyError as error:
        raise PackageValidationError("Package is missing manifest.json.") from error
    if info.file_size > 128 * 1024:
        raise PackageValidationError("manifest.json is too large.")
    try:
        payload = json.loads(archive.read(info))
        manifest = PackageManifest.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise PackageValidationError(f"Invalid manifest.json: {error}") from error

    ordinals = [frame.ordinal for frame in manifest.frames]
    frame_files = [frame.file for frame in manifest.frames]
    source_frames = [frame.source_frame for frame in manifest.frames]
    if sorted(ordinals) != list(range(1, FRAME_COUNT + 1)):
        raise PackageValidationError("Frame ordinals must be exactly 1 through 15.")
    if len(set(frame_files)) != FRAME_COUNT:
        raise PackageValidationError("Every frame must reference a unique image file.")
    if any(right != left + 1 for left, right in zip(source_frames, source_frames[1:])):
        raise PackageValidationError("Source frames must be 15 consecutive frame IDs.")
    if manifest.class_name != CLASS_BY_RAW[manifest.raw_class]:
        raise PackageValidationError("class_name does not match raw_class.")
    expected_binary = int(manifest.raw_class == 1)
    if manifest.binary_target != expected_binary:
        raise PackageValidationError("binary_target does not match raw_class.")

    expected_members = {"manifest.json", manifest.tensors_file, *frame_files}
    actual_members = set(members)
    if actual_members != expected_members:
        missing = sorted(expected_members - actual_members)
        extra = sorted(actual_members - expected_members)
        raise PackageValidationError(
            f"Package file set does not match manifest; missing={missing}, extra={extra}."
        )
    return manifest


def _load_tensors(blob: bytes) -> SampleTensors:
    if len(blob) > MAX_TENSOR_ARCHIVE_BYTES:
        raise PackageValidationError("tensors.npz is too large.")
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as inner:
            members = _safe_members(inner)
            if sum(info.file_size for info in members.values()) > MAX_TENSOR_ARCHIVE_BYTES:
                raise PackageValidationError("Expanded tensors.npz is too large.")
        with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
            actual = set(archive.files)
            expected = set(EXPECTED_SHAPES)
            if actual != expected:
                raise PackageValidationError(
                    "Tensor names do not match the package contract; "
                    f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
                )
            arrays = {name: np.array(archive[name], copy=True) for name in expected}
    except PackageValidationError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise PackageValidationError(f"Invalid tensors.npz: {error}") from error

    for name, expected_shape in EXPECTED_SHAPES.items():
        array = arrays[name]
        if array.shape != expected_shape:
            raise PackageValidationError(
                f"{name} has shape {array.shape}; expected {expected_shape}."
            )
        if name in MASK_ARRAYS:
            if array.dtype != np.uint8 or not np.isin(array, (0, 1)).all():
                raise PackageValidationError(f"{name} must be a binary uint8 mask.")
        elif array.dtype != np.float32:
            raise PackageValidationError(f"{name} must use float32 values.")
        if not np.isfinite(array).all():
            raise PackageValidationError(f"{name} contains non-finite values.")
        if array.dtype == np.float32 and np.max(np.abs(array), initial=0.0) > 1e4:
            raise PackageValidationError(f"{name} contains out-of-range values.")
        if name in ADJACENCY_ARRAYS and (
            np.min(array, initial=0.0) < 0.0 or np.max(array, initial=0.0) > 1.0
        ):
            raise PackageValidationError(f"{name} must contain weights from 0 to 1.")
    return SampleTensors(**arrays)


def _image_type(path: str, content: bytes) -> str:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"} and content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if suffix == ".png" and content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    raise PackageValidationError(f"Frame {path} is not a valid declared image type.")


def load_package(payload: bytes) -> LoadedPackage:
    """Validate an uploaded ZIP and return detached model-ready objects."""
    if not payload:
        raise PackageValidationError("Uploaded package is empty.")
    if len(payload) > MAX_PACKAGE_BYTES:
        raise PackageValidationError("Package exceeds the 25 MB limit.")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = _safe_members(archive)
            manifest = _load_manifest(archive, members)
            tensors = _load_tensors(archive.read(manifest.tensors_file))
            frames: dict[int, UploadedFrame] = {}
            frame_metadata = []
            for frame in sorted(manifest.frames, key=lambda item: item.ordinal):
                content = archive.read(frame.file)
                if not content or len(content) > 5 * 1024 * 1024:
                    raise PackageValidationError(
                        f"Frame {frame.file} is empty or exceeds 5 MB."
                    )
                frames[frame.ordinal] = UploadedFrame(
                    content=content,
                    media_type=_image_type(frame.file, content),
                )
                frame_metadata.append(
                    {
                        "ordinal": frame.ordinal,
                        "source_frame": frame.source_frame,
                        "target_box": frame.target_box,
                    }
                )
    except PackageValidationError:
        raise
    except zipfile.BadZipFile as error:
        raise PackageValidationError("Uploaded file is not a valid ZIP archive.") from error

    metadata = {
        "display_name": manifest.display_name,
        "sample_index": manifest.sample_index,
        "set_id": manifest.set_id,
        "video_id": manifest.video_id,
        "pedestrian_id": manifest.pedestrian_id,
        "class_name": manifest.class_name,
        "raw_class": manifest.raw_class,
        "binary_target": manifest.binary_target,
        "frames": frame_metadata,
    }
    return LoadedPackage(metadata=metadata, tensors=tensors, frames=frames)
