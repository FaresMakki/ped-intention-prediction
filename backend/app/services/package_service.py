"""Validation and loading for portable PIE inference packages."""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.repositories.sample_repository import SampleTensors, UploadedFrame

PACKAGE_FORMAT = "pie-crossing-package"
PACKAGE_VERSION = 2
LEGACY_PACKAGE_VERSION = 1
FRAME_COUNT = 15
MAX_PEDESTRIANS = 16
MAX_PACKAGE_BYTES = 25 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 80 * 1024 * 1024
MAX_TENSOR_ARCHIVE_BYTES = 16 * 1024 * 1024

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
PEDESTRIAN_COLORS = (
    "#B1EF68",
    "#67B7FF",
    "#F7B955",
    "#E788FF",
    "#FF7A90",
    "#65E3C4",
    "#FF9E64",
    "#9EA7FF",
    "#D9E76C",
    "#62D5F5",
    "#E9A1FF",
    "#FFB86B",
    "#73E68C",
    "#FF87C8",
    "#87A8FF",
    "#EFD66F",
)


class PackageValidationError(ValueError):
    """Raised when an uploaded archive violates the versioned contract."""


def _validate_frame_file(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}
    ):
        raise ValueError("frame file must be a safe JPEG or PNG path")
    return value


def _validate_box(value: list[float] | None) -> list[float] | None:
    if value is None:
        return value
    if len(value) != 4 or not all(math.isfinite(item) for item in value):
        raise ValueError("box must contain four finite coordinates")
    x1, y1, x2, y2 = value
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive width and height")
    return value


class PackageFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordinal: int = Field(ge=1, le=FRAME_COUNT)
    source_frame: int = Field(ge=0)
    file: str = Field(min_length=1, max_length=160)
    target_box: list[float] | None

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str) -> str:
        return _validate_frame_file(value)

    @field_validator("target_box")
    @classmethod
    def validate_target_box(cls, value: list[float] | None) -> list[float] | None:
        return _validate_box(value)


class PackagePedestrianBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pedestrian_id: str = Field(min_length=1, max_length=64)
    box: list[float] | None

    @field_validator("box")
    @classmethod
    def validate_box(cls, value: list[float] | None) -> list[float] | None:
        return _validate_box(value)


class PackageFrameV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordinal: int = Field(ge=1, le=FRAME_COUNT)
    source_frame: int = Field(ge=0)
    file: str = Field(min_length=1, max_length=160)
    pedestrian_boxes: list[PackagePedestrianBox] = Field(
        min_length=1,
        max_length=MAX_PEDESTRIANS,
    )

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str) -> str:
        return _validate_frame_file(value)


class PackagePedestrian(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_index: int = Field(ge=0)
    pedestrian_id: str = Field(min_length=1, max_length=64)
    class_name: Literal["No Crossing", "Crossing", "Irrelevant"]
    raw_class: Literal[-1, 0, 1]
    binary_target: Literal[0, 1]
    color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")


class PackageVideo(BaseModel):
    """Identity and playback metadata for an external prepared-video file."""

    model_config = ConfigDict(extra="forbid")

    file_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}\.mp4$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_count: Literal[FRAME_COUNT]
    playback_fps: float = Field(gt=0.0, le=60.0)
    source_fps: float = Field(gt=0.0, le=120.0)
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)


class PackageManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    format_name: Literal[PACKAGE_FORMAT] = Field(alias="format")
    version: Literal[LEGACY_PACKAGE_VERSION]
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
    frames: list[PackageFrameV1] = Field(
        min_length=FRAME_COUNT,
        max_length=FRAME_COUNT,
    )


class PackageManifestV2(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    format_name: Literal[PACKAGE_FORMAT] = Field(alias="format")
    version: Literal[PACKAGE_VERSION]
    frame_count: Literal[FRAME_COUNT]
    tensors_file: Literal["tensors.npz"]
    display_name: str = Field(min_length=1, max_length=80)
    set_id: str = Field(pattern=r"^set\d{2}$")
    video_id: str = Field(pattern=r"^video_\d{4}$")
    video: PackageVideo | None = None
    pedestrians: list[PackagePedestrian] = Field(
        min_length=1,
        max_length=MAX_PEDESTRIANS,
    )
    frames: list[PackageFrameV2] = Field(
        min_length=FRAME_COUNT,
        max_length=FRAME_COUNT,
    )


PackageManifest = PackageManifestV1 | PackageManifestV2


@dataclass(frozen=True)
class LoadedPackage:
    metadata: dict
    tensors: list[SampleTensors]
    frames: dict[int, UploadedFrame]
    video: PackageVideo | None


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


def _validate_label(class_name: str, raw_class: int, binary_target: int) -> None:
    if class_name != CLASS_BY_RAW[raw_class]:
        raise PackageValidationError("class_name does not match raw_class.")
    if binary_target != int(raw_class == 1):
        raise PackageValidationError("binary_target does not match raw_class.")


def _manifest_pedestrians(manifest: PackageManifest) -> list[PackagePedestrian]:
    if isinstance(manifest, PackageManifestV1):
        return [
            PackagePedestrian(
                sample_index=manifest.sample_index,
                pedestrian_id=manifest.pedestrian_id,
                class_name=manifest.class_name,
                raw_class=manifest.raw_class,
                binary_target=manifest.binary_target,
                color=PEDESTRIAN_COLORS[0],
            )
        ]
    return manifest.pedestrians


def _load_manifest(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
) -> PackageManifest:
    try:
        info = members["manifest.json"]
    except KeyError as error:
        raise PackageValidationError("Package is missing manifest.json.") from error
    if info.file_size > 128 * 1024:
        raise PackageValidationError("manifest.json is too large.")
    try:
        payload = json.loads(archive.read(info))
        version = payload.get("version") if isinstance(payload, dict) else None
        if version == LEGACY_PACKAGE_VERSION:
            manifest: PackageManifest = PackageManifestV1.model_validate(payload)
        elif version == PACKAGE_VERSION:
            manifest = PackageManifestV2.model_validate(payload)
        else:
            raise PackageValidationError(
                f"Unsupported package version {version!r}; expected 1 or 2."
            )
    except PackageValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise PackageValidationError(f"Invalid manifest.json: {error}") from error

    ordered_frames = sorted(manifest.frames, key=lambda frame: frame.ordinal)
    ordinals = [frame.ordinal for frame in ordered_frames]
    frame_files = [frame.file for frame in ordered_frames]
    source_frames = [frame.source_frame for frame in ordered_frames]
    if ordinals != list(range(1, FRAME_COUNT + 1)):
        raise PackageValidationError("Frame ordinals must be exactly 1 through 15.")
    if len(set(frame_files)) != FRAME_COUNT:
        raise PackageValidationError("Every frame must reference a unique image file.")
    if any(right != left + 1 for left, right in pairwise(source_frames)):
        raise PackageValidationError("Source frames must be 15 consecutive frame IDs.")

    pedestrians = _manifest_pedestrians(manifest)
    pedestrian_ids = [person.pedestrian_id for person in pedestrians]
    if len(set(pedestrian_ids)) != len(pedestrian_ids):
        raise PackageValidationError("Pedestrian IDs must be unique within a package.")
    for pedestrian in pedestrians:
        _validate_label(
            pedestrian.class_name,
            pedestrian.raw_class,
            pedestrian.binary_target,
        )

    if isinstance(manifest, PackageManifestV2):
        expected_ids = set(pedestrian_ids)
        for frame in ordered_frames:
            frame_ids = [box.pedestrian_id for box in frame.pedestrian_boxes]
            if len(set(frame_ids)) != len(frame_ids) or set(frame_ids) != expected_ids:
                raise PackageValidationError(
                    "Every version-2 frame must contain exactly one box entry for "
                    "every manifest pedestrian."
                )

    expected_members = {"manifest.json", manifest.tensors_file, *frame_files}
    actual_members = set(members)
    if actual_members != expected_members:
        missing = sorted(expected_members - actual_members)
        extra = sorted(actual_members - expected_members)
        raise PackageValidationError(
            f"Package file set does not match manifest; missing={missing}, extra={extra}."
        )
    return manifest


def _load_tensors(
    blob: bytes,
    pedestrian_count: int,
    version: int,
) -> list[SampleTensors]:
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

    for name, base_shape in EXPECTED_SHAPES.items():
        expected_shape = base_shape if version == 1 else (pedestrian_count, *base_shape)
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

    if version == 1:
        return [SampleTensors(**arrays)]
    return [
        SampleTensors(**{name: array[index] for name, array in arrays.items()})
        for index in range(pedestrian_count)
    ]


def _image_type(path: str, content: bytes) -> str:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"} and content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if suffix == ".png" and content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    raise PackageValidationError(f"Frame {path} is not a valid declared image type.")


def _scene_raw_class(pedestrians: list[PackagePedestrian]) -> int:
    raw_classes = {person.raw_class for person in pedestrians}
    if 1 in raw_classes:
        return 1
    if 0 in raw_classes:
        return 0
    return -1


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
            package_pedestrians = _manifest_pedestrians(manifest)
            tensors = _load_tensors(
                archive.read(manifest.tensors_file),
                len(package_pedestrians),
                manifest.version,
            )
            colors = {
                person.pedestrian_id: (
                    person.color or PEDESTRIAN_COLORS[index % len(PEDESTRIAN_COLORS)]
                )
                for index, person in enumerate(package_pedestrians)
            }
            public_pedestrians = [
                {
                    "tensor_index": -1,
                    "sample_index": person.sample_index,
                    "pedestrian_id": person.pedestrian_id,
                    "class_name": person.class_name,
                    "raw_class": person.raw_class,
                    "binary_target": person.binary_target,
                    "color": colors[person.pedestrian_id],
                }
                for person in package_pedestrians
            ]

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
                if isinstance(frame, PackageFrameV1):
                    pedestrian_boxes = [
                        {
                            "pedestrian_id": package_pedestrians[0].pedestrian_id,
                            "box": frame.target_box,
                            "color": colors[package_pedestrians[0].pedestrian_id],
                        }
                    ]
                else:
                    pedestrian_boxes = [
                        {
                            "pedestrian_id": item.pedestrian_id,
                            "box": item.box,
                            "color": colors[item.pedestrian_id],
                        }
                        for item in frame.pedestrian_boxes
                    ]
                frame_metadata.append(
                    {
                        "ordinal": frame.ordinal,
                        "source_frame": frame.source_frame,
                        "target_box": pedestrian_boxes[0]["box"],
                        "pedestrian_boxes": pedestrian_boxes,
                    }
                )
    except PackageValidationError:
        raise
    except zipfile.BadZipFile as error:
        raise PackageValidationError("Uploaded file is not a valid ZIP archive.") from error

    primary = public_pedestrians[0]
    raw_class = _scene_raw_class(package_pedestrians)
    metadata = {
        "display_name": manifest.display_name,
        "sample_index": primary["sample_index"],
        "set_id": manifest.set_id,
        "video_id": manifest.video_id,
        "pedestrian_id": primary["pedestrian_id"],
        "class_name": CLASS_BY_RAW[raw_class],
        "raw_class": raw_class,
        "binary_target": int(raw_class == 1),
        "pedestrian_count": len(public_pedestrians),
        "pedestrians": public_pedestrians,
        "frames": frame_metadata,
    }
    video = manifest.video if isinstance(manifest, PackageManifestV2) else None
    return LoadedPackage(
        metadata=metadata,
        tensors=tensors,
        frames=frames,
        video=video,
    )
