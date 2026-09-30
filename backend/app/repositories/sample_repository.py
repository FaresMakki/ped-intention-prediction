"""Validated access to prepared PIE scenes and pedestrian tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import numpy as np


@dataclass(frozen=True)
class SampleTensors:
    trajectory_nodes: np.ndarray
    trajectory_adjacency: np.ndarray
    trajectory_mask: np.ndarray
    social_nodes: np.ndarray
    social_adjacency: np.ndarray
    social_mask: np.ndarray
    environment_nodes: np.ndarray
    environment_adjacency: np.ndarray
    environment_mask: np.ndarray
    skeleton_input: np.ndarray
    skeleton_keypoint_mask: np.ndarray
    skeleton_frame_mask: np.ndarray
    relation_risk_nodes: np.ndarray
    relation_risk_adjacency: np.ndarray
    relation_risk_mask: np.ndarray


@dataclass(frozen=True)
class UploadedFrame:
    content: bytes
    media_type: str


class SampleRepository:
    """Expose prepared samples plus validated, in-memory package uploads."""

    REQUIRED_ARRAYS = (
        "trajectory_nodes",
        "trajectory_adjacency",
        "trajectory_mask",
        "social_nodes",
        "social_adjacency",
        "social_mask",
        "environment_nodes",
        "environment_adjacency",
        "environment_mask",
        "skeleton_input",
        "skeleton_keypoint_mask",
        "skeleton_frame_mask",
        "relation_risk_nodes",
        "relation_risk_adjacency",
        "relation_risk_mask",
    )

    def __init__(self, manifest_path: Path, tensors_path: Path) -> None:
        self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._samples: list[dict[str, Any]] = self._manifest["samples"]
        self._by_id = {sample["id"]: sample for sample in self._samples}
        if not self._samples or len(self._by_id) != len(self._samples):
            raise ValueError("The deployment manifest must contain unique samples.")

        archive = np.load(tensors_path, allow_pickle=False)
        missing = set(self.REQUIRED_ARRAYS) - set(archive.files)
        if missing:
            raise ValueError(f"Prepared tensor archive is missing: {sorted(missing)}")
        self._arrays = {name: archive[name] for name in self.REQUIRED_ARRAYS}
        tensor_counts = {array.shape[0] for array in self._arrays.values()}
        if len(tensor_counts) != 1:
            raise ValueError("Prepared arrays do not share one tensor count.")
        tensor_count = tensor_counts.pop()
        for sample in self._samples:
            for pedestrian in self._pedestrians(sample):
                index = int(pedestrian["tensor_index"])
                if not 0 <= index < tensor_count:
                    raise ValueError(
                        f"Pedestrian tensor index {index} is outside the archive."
                    )
        self._uploaded_metadata: dict[str, dict[str, Any]] = {}
        self._uploaded_tensors: dict[str, list[SampleTensors]] = {}
        self._uploaded_frames: dict[str, dict[int, UploadedFrame]] = {}
        self._lock = RLock()

    @property
    def selection(self) -> dict[str, Any]:
        return self._manifest["selection"]

    def list_samples(self) -> list[dict[str, Any]]:
        with self._lock:
            uploaded = list(reversed(self._uploaded_metadata.values()))
        return [self._summary(sample) for sample in uploaded] + [
            self._summary(sample) for sample in self._samples
        ]

    def list_prepared_samples(self) -> list[dict[str, Any]]:
        return [self._summary(sample) for sample in self._samples]

    def get_metadata(self, sample_id: str) -> dict[str, Any]:
        with self._lock:
            uploaded = self._uploaded_metadata.get(sample_id)
        if uploaded is not None:
            return dict(uploaded)
        try:
            metadata = dict(self._by_id[sample_id])
            metadata["source"] = "prepared"
            return metadata
        except KeyError as error:
            raise KeyError(f"Unknown sample: {sample_id}") from error

    def get_summary(self, sample_id: str) -> dict[str, Any]:
        return self._summary(self.get_metadata(sample_id))

    def get_tensors(self, sample_id: str) -> SampleTensors:
        with self._lock:
            uploaded = self._uploaded_tensors.get(sample_id)
        if uploaded is not None:
            return uploaded[0]
        metadata = self.get_metadata(sample_id)
        return self._tensors_at(int(metadata["tensor_index"]))

    def get_scene_members(
        self, sample_id: str
    ) -> list[tuple[dict[str, Any], SampleTensors]]:
        """Return every annotated pedestrian and tensor bundle for one scene."""
        metadata = self.get_metadata(sample_id)
        if metadata["source"] == "upload":
            with self._lock:
                uploaded_tensors = self._uploaded_tensors[sample_id]
            pedestrians = self._pedestrians(metadata)
            if len(pedestrians) != len(uploaded_tensors):
                raise ValueError("Uploaded pedestrian metadata and tensors are misaligned.")
            return [
                (
                    {
                        **metadata,
                        **pedestrian,
                        "id": sample_id,
                        "source": "upload",
                    },
                    tensors,
                )
                for pedestrian, tensors in zip(
                    pedestrians,
                    uploaded_tensors,
                    strict=True,
                )
            ]

        members: list[tuple[dict[str, Any], SampleTensors]] = []
        for pedestrian in self._pedestrians(metadata):
            person_metadata = {
                **metadata,
                **pedestrian,
                "id": sample_id,
                "source": "prepared",
            }
            members.append(
                (person_metadata, self._tensors_at(int(pedestrian["tensor_index"])))
            )
        return members

    def _tensors_at(self, index: int) -> SampleTensors:
        return SampleTensors(
            trajectory_nodes=self._arrays["trajectory_nodes"][index],
            trajectory_adjacency=self._arrays["trajectory_adjacency"][index],
            trajectory_mask=self._arrays["trajectory_mask"][index],
            social_nodes=self._arrays["social_nodes"][index],
            social_adjacency=self._arrays["social_adjacency"][index],
            social_mask=self._arrays["social_mask"][index],
            environment_nodes=self._arrays["environment_nodes"][index],
            environment_adjacency=self._arrays["environment_adjacency"][index],
            environment_mask=self._arrays["environment_mask"][index],
            skeleton_input=self._arrays["skeleton_input"][index],
            skeleton_keypoint_mask=self._arrays["skeleton_keypoint_mask"][index],
            skeleton_frame_mask=self._arrays["skeleton_frame_mask"][index],
            relation_risk_nodes=self._arrays["relation_risk_nodes"][index],
            relation_risk_adjacency=self._arrays["relation_risk_adjacency"][index],
            relation_risk_mask=self._arrays["relation_risk_mask"][index],
        )

    def add_uploaded_sample(
        self,
        metadata: dict[str, Any],
        tensors: list[SampleTensors],
        frames: dict[int, UploadedFrame],
    ) -> dict[str, Any]:
        """Register one validated single- or multi-pedestrian package."""
        pedestrians = self._pedestrians(metadata)
        if not tensors or len(pedestrians) != len(tensors):
            raise ValueError("Every uploaded pedestrian must have one tensor bundle.")
        sample_id = f"upload-{uuid4().hex[:12]}"
        public_frames = [
            {
                **frame,
                "url": f"/api/uploads/{sample_id}/frames/{frame['ordinal']}",
            }
            for frame in metadata["frames"]
        ]
        public_metadata = {
            **metadata,
            "id": sample_id,
            "tensor_index": -1,
            "source": "upload",
            "pedestrian_count": len(pedestrians),
            "pedestrians": pedestrians,
            "frames": public_frames,
        }
        with self._lock:
            self._uploaded_metadata[sample_id] = public_metadata
            self._uploaded_tensors[sample_id] = list(tensors)
            self._uploaded_frames[sample_id] = frames
        return dict(public_metadata)

    def get_uploaded_frame(self, sample_id: str, ordinal: int) -> UploadedFrame:
        with self._lock:
            try:
                return self._uploaded_frames[sample_id][ordinal]
            except KeyError as error:
                raise KeyError(
                    f"Unknown uploaded frame: {sample_id}/{ordinal}"
                ) from error

    @staticmethod
    def _pedestrians(sample: dict[str, Any]) -> list[dict[str, Any]]:
        pedestrians = sample.get("pedestrians")
        if pedestrians:
            return list(pedestrians)
        return [
            {
                "tensor_index": sample["tensor_index"],
                "sample_index": sample["sample_index"],
                "pedestrian_id": sample["pedestrian_id"],
                "class_name": sample["class_name"],
                "raw_class": sample["raw_class"],
                "binary_target": sample["binary_target"],
                "color": "#B1EF68",
            }
        ]

    @staticmethod
    def _summary(sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": sample["id"],
            "display_name": sample["display_name"],
            "sample_index": sample["sample_index"],
            "set_id": sample["set_id"],
            "video_id": sample["video_id"],
            "pedestrian_id": sample["pedestrian_id"],
            "class_name": sample["class_name"],
            "raw_class": sample["raw_class"],
            "binary_target": sample["binary_target"],
            "pedestrian_count": len(SampleRepository._pedestrians(sample)),
            "poster_url": sample["frames"][7]["url"],
            "source": sample.get("source", "prepared"),
        }
