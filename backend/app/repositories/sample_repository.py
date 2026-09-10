"""Validated access to the 15 prepared PIE graph sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


class SampleRepository:
    """Load metadata and tensors once, then expose immutable sample views."""

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
        if len(self._samples) != 15 or len(self._by_id) != len(self._samples):
            raise ValueError("The deployment manifest must contain 15 unique samples.")

        archive = np.load(tensors_path, allow_pickle=False)
        missing = set(self.REQUIRED_ARRAYS) - set(archive.files)
        if missing:
            raise ValueError(f"Prepared tensor archive is missing: {sorted(missing)}")
        self._arrays = {name: archive[name] for name in self.REQUIRED_ARRAYS}
        if any(array.shape[0] != len(self._samples) for array in self._arrays.values()):
            raise ValueError("Prepared arrays do not match the manifest sample count.")

    @property
    def selection(self) -> dict[str, Any]:
        return self._manifest["selection"]

    def list_samples(self) -> list[dict[str, Any]]:
        return [self._summary(sample) for sample in self._samples]

    def get_metadata(self, sample_id: str) -> dict[str, Any]:
        try:
            return dict(self._by_id[sample_id])
        except KeyError as error:
            raise KeyError(f"Unknown sample: {sample_id}") from error

    def get_tensors(self, sample_id: str) -> SampleTensors:
        metadata = self.get_metadata(sample_id)
        index = int(metadata["tensor_index"])
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
            "poster_url": sample["frames"][7]["url"],
        }
