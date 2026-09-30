"""Run the additional research checkpoints for side-by-side comparison."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch

from app.ml.checkpoint_loader import load_checkpoint, sha256
from app.ml.environment_skeleton_fusion import (
    build_environment_skeleton_fusion,
    normalized_meta_features,
)
from app.ml.fusion import CLASS_NAMES, fixed_binary_mapping
from app.ml.graph_model import build_environment_model, build_relation_risk_model
from app.ml.skeleton_model import build_skeleton_model
from app.repositories.sample_repository import SampleRepository, SampleTensors
from app.services.comparison_catalog import METHOD_CATALOG, METHOD_ORDER
from app.services.inference_service import InferenceService


class ComparisonService:
    """Own the four comparison checkpoints and coordinate one comparison run."""

    def __init__(
        self,
        repository: SampleRepository,
        primary_service: InferenceService,
        environment_checkpoint: Path,
        skeleton_checkpoint: Path,
        fusion_checkpoint: Path,
        relation_risk_checkpoint: Path,
        device_name: str = "cpu",
    ) -> None:
        self.repository = repository
        self.primary_service = primary_service
        self.device = torch.device(device_name)
        self._lock = Lock()
        self.environment_model, environment_payload = load_checkpoint(
            environment_checkpoint,
            build_environment_model,
            self.device,
        )
        self.skeleton_model, skeleton_payload = load_checkpoint(
            skeleton_checkpoint,
            build_skeleton_model,
            self.device,
        )
        self.environment_skeleton_model, fusion_payload = load_checkpoint(
            fusion_checkpoint,
            build_environment_skeleton_fusion,
            self.device,
        )
        self.relation_risk_model, relation_risk_payload = load_checkpoint(
            relation_risk_checkpoint,
            build_relation_risk_model,
            self.device,
        )
        self.meta_normalization = fusion_payload["meta_normalization"]
        self.model_info = {
            "environment": self._checkpoint_info(
                environment_checkpoint, environment_payload
            ),
            "skeleton": self._checkpoint_info(skeleton_checkpoint, skeleton_payload),
            "environment_skeleton": self._checkpoint_info(
                fusion_checkpoint, fusion_payload
            ),
            "relation_risk": self._checkpoint_info(
                relation_risk_checkpoint, relation_risk_payload
            ),
        }

    @staticmethod
    def _checkpoint_info(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "file": path.name,
            "sha256": sha256(path),
            "selected_epoch": payload.get("selected_epoch"),
            "selection_metric": payload.get("selection_metric"),
        }

    def predict(self, sample_id: str) -> dict[str, Any]:
        """Compare every model for every pedestrian and aggregate scene risk."""
        started = time.perf_counter()
        members = self.repository.get_scene_members(sample_id)
        primary_predictions = self.primary_service.predict_members(members)
        with self._lock, torch.inference_mode():
            batched_logits = self._run_comparison_models_batch(
                [tensors for _metadata, tensors in members]
            )

        pedestrian_results = []
        for index, ((metadata, _tensors), primary) in enumerate(
            zip(members, primary_predictions, strict=True)
        ):
            logits = {
                method_id: values[index : index + 1]
                for method_id, values in batched_logits.items()
            }
            logits["trajectory_social"] = torch.tensor(
                [[primary["fusion"]["logits"][name] for name in CLASS_NAMES]],
                dtype=torch.float64,
            )
            methods = [
                self._method_result(method_id, logits[method_id], metadata)
                for method_id in METHOD_ORDER
            ]
            pedestrian_results.append(
                {
                    "pedestrian_id": metadata["pedestrian_id"],
                    "sample_index": metadata["sample_index"],
                    "ground_truth": primary["ground_truth"],
                    "methods": methods,
                    "agreement": self._agreement(methods),
                }
            )

        global_target = int(
            any(
                person["ground_truth"]["binary_target"]
                for person in pedestrian_results
            )
        )
        global_methods = [
            self._global_method_result(
                method_id,
                pedestrian_results,
                global_target,
            )
            for method_id in METHOD_ORDER
        ]
        return {
            "sample_id": sample_id,
            "pedestrian_count": len(pedestrian_results),
            "pedestrians": pedestrian_results,
            "global_methods": global_methods,
            "global_ground_truth": {
                "binary_target": global_target,
                "binary_label": "Crossing" if global_target else "Not Crossing",
            },
            "agreement": self._agreement(global_methods),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def _run_comparison_models_batch(
        self,
        tensors: list[SampleTensors],
    ) -> dict[str, torch.Tensor]:
        def batch(name: str) -> torch.Tensor:
            values = np.stack([getattr(item, name) for item in tensors]).astype(
                np.float32,
                copy=False,
            )
            return torch.from_numpy(values).to(self.device)

        environment_logits = self.environment_model(
            batch("environment_nodes"),
            batch("environment_adjacency"),
            batch("environment_mask"),
        )
        skeleton_input = batch("skeleton_input")
        keypoint_mask = batch("skeleton_keypoint_mask")
        frame_mask = batch("skeleton_frame_mask")
        skeleton_logits = self.skeleton_model(
            skeleton_input,
            keypoint_mask,
            frame_mask,
        )
        meta_features = normalized_meta_features(
            environment_logits,
            skeleton_logits,
            skeleton_input,
            keypoint_mask,
            frame_mask,
            self.meta_normalization,
        )
        environment_skeleton_logits = self.environment_skeleton_model(
            environment_logits,
            skeleton_logits,
            meta_features,
        )
        relation_risk_logits = self.relation_risk_model(
            batch("relation_risk_nodes"),
            batch("relation_risk_adjacency"),
            batch("relation_risk_mask"),
        )
        return {
            "environment": environment_logits.cpu(),
            "skeleton": skeleton_logits.cpu(),
            "environment_skeleton": environment_skeleton_logits.cpu(),
            "relation_risk": relation_risk_logits.cpu(),
        }

    def _method_result(
        self,
        method_id: str,
        logits: torch.Tensor,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        probabilities = torch.sigmoid(logits.to(torch.float64))
        binary_score, binary_prediction = fixed_binary_mapping(probabilities)
        prediction = int(binary_prediction.item())
        three_class_index = int(probabilities.argmax(dim=1).item())
        catalog = METHOD_CATALOG[method_id]
        return {
            "id": method_id,
            **catalog,
            "checkpoint": self._checkpoint_for(method_id),
            "logits": self._class_values(logits),
            "probabilities": self._class_values(probabilities),
            "three_class_prediction": CLASS_NAMES[three_class_index],
            "binary_score": float(binary_score.item()),
            "binary_prediction": prediction,
            "binary_label": "Crossing" if prediction else "Not Crossing",
            "threshold": 0.5,
            "is_correct": prediction == int(metadata["binary_target"]),
        }

    def _global_method_result(
        self,
        method_id: str,
        pedestrians: list[dict[str, Any]],
        global_target: int,
    ) -> dict[str, Any]:
        individual_methods = [
            next(method for method in person["methods"] if method["id"] == method_id)
            for person in pedestrians
        ]
        crossing_ids = [
            person["pedestrian_id"]
            for person, method in zip(pedestrians, individual_methods, strict=True)
            if method["binary_prediction"]
        ]
        not_crossing_ids = [
            person["pedestrian_id"]
            for person, method in zip(pedestrians, individual_methods, strict=True)
            if not method["binary_prediction"]
        ]
        prediction = int(bool(crossing_ids))
        return {
            "id": method_id,
            **METHOD_CATALOG[method_id],
            "checkpoint": self._checkpoint_for(method_id),
            "binary_score": max(method["binary_score"] for method in individual_methods),
            "binary_prediction": prediction,
            "binary_label": "Crossing" if prediction else "Not Crossing",
            "threshold": 0.5,
            "crossing_pedestrian_ids": crossing_ids,
            "not_crossing_pedestrian_ids": not_crossing_ids,
            "is_correct": prediction == global_target,
        }

    def _checkpoint_for(self, method_id: str) -> str:
        if method_id == "trajectory_social":
            return "trajectory_model.pt + social_model.pt"
        return str(self.model_info[method_id]["file"])

    @staticmethod
    def _agreement(methods: list[dict[str, Any]]) -> dict[str, Any]:
        predictions = [method["binary_prediction"] for method in methods]
        crossing_votes = sum(predictions)
        return {
            "unanimous": len(set(predictions)) == 1,
            "crossing_votes": crossing_votes,
            "not_crossing_votes": len(methods) - crossing_votes,
            "correct_methods": sum(method["is_correct"] for method in methods),
            "total_methods": len(methods),
        }

    @staticmethod
    def _class_values(values: torch.Tensor) -> dict[str, float]:
        row = values.squeeze(0).tolist()
        return {name: float(row[index]) for index, name in enumerate(CLASS_NAMES)}
