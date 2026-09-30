"""Run the additional research checkpoints for side-by-side comparison."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from typing import Any

import torch

from app.ml.checkpoint_loader import load_checkpoint, sha256, to_tensor
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
        metadata = self.repository.get_metadata(sample_id)
        tensors = self.repository.get_tensors(sample_id)
        return self.predict_tensors(metadata, tensors)

    def predict_tensors(
        self,
        metadata: dict[str, Any],
        tensors: SampleTensors,
    ) -> dict[str, Any]:
        """Run every comparison checkpoint on one validated tensor bundle."""
        started = time.perf_counter()
        primary = self.primary_service.predict_tensors(metadata, tensors)
        with self._lock, torch.inference_mode():
            logits = self._run_comparison_models(tensors)
        final_logits = torch.tensor(
            [[primary["fusion"]["logits"][name] for name in CLASS_NAMES]],
            dtype=torch.float64,
        )
        logits["trajectory_social"] = final_logits
        methods = [
            self._method_result(method_id, logits[method_id], metadata)
            for method_id in METHOD_ORDER
        ]
        predictions = [method["binary_prediction"] for method in methods]
        correct_count = sum(method["is_correct"] for method in methods)
        crossing_votes = sum(predictions)
        return {
            "sample_id": metadata["id"],
            "sample_index": metadata["sample_index"],
            "ground_truth": primary["ground_truth"],
            "methods": methods,
            "agreement": {
                "unanimous": len(set(predictions)) == 1,
                "crossing_votes": crossing_votes,
                "not_crossing_votes": len(methods) - crossing_votes,
                "correct_methods": correct_count,
                "total_methods": len(methods),
            },
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def _run_comparison_models(
        self,
        tensors: SampleTensors,
    ) -> dict[str, torch.Tensor]:
        environment_logits = self.environment_model(
            to_tensor(tensors.environment_nodes, self.device),
            to_tensor(tensors.environment_adjacency, self.device),
            to_tensor(tensors.environment_mask, self.device),
        )
        skeleton_input = to_tensor(tensors.skeleton_input, self.device)
        keypoint_mask = to_tensor(tensors.skeleton_keypoint_mask, self.device)
        frame_mask = to_tensor(tensors.skeleton_frame_mask, self.device)
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
            to_tensor(tensors.relation_risk_nodes, self.device),
            to_tensor(tensors.relation_risk_adjacency, self.device),
            to_tensor(tensors.relation_risk_mask, self.device),
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
        checkpoint = (
            "trajectory_model.pt + social_model.pt"
            if method_id == "trajectory_social"
            else self.model_info[method_id]["file"]
        )
        return {
            "id": method_id,
            **catalog,
            "checkpoint": checkpoint,
            "logits": self._class_values(logits),
            "probabilities": self._class_values(probabilities),
            "three_class_prediction": CLASS_NAMES[three_class_index],
            "binary_score": float(binary_score.item()),
            "binary_prediction": prediction,
            "binary_label": "Crossing" if prediction else "Not Crossing",
            "threshold": 0.5,
            "is_correct": prediction == int(metadata["binary_target"]),
        }

    @staticmethod
    def _class_values(values: torch.Tensor) -> dict[str, float]:
        row = values.squeeze(0).tolist()
        return {name: float(row[index]) for index, name in enumerate(CLASS_NAMES)}
