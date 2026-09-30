"""Load both checkpoints and execute the complete locked-fusion pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from typing import Any

import torch

from app.ml.checkpoint_loader import load_checkpoint, sha256, to_tensor
from app.ml.fusion import (
    CLASS_NAMES,
    FusionWeights,
    fixed_binary_mapping,
    fuse_logits,
)
from app.ml.graph_model import build_social_model, build_trajectory_model
from app.repositories.sample_repository import SampleRepository, SampleTensors


class InferenceService:
    """Thread-safe orchestrator for two source networks and fixed fusion."""

    def __init__(
        self,
        repository: SampleRepository,
        trajectory_checkpoint: Path,
        social_checkpoint: Path,
        device_name: str = "cpu",
    ) -> None:
        self.repository = repository
        self.device = torch.device(device_name)
        self.weights = FusionWeights()
        self._lock = Lock()
        self.trajectory_model, trajectory_payload = load_checkpoint(
            trajectory_checkpoint,
            build_trajectory_model,
            self.device,
        )
        self.social_model, social_payload = load_checkpoint(
            social_checkpoint,
            build_social_model,
            self.device,
        )
        self.model_info = {
            "trajectory": {
                "file": trajectory_checkpoint.name,
                "sha256": sha256(trajectory_checkpoint),
                "selected_epoch": trajectory_payload.get("selected_epoch"),
            },
            "social": {
                "file": social_checkpoint.name,
                "sha256": sha256(social_checkpoint),
                "selected_epoch": social_payload.get("selected_epoch"),
            },
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
        """Run inference for prepared or validated uploaded tensors."""
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            trajectory_logits, social_logits = self._run_models(tensors)
            result = self._format_prediction(
                metadata,
                trajectory_logits,
                social_logits,
            )
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    def predict_all(self) -> dict[str, Any]:
        started = time.perf_counter()
        predictions = [
            self.predict(sample["id"])
            for sample in self.repository.list_prepared_samples()
        ]
        correct = sum(prediction["is_correct"] for prediction in predictions)
        return {
            "predictions": predictions,
            "correct": correct,
            "total": len(predictions),
            "accuracy": correct / len(predictions),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def _run_models(
        self,
        tensors: SampleTensors,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory_logits = self.trajectory_model(
            to_tensor(tensors.trajectory_nodes, self.device),
            to_tensor(tensors.trajectory_adjacency, self.device),
            to_tensor(tensors.trajectory_mask, self.device),
        )
        social_logits = self.social_model(
            to_tensor(tensors.social_nodes, self.device),
            to_tensor(tensors.social_adjacency, self.device),
            to_tensor(tensors.social_mask, self.device),
        )
        return trajectory_logits.cpu(), social_logits.cpu()

    def _format_prediction(
        self,
        metadata: dict[str, Any],
        trajectory_logits: torch.Tensor,
        social_logits: torch.Tensor,
    ) -> dict[str, Any]:
        fused_logits = fuse_logits(trajectory_logits, social_logits, self.weights)
        trajectory_probabilities = torch.sigmoid(trajectory_logits.to(torch.float64))
        social_probabilities = torch.sigmoid(social_logits.to(torch.float64))
        fused_probabilities = torch.sigmoid(fused_logits)
        binary_score, binary_prediction = fixed_binary_mapping(fused_probabilities)
        three_class_index = int(fused_probabilities.argmax(dim=1).item())
        predicted_binary = int(binary_prediction.item())
        binary_label = "Crossing" if predicted_binary else "Not Crossing"

        return {
            "sample_id": metadata["id"],
            "sample_index": metadata["sample_index"],
            "ground_truth": {
                "raw_class": metadata["raw_class"],
                "class_name": metadata["class_name"],
                "binary_target": metadata["binary_target"],
                "binary_label": (
                    "Crossing" if metadata["binary_target"] else "Not Crossing"
                ),
            },
            "trajectory": self._model_output(
                "Trajectory context",
                trajectory_logits,
                trajectory_probabilities,
                self.model_info["trajectory"]["file"],
            ),
            "social": self._model_output(
                "Social interaction",
                social_logits,
                social_probabilities,
                self.model_info["social"]["file"],
            ),
            "fusion": {
                "name": "Locked classwise fusion — Rule B",
                "weights": {
                    "no_crossing": self.weights.no_crossing,
                    "crossing": self.weights.crossing,
                    "irrelevant": self.weights.irrelevant,
                },
                "logits": self._class_values(fused_logits),
                "probabilities": self._class_values(fused_probabilities),
                "three_class_prediction": CLASS_NAMES[three_class_index],
                "binary_score": float(binary_score.item()),
                "binary_prediction": predicted_binary,
                "binary_label": binary_label,
                "threshold": 0.5,
            },
            "is_correct": predicted_binary == int(metadata["binary_target"]),
        }

    def _model_output(
        self,
        name: str,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
        checkpoint: str,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "checkpoint": checkpoint,
            "logits": self._class_values(logits),
            "probabilities": self._class_values(probabilities),
        }

    @staticmethod
    def _class_values(values: torch.Tensor) -> dict[str, float]:
        row = values.squeeze(0).tolist()
        return {name: float(row[index]) for index, name in enumerate(CLASS_NAMES)}
