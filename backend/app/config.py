"""Application paths and runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Resolved deployment settings with safe local defaults."""

    deployment_root: Path = Path(__file__).resolve().parents[2]
    device: str = os.getenv("PIE_DEVICE", "cpu")

    @property
    def data_dir(self) -> Path:
        return self.deployment_root / "data"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "samples.json"

    @property
    def tensors_path(self) -> Path:
        return self.data_dir / "prepared_samples.npz"

    @property
    def trajectory_checkpoint(self) -> Path:
        return self.deployment_root / "models" / "trajectory_model.pt"

    @property
    def social_checkpoint(self) -> Path:
        return self.deployment_root / "models" / "social_model.pt"

    @property
    def environment_checkpoint(self) -> Path:
        return self.deployment_root / "models" / "environment_model.pt"

    @property
    def skeleton_checkpoint(self) -> Path:
        return self.deployment_root / "models" / "skeleton_model.pt"

    @property
    def environment_skeleton_checkpoint(self) -> Path:
        return self.deployment_root / "models" / "environment_skeleton_fusion_model.pt"

    @property
    def relation_risk_checkpoint(self) -> Path:
        return self.deployment_root / "models" / "relation_risk_model.pt"

    @property
    def frontend_dist(self) -> Path:
        return self.deployment_root / "frontend" / "dist"


settings = Settings()
