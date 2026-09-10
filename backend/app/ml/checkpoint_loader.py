"""Small, shared utilities for loading immutable inference artifacts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(
    path: Path,
    factory: Callable[[], nn.Module],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Unsupported checkpoint structure: {path.name}")
    model = factory().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    values = np.asarray(array, dtype=np.float32)
    return torch.from_numpy(values).unsqueeze(0).to(device)
