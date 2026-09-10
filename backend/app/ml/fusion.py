"""Locked classwise fusion and its final binary mapping."""

from __future__ import annotations

from dataclasses import dataclass

import torch

CLASS_NAMES = ("No Crossing", "Crossing", "Irrelevant")


@dataclass(frozen=True)
class FusionWeights:
    """Validation-selected Rule B weights."""

    no_crossing: float = 0.75
    crossing: float = 1.0
    irrelevant: float = 0.80


DEFAULT_WEIGHTS = FusionWeights()


def fuse_logits(
    trajectory_logits: torch.Tensor,
    social_logits: torch.Tensor,
    weights: FusionWeights = DEFAULT_WEIGHTS,
) -> torch.Tensor:
    """Blend class logits while keeping Crossing trajectory-only."""
    if trajectory_logits.shape != social_logits.shape:
        raise ValueError("Trajectory and social logits must have matching shapes.")
    fused = trajectory_logits.to(torch.float64).clone()
    social = social_logits.to(torch.float64)
    fused[:, 0] = (
        weights.no_crossing * fused[:, 0] + (1.0 - weights.no_crossing) * social[:, 0]
    )
    fused[:, 2] = (
        weights.irrelevant * fused[:, 2] + (1.0 - weights.irrelevant) * social[:, 2]
    )
    return fused


def fixed_binary_mapping(
    probabilities: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map three independent sigmoid probabilities to Crossing decisions."""
    no_crossing, crossing, irrelevant = probabilities.unbind(dim=1)
    adjusted_no_crossing = torch.minimum(
        torch.ones_like(no_crossing),
        no_crossing + irrelevant,
    )
    adjusted_crossing = torch.maximum(
        torch.zeros_like(crossing),
        crossing - irrelevant,
    )
    score = torch.where(
        adjusted_crossing > adjusted_no_crossing,
        adjusted_crossing,
        1.0 - adjusted_no_crossing,
    )
    return score, score > threshold
