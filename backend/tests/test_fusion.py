"""Unit tests for locked Rule B fusion."""

import torch

from app.ml.fusion import fixed_binary_mapping, fuse_logits


def test_crossing_logit_is_trajectory_only() -> None:
    trajectory = torch.tensor([[2.0, 3.0, 4.0]])
    social = torch.tensor([[10.0, -100.0, 20.0]])
    fused = fuse_logits(trajectory, social)
    assert torch.allclose(fused, torch.tensor([[4.0, 3.0, 7.2]], dtype=torch.float64))


def test_fixed_mapping_uses_strict_threshold() -> None:
    probabilities = torch.tensor([[0.2, 0.9, 0.1], [0.5, 0.5, 0.0]])
    scores, predictions = fixed_binary_mapping(probabilities)
    assert torch.allclose(scores, torch.tensor([0.8, 0.5]))
    assert predictions.tolist() == [True, False]
