"""Selected uncertainty-gated Environment + Skeleton logit correction."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

META_FEATURE_COUNT = 13
FUSION_INPUT_COUNT = 19
HIDDEN_COUNT = 32
UNCERTAINTY_K = 5.0
CONTINUOUS_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12)
BINARY_INDICES = (8, 9)


def _fixed_mapping(probabilities: torch.Tensor) -> torch.Tensor:
    no_crossing = torch.clamp(probabilities[:, 0] + probabilities[:, 2], max=1.0)
    crossing = torch.clamp(probabilities[:, 1] - probabilities[:, 2], min=0.0)
    choose_crossing = crossing > no_crossing
    return torch.where(choose_crossing, crossing, 1.0 - no_crossing)


def environment_uncertainty_features(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    top_two = torch.topk(probabilities, k=2, dim=1, sorted=True).values
    total = probabilities.sum(dim=1, keepdim=True)
    normalized = probabilities / total.clamp_min(torch.finfo(probabilities.dtype).tiny)
    uniform = torch.full_like(probabilities, 1.0 / 3.0)
    normalized = torch.where(total > 0, normalized, uniform)
    entropy = -(
        normalized
        * torch.log(normalized.clamp_min(torch.finfo(probabilities.dtype).tiny))
    ).sum(dim=1)
    binary_probability = _fixed_mapping(probabilities)
    return torch.stack(
        (
            top_two[:, 0],
            top_two[:, 0] - top_two[:, 1],
            entropy,
            binary_probability,
            torch.abs(binary_probability - 0.5),
        ),
        dim=1,
    )


def skeleton_quality_features(
    skeleton: torch.Tensor,
    keypoint_mask: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    keypoint_mask = keypoint_mask.to(dtype=skeleton.dtype)
    frame_mask = frame_mask.to(dtype=skeleton.dtype)
    effective = keypoint_mask * frame_mask.unsqueeze(-1)
    valid_frames = frame_mask.sum(dim=1)
    valid_keypoints = effective.sum(dim=(1, 2))
    confidence = (skeleton[..., 2] * effective).sum(dim=(1, 2))
    confidence = confidence / valid_keypoints.clamp_min(1.0)
    return torch.stack(
        (
            valid_frames / 15.0,
            valid_keypoints / (15.0 * 19.0),
            confidence,
            (valid_frames == 0).to(dtype=skeleton.dtype),
        ),
        dim=1,
    )


def disagreement_features(
    environment_logits: torch.Tensor,
    skeleton_logits: torch.Tensor,
) -> torch.Tensor:
    environment_probabilities = torch.sigmoid(environment_logits)
    skeleton_probabilities = torch.sigmoid(skeleton_logits)
    class_disagreement = (
        environment_probabilities.argmax(dim=1) != skeleton_probabilities.argmax(dim=1)
    ).to(dtype=environment_logits.dtype)
    crossing_difference = torch.abs(
        environment_probabilities[:, 1] - skeleton_probabilities[:, 1]
    )
    binary_difference = torch.abs(
        _fixed_mapping(environment_probabilities)
        - _fixed_mapping(skeleton_probabilities)
    )
    signed_crossing_difference = (
        environment_probabilities[:, 1] - skeleton_probabilities[:, 1]
    )
    return torch.stack(
        (
            class_disagreement,
            crossing_difference,
            binary_difference,
            signed_crossing_difference,
        ),
        dim=1,
    )


def normalized_meta_features(
    environment_logits: torch.Tensor,
    skeleton_logits: torch.Tensor,
    skeleton: torch.Tensor,
    keypoint_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    normalization: Mapping[str, object],
) -> torch.Tensor:
    raw = torch.cat(
        (
            environment_uncertainty_features(environment_logits),
            skeleton_quality_features(skeleton, keypoint_mask, frame_mask),
            disagreement_features(environment_logits, skeleton_logits),
        ),
        dim=1,
    )
    mean = torch.as_tensor(normalization["mean"], dtype=raw.dtype, device=raw.device)
    std = torch.as_tensor(normalization["std"], dtype=raw.dtype, device=raw.device)
    normalized = (raw - mean.unsqueeze(0)) / std.unsqueeze(0)
    if not torch.equal(
        normalized[:, list(BINARY_INDICES)], raw[:, list(BINARY_INDICES)]
    ):
        raise RuntimeError("Binary fusion indicators changed during normalization.")
    return normalized


class UncertaintyGatedResidualCorrection(nn.Module):
    """Small validation-selected correction module; source branches stay frozen."""

    def __init__(self) -> None:
        super().__init__()
        self.variant = "uncertainty_gated_residual_correction"
        self.correction_network = nn.Sequential(
            nn.Linear(FUSION_INPUT_COUNT, HIDDEN_COUNT),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_COUNT, 3),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(FUSION_INPUT_COUNT, HIDDEN_COUNT),
            nn.ReLU(inplace=False),
            nn.Linear(HIDDEN_COUNT, 3),
            nn.Sigmoid(),
        )
        self.lambda_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(
        self,
        environment_logits: torch.Tensor,
        skeleton_logits: torch.Tensor,
        meta_features: torch.Tensor,
    ) -> torch.Tensor:
        environment = environment_logits.detach()
        skeleton = skeleton_logits.detach()
        meta = meta_features.detach()
        fusion_input = torch.cat((environment, skeleton, meta), dim=1)
        binary_probability = _fixed_mapping(torch.sigmoid(environment))
        distance = torch.abs(binary_probability - 0.5)
        uncertainty_weight = torch.exp(-UNCERTAINTY_K * distance).unsqueeze(1)
        correction = self.correction_network(fusion_input)
        gate = self.gate_network(fusion_input)
        return environment + self.lambda_scale * uncertainty_weight * gate * correction


def build_environment_skeleton_fusion() -> UncertaintyGatedResidualCorrection:
    return UncertaintyGatedResidualCorrection()
