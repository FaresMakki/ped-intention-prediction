"""Inference architecture for the selected PedGraphPlus skeleton checkpoint."""

from __future__ import annotations

import torch
from torch import nn

FRAME_COUNT = 15
JOINT_COUNT = 19
INPUT_CHANNELS = 4
CLASS_COUNT = 3
PARTITION_COUNT = 3

SKELETON_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (17, 5),
    (17, 6),
    (17, 0),
    (17, 18),
    (18, 11),
    (18, 12),
)


def build_base_adjacency() -> torch.Tensor:
    adjacency = torch.eye(JOINT_COUNT, dtype=torch.float32)
    for source, target in SKELETON_EDGES:
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    degree = adjacency.sum(dim=1)
    inverse_sqrt = degree.clamp_min(torch.finfo(adjacency.dtype).eps).rsqrt()
    normalized = inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    return normalized.unsqueeze(0)


class AdaptiveGraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.A_adaptive = nn.Parameter(
            torch.zeros(PARTITION_COUNT, JOINT_COUNT, JOINT_COUNT)
        )
        self.partition_projections = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            for _ in range(PARTITION_COUNT)
        )

    def effective_adjacency(self, base: torch.Tensor) -> torch.Tensor:
        combined = base.to(self.A_adaptive) + self.A_adaptive
        row_norm = torch.linalg.vector_norm(combined, ord=2, dim=-1, keepdim=True)
        return combined / row_norm.clamp_min(torch.finfo(combined.dtype).eps)

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        adjacency = self.effective_adjacency(base).to(dtype=x.dtype)
        outputs = [
            projection(torch.einsum("bctv,vw->bctw", x, adjacency[partition]))
            for partition, projection in enumerate(self.partition_projections)
        ]
        return torch.stack(outputs, dim=0).sum(dim=0)


class PedGraphPlusBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.graph_conv = AdaptiveGraphConvolution(in_channels, out_channels)
        self.temporal_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(5, 1),
            padding=(2, 0),
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(
        self,
        x: torch.Tensor,
        skeleton_mask: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.residual(x)
        output = self.graph_conv(x, base)
        output = self.temporal_conv(output)
        output = self.batch_norm(output)
        output = self.relu(output + residual)
        return output * skeleton_mask.to(dtype=output.dtype)


class SqueezeExcitation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reduce = nn.Linear(128, 32)
        self.relu = nn.ReLU(inplace=False)
        self.expand = nn.Linear(32, 128)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        weights = self.sigmoid(self.expand(self.relu(self.reduce(pooled))))
        return pooled * weights


class PedGraphPlusSkeletonGCN(nn.Module):
    """Selected two-block adaptive 19-joint GCN returning three raw logits."""

    def __init__(self, dropout: float = 0.75, num_partitions: int = 3) -> None:
        super().__init__()
        if num_partitions != PARTITION_COUNT:
            raise ValueError(f"Expected {PARTITION_COUNT} adjacency partitions.")
        self.register_buffer("A_base", build_base_adjacency())
        self.blocks = nn.ModuleList(
            (PedGraphPlusBlock(4, 64), PedGraphPlusBlock(64, 128))
        )
        self.se = SqueezeExcitation()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, CLASS_COUNT)

    @staticmethod
    def _prepare_inputs(
        skeleton: torch.Tensor,
        keypoint_mask: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (FRAME_COUNT, JOINT_COUNT, INPUT_CHANNELS)
        if skeleton.ndim != 4 or tuple(skeleton.shape[1:]) != expected:
            raise ValueError(f"Expected skeleton [B,15,19,4], got {skeleton.shape}.")
        batch = skeleton.shape[0]
        if keypoint_mask.shape != (batch, FRAME_COUNT, JOINT_COUNT):
            raise ValueError("Keypoint-mask shape does not match skeleton input.")
        if frame_mask.shape != (batch, FRAME_COUNT):
            raise ValueError("Frame-mask shape does not match skeleton input.")
        keypoint_mask = keypoint_mask.to(device=skeleton.device, dtype=skeleton.dtype)
        frame_mask = frame_mask.to(device=skeleton.device, dtype=skeleton.dtype)
        if not torch.equal(keypoint_mask, skeleton[..., 3]):
            raise ValueError("Explicit keypoint mask differs from skeleton input.")
        if torch.any(keypoint_mask > frame_mask.unsqueeze(-1)):
            raise ValueError("A pose-invalid frame contains a valid keypoint.")
        mask = (keypoint_mask * frame_mask.unsqueeze(-1)).unsqueeze(1)
        channels_first = skeleton.permute(0, 3, 1, 2).contiguous()
        return channels_first * mask, mask

    def forward(
        self,
        skeleton: torch.Tensor,
        keypoint_mask: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        output, mask = self._prepare_inputs(skeleton, keypoint_mask, frame_mask)
        for block in self.blocks:
            output = block(output, mask, self.A_base)
        pooled = (output * mask).sum(dim=(2, 3))
        pooled = pooled / mask.sum(dim=(2, 3)).clamp_min(1.0)
        return self.classifier(self.dropout(self.se(pooled)))


def build_skeleton_model() -> PedGraphPlusSkeletonGCN:
    return PedGraphPlusSkeletonGCN(dropout=0.75, num_partitions=3)
