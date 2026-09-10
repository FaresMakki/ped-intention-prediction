"""Inference-only graph architecture shared by both source checkpoints.

The layer names and tensor operations intentionally match the research models,
so the original state dictionaries load strictly and produce the same logits.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

FRAME_COUNT = 15
CLASS_COUNT = 3
BLOCK_CHANNELS = (64, 64, 128)
RESIDUAL_PATTERN = (False, True, False)


class DynamicGraphConvolution(nn.Module):
    """Aggregate each frame with symmetrically normalized adjacency."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_channels, out_channels)

    @staticmethod
    def normalize_adjacency(
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = node_mask.to(dtype=adjacency.dtype)
        valid_pairs = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        masked = adjacency * valid_pairs
        degree = masked.sum(dim=-1)
        inverse = torch.where(
            degree > 0,
            degree.clamp_min(torch.finfo(adjacency.dtype).eps).rsqrt(),
            torch.zeros_like(degree),
        )
        return inverse.unsqueeze(-1) * masked * inverse.unsqueeze(-2)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.normalize_adjacency(adjacency, node_mask)
        aggregated = torch.einsum("btvw,bctw->bctv", normalized, x)
        projected = self.projection(aggregated.permute(0, 2, 3, 1))
        return projected.permute(0, 3, 1, 2).contiguous()


class TemporalAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=(1, 0))

    def forward(
        self,
        x: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = node_mask.to(dtype=x.dtype).unsqueeze(1)
        denominator = (mask.sum(dim=3, keepdim=True) * x.shape[1]).clamp_min(1.0)
        descriptor = (x * mask).sum(dim=(1, 3), keepdim=True) / denominator
        weights = torch.sigmoid(self.conv(descriptor))
        return x * weights, weights


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(channels // 4, 1)
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = node_mask.to(dtype=x.dtype).unsqueeze(1)
        valid_count = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        descriptor = (x * mask).sum(dim=(2, 3), keepdim=True) / valid_count
        weights = torch.sigmoid(self.expand(self.relu(self.reduce(descriptor))))
        return x * weights, weights


class NodeAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = node_mask.to(dtype=x.dtype).unsqueeze(1)
        weights = torch.sigmoid(self.conv(x)) * mask
        return x * weights, weights


class GraphAttentionBlock(nn.Module):
    """Graph convolution followed by temporal, channel, and node attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        use_residual: bool,
    ) -> None:
        super().__init__()
        self.graph_conv = DynamicGraphConvolution(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.temporal_attention = TemporalAttention()
        self.channel_attention = ChannelAttention(out_channels)
        self.node_attention = NodeAttention(out_channels)
        self.temporal_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
        )
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = x if self.use_residual else None
        x = self.relu(self.batch_norm(self.graph_conv(x, adjacency, node_mask)))
        x, _ = self.temporal_attention(x, node_mask)
        x, _ = self.channel_attention(x, node_mask)
        x, _ = self.node_attention(x, node_mask)
        x = self.dropout(self.temporal_conv(x))
        if residual is not None:
            x = x + residual
        return self.relu(x) * node_mask.to(dtype=x.dtype).unsqueeze(1)


class ThreeClassGraphModel(nn.Module):
    """Configurable inference form of the trajectory and social networks."""

    def __init__(
        self,
        input_features: int,
        node_count: int,
        block_channels: Sequence[int] = BLOCK_CHANNELS,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.node_count = node_count
        channels = (input_features, *block_channels)
        self.blocks = nn.ModuleList(
            GraphAttentionBlock(
                channels[index],
                channels[index + 1],
                dropout,
                RESIDUAL_PATTERN[index],
            )
            for index in range(len(block_channels))
        )
        self.classifier = nn.Sequential(
            nn.Linear(BLOCK_CHANNELS[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, CLASS_COUNT),
        )

    def _validate(
        self,
        nodes: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> None:
        expected_nodes = (FRAME_COUNT, self.node_count, self.input_features)
        if nodes.ndim != 4 or tuple(nodes.shape[1:]) != expected_nodes:
            raise ValueError(
                f"Expected nodes [B,{FRAME_COUNT},{self.node_count},"
                f"{self.input_features}], found {tuple(nodes.shape)}."
            )
        batch_size = nodes.shape[0]
        if adjacency.shape != (
            batch_size,
            FRAME_COUNT,
            self.node_count,
            self.node_count,
        ):
            raise ValueError("Adjacency shape does not match the node tensor.")
        if node_mask.shape != (batch_size, FRAME_COUNT, self.node_count):
            raise ValueError("Node-mask shape does not match the node tensor.")

    def forward(
        self,
        nodes: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate(nodes, adjacency, node_mask)
        x = nodes.permute(0, 3, 1, 2).contiguous()
        mask = node_mask.to(dtype=x.dtype)
        adjacency = adjacency.to(dtype=x.dtype)
        x = x * mask.unsqueeze(1)
        for block in self.blocks:
            x = block(x, adjacency, mask)
        pooled = (x * mask.unsqueeze(1)).sum(dim=(2, 3))
        pooled = pooled / mask.sum(dim=(1, 2)).clamp_min(1.0).unsqueeze(1)
        return self.classifier(pooled)


def build_trajectory_model() -> ThreeClassGraphModel:
    return ThreeClassGraphModel(input_features=64, node_count=8)


def build_social_model() -> ThreeClassGraphModel:
    return ThreeClassGraphModel(input_features=80, node_count=12)


def build_environment_model() -> ThreeClassGraphModel:
    """Build the protected 12-node environment graph architecture."""

    return ThreeClassGraphModel(input_features=37, node_count=12)


def build_relation_risk_model() -> ThreeClassGraphModel:
    """Build the protected-style 8-node Relation-Risk architecture."""

    return ThreeClassGraphModel(input_features=64, node_count=8)
