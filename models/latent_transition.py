from __future__ import annotations

import torch
from torch import nn


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class SpatialLatentTransition(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int | None = None,
        num_blocks: int = 4,
        residual: bool = True,
        num_stages: int | None = None,
        stage_embed_dim: int = 16,
    ):
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        hidden_channels = int(hidden_channels or latent_channels)
        self.residual = bool(residual)
        self.stage_embed = None if num_stages is None else nn.Embedding(int(num_stages), int(stage_embed_dim))
        self.stage_proj = None
        if self.stage_embed is not None:
            self.stage_proj = nn.Sequential(
                nn.SiLU(inplace=True),
                nn.Linear(int(stage_embed_dim), hidden_channels),
            )
        self.in_proj = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(num_blocks)])
        self.out_proj = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, stage_idx: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.in_proj(z)
        if self.stage_embed is not None:
            if stage_idx is None:
                raise ValueError("stage_idx is required when stage conditioning is enabled")
            if self.stage_proj is None:
                raise RuntimeError("stage_proj was not initialized")
            stage_bias = self.stage_proj(self.stage_embed(stage_idx)).unsqueeze(-1).unsqueeze(-1)
            hidden = hidden + stage_bias
        pred = self.out_proj(self.blocks(hidden))
        if self.residual:
            return z + pred
        return pred
