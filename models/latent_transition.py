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
    ):
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        hidden_channels = int(hidden_channels or latent_channels)
        self.residual = bool(residual)
        self.in_proj = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(num_blocks)])
        self.out_proj = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pred = self.out_proj(self.blocks(self.in_proj(z)))
        if self.residual:
            return z + pred
        return pred
