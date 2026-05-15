from __future__ import annotations

import torch
from torch import nn

from models.stage_conditioning import StageConditioner


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _apply_scale_shift(x: torch.Tensor, condition: torch.Tensor | None, projector: nn.Linear | None) -> torch.Tensor:
    if condition is None or projector is None:
        return x
    scale, shift = projector(condition).chunk(2, dim=1)
    scale = scale.unsqueeze(-1).unsqueeze(-1)
    shift = shift.unsqueeze(-1).unsqueeze(-1)
    return x * (1.0 + scale) + shift


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.act2 = nn.SiLU(inplace=True)
        self.cond_proj1 = None if cond_dim is None else nn.Linear(cond_dim, channels * 2)
        self.cond_proj2 = None if cond_dim is None else nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = _apply_scale_shift(h, condition, self.cond_proj1)
        h = self.act1(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = _apply_scale_shift(h, condition, self.cond_proj2)
        return self.act2(x + h)


class SpatialLatentTransition(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int | None = None,
        num_blocks: int = 4,
        residual: bool = True,
        num_stages: int | None = None,
        stage_embed_dim: int = 32,
    ):
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        hidden_channels = int(hidden_channels or latent_channels)
        self.residual = bool(residual)
        self.stage_conditioner = None if num_stages is None else StageConditioner(int(stage_embed_dim), hidden_channels)
        cond_dim = hidden_channels if self.stage_conditioner is not None else None
        self.in_proj = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.in_proj_cond = None if cond_dim is None else nn.Linear(cond_dim, hidden_channels * 2)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_channels, cond_dim=cond_dim) for _ in range(num_blocks)])
        self.out_proj = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, stage_idx: torch.Tensor | None = None) -> torch.Tensor:
        condition = None
        if self.stage_conditioner is not None:
            if stage_idx is None:
                raise ValueError("stage_idx is required when stage conditioning is enabled")
            condition = self.stage_conditioner(stage_idx)

        hidden = self.in_proj(z)
        hidden = _apply_scale_shift(hidden, condition, self.in_proj_cond)
        for block in self.blocks:
            hidden = block(hidden, condition)

        pred = self.out_proj(hidden)
        if self.residual:
            return z + pred
        return pred
