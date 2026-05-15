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


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StageConditionedTransition(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        num_stage_transitions: int | None = None,
        stage_embed_dim: int = 16,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.stage_embed = None if num_stage_transitions is None else nn.Embedding(int(num_stage_transitions), int(stage_embed_dim))
        self.stage_embed_dim = int(stage_embed_dim) if self.stage_embed is not None else 0
        self.input_channels = self.latent_channels + self.stage_embed_dim

    def add_stage_condition(self, z: torch.Tensor, transition_ids: torch.Tensor | None) -> torch.Tensor:
        if self.stage_embed is None:
            return z
        if transition_ids is None:
            raise ValueError("transition_ids are required when stage conditioning is enabled")
        embedding = self.stage_embed(transition_ids)
        emb_map = embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, z.shape[-2], z.shape[-1])
        return torch.cat([z, emb_map], dim=1)


class SpatialLatentTransition(StageConditionedTransition):
    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int | None = None,
        num_blocks: int = 4,
        residual: bool = True,
        num_stage_transitions: int | None = None,
        stage_embed_dim: int = 16,
    ):
        super().__init__(latent_channels, num_stage_transitions=num_stage_transitions, stage_embed_dim=stage_embed_dim)
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        hidden_channels = int(hidden_channels or latent_channels)
        self.residual = bool(residual)
        self.in_proj = nn.Sequential(
            nn.Conv2d(self.input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(num_blocks)])
        self.out_proj = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, transition_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = self.add_stage_condition(z, transition_ids)
        delta = self.out_proj(self.blocks(self.in_proj(x)))
        if self.residual:
            return z + delta
        return delta


class LatentUNetTransition(StageConditionedTransition):
    def __init__(
        self,
        latent_channels: int,
        base_channels: int | None = None,
        num_bottleneck_blocks: int = 2,
        residual: bool = True,
        num_stage_transitions: int | None = None,
        stage_embed_dim: int = 16,
    ):
        super().__init__(latent_channels, num_stage_transitions=num_stage_transitions, stage_embed_dim=stage_embed_dim)
        if num_bottleneck_blocks <= 0:
            raise ValueError("num_bottleneck_blocks must be positive")

        base_channels = int(base_channels or latent_channels)
        self.residual = bool(residual)
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.stem = ConvBlock(self.input_channels, c1)
        self.down1 = DownsampleBlock(c1, c2)
        self.down2 = DownsampleBlock(c2, c3)
        self.bottleneck = nn.Sequential(*[ResidualBlock(c3) for _ in range(num_bottleneck_blocks)])
        self.up1 = UpsampleBlock(c3, c2)
        self.dec1 = ConvBlock(c2 + c2, c2)
        self.up2 = UpsampleBlock(c2, c1)
        self.dec2 = ConvBlock(c1 + c1, c1)
        self.out_proj = nn.Conv2d(c1, latent_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, transition_ids: torch.Tensor | None = None) -> torch.Tensor:
        if min(z.shape[-2:]) < 16:
            raise ValueError("LatentUNetTransition expects latent spatial resolution of at least 16x16")

        x = self.add_stage_condition(z, transition_ids)
        skip1 = self.stem(x)
        skip2 = self.down1(skip1)
        bottleneck = self.bottleneck(self.down2(skip2))
        up1 = self.up1(bottleneck)
        dec1 = self.dec1(torch.cat([up1, skip2], dim=1))
        up2 = self.up2(dec1)
        dec2 = self.dec2(torch.cat([up2, skip1], dim=1))
        delta = self.out_proj(dec2)
        if self.residual:
            return z + delta
        return delta
