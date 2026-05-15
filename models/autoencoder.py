from __future__ import annotations

from typing import List

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.cond_proj1 = None if cond_dim is None else nn.Linear(cond_dim, out_channels * 2)
        self.cond_proj2 = None if cond_dim is None else nn.Linear(cond_dim, out_channels * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = _apply_scale_shift(h, condition, self.cond_proj1)
        h = self.act1(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = _apply_scale_shift(h, condition, self.cond_proj2)
        h = self.act2(h)
        return h


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.act2(self.norm2(self.conv2(h)))
        return h


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int | None = None):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.cond_proj1 = None if cond_dim is None else nn.Linear(cond_dim, out_channels * 2)
        self.cond_proj2 = None if cond_dim is None else nn.Linear(cond_dim, out_channels * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        h = self.deconv(x)
        h = self.norm1(h)
        h = _apply_scale_shift(h, condition, self.cond_proj1)
        h = self.act1(h)
        h = self.conv(h)
        h = self.norm2(h)
        h = _apply_scale_shift(h, condition, self.cond_proj2)
        h = self.act2(h)
        return h


class OutputBlock(nn.Module):
    def __init__(self, channels: int, out_channels: int, cond_dim: int | None = None):
        super().__init__()
        self.block = ConvBlock(channels, channels, cond_dim=cond_dim)
        self.proj = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        return self.activation(self.proj(self.block(x, condition)))


class ConvAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_channels: int = 64,
        num_downsamples: int = 3,
        num_stages: int | None = None,
        stage_embed_dim: int = 32,
        condition_decoder: bool = False,
    ):
        super().__init__()
        if num_downsamples <= 0:
            raise ValueError("num_downsamples must be positive")

        widths: List[int] = [base_channels]
        for idx in range(1, num_downsamples):
            widths.append(base_channels * min(2**idx, 4))

        self.condition_decoder = bool(condition_decoder and num_stages is not None)
        cond_dim = latent_channels if self.condition_decoder else None
        self.stage_conditioner = None if not self.condition_decoder else StageConditioner(int(stage_embed_dim), latent_channels)

        self.stem = ConvBlock(in_channels, widths[0])

        encoder_blocks = []
        prev_channels = widths[0]
        for width in widths:
            encoder_blocks.append(DownsampleBlock(prev_channels, width))
            prev_channels = width
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.to_latent = nn.Conv2d(prev_channels, latent_channels, kernel_size=3, padding=1)
        self.latent_refine = ConvBlock(latent_channels, latent_channels)

        self.from_latent = ConvBlock(latent_channels, prev_channels, cond_dim=cond_dim)
        decoder_blocks = []
        decode_widths = list(reversed(widths))
        current_channels = prev_channels
        for width in decode_widths:
            decoder_blocks.append(UpsampleBlock(current_channels, width, cond_dim=cond_dim))
            current_channels = width
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.out = OutputBlock(current_channels, in_channels, cond_dim=cond_dim)

    def _decoder_condition(self, stage_idx: torch.Tensor | None) -> torch.Tensor | None:
        if not self.condition_decoder:
            return None
        if self.stage_conditioner is None:
            raise RuntimeError("stage_conditioner was not initialized")
        if stage_idx is None:
            raise ValueError("stage_idx is required when decoder stage conditioning is enabled")
        return self.stage_conditioner(stage_idx)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.encoder_blocks:
            h = block(h)
        z = self.to_latent(h)
        return self.latent_refine(z)

    def decode(self, z: torch.Tensor, stage_idx: torch.Tensor | None = None) -> torch.Tensor:
        condition = self._decoder_condition(stage_idx)
        h = self.from_latent(z, condition)
        for block in self.decoder_blocks:
            h = block(h, condition)
        return self.out(h, condition)

    def forward(self, x: torch.Tensor, latent_noise_std: float = 0.0, stage_idx: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encode(x)
        if latent_noise_std > 0.0 and self.training:
            z = z + torch.randn_like(z) * float(latent_noise_std)
        return self.decode(z, stage_idx=stage_idx)
