from __future__ import annotations

from typing import List

import torch
from torch import nn


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


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


class ConvAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_channels: int = 64,
        num_downsamples: int = 3,
    ):
        super().__init__()
        if num_downsamples <= 0:
            raise ValueError("num_downsamples must be positive")

        widths: List[int] = [base_channels]
        for idx in range(1, num_downsamples):
            widths.append(base_channels * min(2**idx, 4))

        self.stem = ConvBlock(in_channels, widths[0])

        encoder_blocks = []
        prev_channels = widths[0]
        for width in widths:
            encoder_blocks.append(DownsampleBlock(prev_channels, width))
            prev_channels = width
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.to_latent = nn.Conv2d(prev_channels, latent_channels, kernel_size=3, padding=1)
        self.latent_refine = ConvBlock(latent_channels, latent_channels)

        self.from_latent = ConvBlock(latent_channels, prev_channels)
        decoder_blocks = []
        decode_widths = list(reversed(widths))
        current_channels = prev_channels
        for width in decode_widths:
            decoder_blocks.append(UpsampleBlock(current_channels, width))
            current_channels = width
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.out = nn.Sequential(
            ConvBlock(current_channels, current_channels),
            nn.Conv2d(current_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.encoder_blocks:
            h = block(h)
        z = self.to_latent(h)
        return self.latent_refine(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        for block in self.decoder_blocks:
            h = block(h)
        return self.out(h)

    def forward(self, x: torch.Tensor, latent_noise_std: float = 0.0) -> torch.Tensor:
        z = self.encode(x)
        if latent_noise_std > 0.0 and self.training:
            z = z + torch.randn_like(z) * float(latent_noise_std)
        return self.decode(z)
