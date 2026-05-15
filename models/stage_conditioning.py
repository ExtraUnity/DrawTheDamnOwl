from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(values: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("dim must be positive")
    half = dim // 2
    if half == 0:
        return values.to(dtype=torch.float32).unsqueeze(-1)

    values = values.to(dtype=torch.float32)
    exponent = -math.log(float(max_period)) * torch.arange(half, device=values.device, dtype=torch.float32) / max(half, 1)
    freqs = torch.exp(exponent)
    args = values.unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class StageConditioner(nn.Module):
    def __init__(self, embed_dim: int, cond_dim: int):
        super().__init__()
        if embed_dim <= 0 or cond_dim <= 0:
            raise ValueError("embed_dim and cond_dim must be positive")
        self.embed_dim = int(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, stage_idx: torch.Tensor) -> torch.Tensor:
        embedding = sinusoidal_embedding(stage_idx, self.embed_dim)
        return self.net(embedding)
