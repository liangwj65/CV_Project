# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from torch import nn


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def patchify_image(
    img: torch.Tensor,
    patch_size: tuple[int, int],
    stride: tuple[int, int]
) -> torch.Tensor:
    """Splits an input image into patches."""
    kh, kw = patch_size
    dh, dw = stride
    img = img.unfold(2, kh, dh).unfold(3, kw, dw)
    img = img.permute(0, 2, 3, 1, 4, 5)
    img = img.contiguous()
    img = img.view(img.size(0), -1, img.size(3), kh, kw)
    return img


class ExportableImageNormalization(nn.Module):
    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.mean: tuple[float, ...] = mean
        self.std: tuple[float, ...] = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        device = x.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.std, dtype=dtype, device=device)
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        return x.sub_(mean).div_(std)


def exportable_std(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard deviation operation exportable to ONNX."""
    m: torch.Tensor = x.mean(dim, keepdim=True)
    return torch.sqrt(torch.pow(x-m, 2).sum(dim) / (x.size(dim) - 1))

