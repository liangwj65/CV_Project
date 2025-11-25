# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
工具函数模块
"""

import sys
from pathlib import Path
import logging
from typing import Optional
import torch
import numpy as np
from timm.utils import AverageMeter
from matplotlib import colormaps
from PIL import Image

from fusion import metrics


def create_logger(name="fusion"):
    """创建logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@torch.no_grad()
def validate(model, data_loader, criterion, device="cuda", logger=None, config=None):
    """
    验证函数
    
    Args:
        model: 模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备
        logger: 日志记录器
        config: 配置对象（可选）
    
    Returns:
        acc, ap, auc, loss_avg
    """
    model.eval()
    criterion.eval()

    loss_meter = AverageMeter()
    cls_metrics = metrics.Metrics(metrics=("auc", "ap", "accuracy"))

    feature_extraction_batch = None
    if config is not None and hasattr(config, 'MODEL') and hasattr(config.MODEL, 'FEATURE_EXTRACTION_BATCH'):
        feature_extraction_batch = config.MODEL.FEATURE_EXTRACTION_BATCH

    for idx, (images, target, dataset_idx) in enumerate(data_loader):
        if isinstance(images, list):
            images = [img.to(device, non_blocking=True) for img in images]
            images = [img.squeeze(dim=1) for img in images]
        else:
            images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 前向传播
        if isinstance(images, list):
            output = model(images, feature_extraction_batch)
        else:
            if images.size(dim=1) > 1:
                predictions = [model(images[:, i]) for i in range(images.size(dim=1))]
                predictions = torch.stack(predictions, dim=1)
                output = predictions.mean(dim=1)
            else:
                images = images.squeeze(dim=1)
                output = model(images)

        loss = criterion(output.squeeze(dim=1), target)
        output = torch.sigmoid(output)

        # 更新指标
        loss_meter.update(loss.item(), target.size(0))
        cls_metrics.update(output[:, 0].cpu(), target.cpu())

    metric_values = cls_metrics.compute()
    auc = metric_values["auc"].item()
    ap = metric_values["ap"].item()
    acc = metric_values["accuracy"].item()

    if logger:
        logger.info(f"Validation - Loss: {loss_meter.avg:.4f}, ACC: {acc:.3f}, AP: {ap:.3f}, AUC: {auc:.3f}")

    return acc, ap, auc, loss_meter.avg


def save_image_with_attention_overlay(
    image_patches: torch.Tensor,
    attention_scores: list[float],
    original_height: int,
    original_width: int,
    patch_size: int,
    stride: int,
    overlayed_image_path: Path,
    alpha: float = 0.4,
    palette: str = "coolwarm",
    mask_path: Optional[Path] = None,
    overlay_path: Optional[Path] = None
) -> None:
    """Overlays attention scores over image patches."""
    out_height_blocks: int = ((original_height - patch_size) // stride) + 1
    out_width_blocks: int = ((original_width - patch_size) // stride) + 1
    out_height: int = ((out_height_blocks - 1) * stride) + patch_size
    out_width: int = ((out_width_blocks - 1) * stride) + patch_size

    attention_scores: np.ndarray = np.array(attention_scores)
    # Normalize attention scores in [0, 1].
    # attention_scores = (attention_scores - attention_scores.min()) / attention_scores.max()

    cmap = colormaps[palette]
    # cmap_attention_scores: np.ndarray = cmap(np.array(attention_scores))[:, :3]

    out_image: np.ndarray = np.ones((out_height, out_width, 3))
    overlay: np.ndarray = np.zeros((out_height, out_width))
    overlay_count: np.ndarray = np.zeros_like(overlay)
    for i in range(out_height_blocks):
        for j in range(out_width_blocks):
            out_image[
                i*stride:i*stride+patch_size,
                j*stride:j*stride+patch_size,
                :
            ] = image_patches[0, (i*out_width_blocks)+j].detach().cpu().permute((1, 2, 0)).numpy()
            overlay[
                i*stride:i*stride+patch_size,
                j*stride:j*stride+patch_size
            ] += attention_scores[(i*out_width_blocks)+j]
            overlay_count[
                i * stride:i * stride + patch_size,
                j * stride:j * stride + patch_size
            ] += 1
    overlay = overlay / overlay_count
    overlay = (overlay - overlay.min()) / overlay.max()

    colormapped_overlay = cmap(overlay)[:, :, :3]

    overlayed_image: np.ndarray = (1 - alpha) * out_image + alpha * colormapped_overlay
    overlayed_image = (overlayed_image * 255).astype(np.uint8)

    Image.fromarray(overlayed_image).save(overlayed_image_path)
    if mask_path is not None:
        Image.fromarray((overlay*255).astype(np.uint8)).save(mask_path)
    if overlay_path is not None:
        Image.fromarray((colormapped_overlay*255).astype(np.uint8)).save(overlay_path)

