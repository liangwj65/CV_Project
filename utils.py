
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
import numpy as np
from timm.utils import AverageMeter
from matplotlib import colormaps
from PIL import Image

from fusion import metrics
from models.sid import AttentionMask


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
def validate(
    config,
    data_loader,
    model,
    criterion,
    logger: Optional[logging.Logger] = None,
    neptune_run=None,
    verbose: bool = True,
    return_predictions: bool = False
) -> Tuple[float, float, float, float, Optional[Dict[int, Tuple[float, Optional[AttentionMask]]]]]:
    """
    与SPAI保持一致的验证流程，可返回预测得分及注意力热力图路径。
    """
    del neptune_run  # 占位，保持接口一致

    model.eval()
    criterion.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_metrics: metrics.Metrics = metrics.Metrics(metrics=("auc", "ap", "accuracy"))

    predicted_scores: Dict[int, Tuple[float, Optional[AttentionMask]]] = {}
    feature_extraction_batch = getattr(config.MODEL, "FEATURE_EXTRACTION_BATCH", None)
    device = next(model.parameters()).device

    end = time.time()
    for idx, (images, target, dataset_idx) in enumerate(data_loader):
        if isinstance(images, list):
            images = [img.to(device, non_blocking=True) for img in images]
            images = [img.squeeze(dim=1) for img in images]
        else:
            images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 处理不同输入形态
        if isinstance(images, list) and getattr(config.TEST, "EXPORT_IMAGE_PATCHES", False):
            export_dirs = []
            base_export_dir = Path(config.OUTPUT) / "images"
            base_export_dir.mkdir(parents=True, exist_ok=True)
            for sample_idx in dataset_idx.detach().cpu().tolist():
                sample_dir = base_export_dir / f"{sample_idx}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                export_dirs.append(sample_dir)
            output, attention_masks = model(images, feature_extraction_batch, export_dirs)
        elif isinstance(images, list):
            output = model(images, feature_extraction_batch)
            attention_masks = [None] * len(images)
        else:
            if images.size(dim=1) > 1:
                predictions_batch = [model(images[:, i]) for i in range(images.size(dim=1))]
                predictions_batch = torch.stack(predictions_batch, dim=1)
                reduction = getattr(config.TEST, "VIEWS_REDUCTION_APPROACH", "mean")
                if reduction == "max":
                    output = predictions_batch.max(dim=1).values
                elif reduction == "mean":
                    output = predictions_batch.mean(dim=1)
                else:
                    raise TypeError(f"{reduction} is not a supported views reduction approach")
            else:
                images = images.squeeze(dim=1)
                output = model(images)
            attention_masks = [None] * output.size(0)

        loss = criterion(output.squeeze(dim=1), target)
        output = torch.sigmoid(output)

        loss_meter.update(loss.item(), target.size(0))
        cls_metrics.update(output[:, 0].detach().cpu(), target.detach().cpu())

        if return_predictions:
            batch_predictions = output.squeeze(dim=1).detach().cpu().tolist()
            batch_dataset_idx = dataset_idx.detach().cpu().tolist()
            for i, pred, mask in zip(batch_dataset_idx, batch_predictions, attention_masks):
                predicted_scores[i] = (pred, mask)

        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and idx % max(1, config.PRINT_FREQ) == 0:
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            else:
                memory_used = 0.0
            if logger:
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}] | '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | '
                    f'Mem {memory_used:.0f}MB'
                )

    metric_values = cls_metrics.compute()
    auc = metric_values["auc"].item()
    ap = metric_values["ap"].item()
    acc = metric_values["accuracy"].item()

    if return_predictions:
        return acc, ap, auc, loss_meter.avg, predicted_scores
    return acc, ap, auc, loss_meter.avg, None


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

