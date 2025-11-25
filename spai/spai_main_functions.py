# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pathlib
import time
import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import neptune
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import yacs
import filetype
from torch import nn
from torch.nn import TripletMarginLoss
from torch.utils.tensorboard import SummaryWriter
from timm.utils import AverageMeter
from yacs.config import CfgNode

import spai.data.data_finetune
from spai.config import get_config
from spai.models import build_cls_model
from spai.data import build_loader, build_loader_test
from spai.lr_scheduler import build_scheduler
from spai.models.sid import AttentionMask
from spai.onnx import compare_pytorch_onnx_models
from spai.optimizer import build_optimizer
from spai.logger import create_logger
from spai.utils import (
    load_pretrained,
    save_checkpoint,
    get_grad_norm,
    find_pretrained_checkpoints,
    inf_nan_to_num
)
from spai.models import losses
from spai import metrics
from spai import data_utils

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

cv2.setNumThreads(1)
logger: Optional[logging.Logger] = None


def infer(
    cfg: Path = Path("./configs/spai.yaml"),
    batch_size: int = 1,
    input_paths: list[Path] | None = None,
    input_csv_root_dir: list[Path] | None = None,
    split: str = "test",
    lmdb_path: Optional[Path] = None,
    model: Path = Path("./weights/spai.pth"),
    output: Path = Path("./output"),
    tag: str = "spai",
    resize_to: Optional[int] = None,
    extra_options: tuple[str, str] = (),
) -> None:
    if input_paths is None:
        input_paths = []
    if input_csv_root_dir is None:
        input_csv_root_dir = []
    
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "test_csv": [str(p) for p in input_paths],
        "test_csv_root": [str(p) for p in input_csv_root_dir],
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "resize_to": resize_to,
        "opts": extra_options
    })

    output.mkdir(exist_ok=True, parents=True)

    # Create a console logger.
    global logger
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create dataloaders.
    test_datasets_names, test_datasets, test_loaders = build_loader_test(
        config, logger, split=split, dummy_csv_dir=output
    )

    # Load the trained weights' checkpoint.
    model_ckpt: pathlib.Path = find_pretrained_checkpoints(config)[0]
    criterion = losses.build_loss(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_cls_model(config)
    model.cuda()
    load_pretrained(config, model, logger,  checkpoint_path=model_ckpt, verbose=False)

    # Infer predictions and compute performance metrics (only on csv inputs with ground-truths).
    for test_data_loader, test_dataset, test_data_name, input_path in zip(test_loaders,
                                                                          test_datasets,
                                                                          test_datasets_names,
                                                                          input_paths):
        predictions: Optional[dict[int, tuple[float, Optional[AttentionMask]]]]
        acc, ap, auc, loss, predictions = validate(
            config, test_data_loader, model, criterion, logger, None, return_predictions=True
        )

        if input_path.is_file():  # When input path is a dir, no ground-truth exists.
            logger.info(f"Test | {test_data_name} | Images: {len(test_dataset)} | loss: {loss:.4f}")
            logger.info(f"Test | {test_data_name} | Images: {len(test_dataset)} | ACC: {acc:.3f}")
            if test_dataset.get_classes_num() > 1:  # AUC and AP make no sense with only 1 class.
                logger.info(
                    f"Test | {test_data_name} | Images: {len(test_dataset)} | AP: {ap:.3f}")
                logger.info(
                    f"Test | {test_data_name} | Images: {len(test_dataset)} | AUC: {auc:.3f}")

        # Update the output CSV.
        if predictions is not None:
            column_name: str = f"{tag}"
            scores: dict[int, float] = {i: t[0] for i, t in predictions.items()}
            attention_masks: dict[int, pathlib.Path] = {
                i: t[1].mask for i, t in predictions.items() if t[1] is not None
            }
            test_dataset.update_dataset_csv(
                column_name, scores, export_dir=Path(output)
            )
            if len(attention_masks) == len(scores):
                test_dataset.update_dataset_csv(
                    f"{column_name}_mask", attention_masks, export_dir=Path(output)
                )


def tsne(
    cfg: Path = Path("./configs/spai.yaml"),
    test_csv: list[Path] | None = None,
    test_csv_root_dir: list[Path] | None = None,
    lmdb_path: Optional[Path] = None,
    model: Path = Path("./weights/spai.pth"),
    output: Path = Path("./output"),
    tag: str = "spai",
    resize_to: Optional[int] = None,
    extra_options: tuple[str, str] = (),
) -> None:
    if test_csv is None:
        test_csv = []
    if test_csv_root_dir is None:
        test_csv_root_dir = []
    
    config = get_config({
        "cfg": str(cfg),
        "batch_size": 1,  # Currently, required to be 1 for correctly distinguishing embeddings.
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "resize_to": resize_to,
        "opts": extra_options
    })
    from spai import tsne as tsne_utils

    pathlib.Path(config.OUTPUT).mkdir(exist_ok=True, parents=True)
    global logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    log_writer = SummaryWriter(log_dir=config.OUTPUT)
    # print config
    logger.info(config.dump())

    neptune_tags: list[str] = ["mfm", "tsne"]
    neptune_tags.extend([p.stem for p in test_csv])
    neptune_run = neptune.init_run(
        name=config.TAG,
        tags=neptune_tags
    )

    test_datasets_names, test_datasets, test_loaders = build_loader_test(config, logger)
    model_ckpt: pathlib.Path = find_pretrained_checkpoints(config)[0]

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_cls_model(config)
    model.cuda()
    logger.info(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of Params: {n_parameters}")
    if hasattr(model, "flops"):
        flops = model.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")

    checkpoint_epoch: int = load_pretrained(config, model, logger, checkpoint_path=model_ckpt)

    # Test the model.
    for test_data_loader, test_dataset, test_data_name in zip(test_loaders,
                                                              test_datasets,
                                                              test_datasets_names):
        tsne_utils.visualize_tsne(config, test_data_loader, test_data_name, model, neptune_run)

        if log_writer is not None:
            log_writer.flush()
        if neptune_run is not None:
            neptune_run.sync()


def export_onnx(
    cfg: Path = Path("./configs/spai.yaml"),
    model: Path = Path("./weights/spai.pth"),
    output: Path = Path("./output"),
    tag: str = "spai",
    extra_options: tuple[str, str] = (),
    exclude_preprocessing: bool = False,
) -> None:
    config = get_config({
        "cfg": str(cfg),
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "opts": extra_options
    })

    output: Path = Path(config.OUTPUT)
    output.mkdir(exist_ok=True, parents=True)

    global logger
    logger = create_logger(output_dir=output, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    config_export_path: Path = output / "config.json"
    with config_export_path.open("w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {config_export_path}")
    logger.info(config.dump())

    model_checkpoints: list[pathlib.Path] = find_pretrained_checkpoints(config)
    onnx_export_dir: Path = output / "onnx"
    onnx_export_dir.mkdir(exist_ok=True, parents=True)

    for i, model_ckpt in enumerate(model_checkpoints):
        if i == 0:
            logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_cls_model(config)
        checkpoint_epoch: int = load_pretrained(config, model, logger,
                                                checkpoint_path=model_ckpt, verbose=i == 0)

        model.to("cpu")
        model.eval()

        patch_encoder: Path = onnx_export_dir / "patch_encoder.onnx"
        patch_aggregator: Path = onnx_export_dir / "patch_aggregator.onnx"
        model.export_onnx(patch_encoder, patch_aggregator,
                          include_fft_preprocessing=not exclude_preprocessing)


def validate_onnx(
    cfg: Path = Path("./configs/spai.yaml"),
    batch_size: Optional[int] = None,
    test_csv: list[Path] | None = None,
    test_csv_root_dir: list[Path] | None = None,
    split: str = "test",
    lmdb_path: Optional[Path] = None,
    model: Path = Path("./weights/spai.pth"),
    output: Path = Path("./output"),
    tag: str = "spai",
    device: str = "cpu",
    extra_options: tuple[str, str] = (),
    exclude_preprocessing: bool = False,
) -> None:
    if test_csv is None:
        test_csv = []
    if test_csv_root_dir is None:
        test_csv_root_dir = []
    
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "output": str(output),
        "tag": tag,
        "pretrained": str(model),
        "opts": extra_options
    })

    output: Path = Path(config.OUTPUT)
    output.mkdir(exist_ok=True, parents=True)

    global logger
    logger = create_logger(output_dir=output, dist_rank=0, name=f"{config.MODEL.NAME}")

    # Export current config.
    config_export_path: Path = output / "config.json"
    with config_export_path.open("w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {config_export_path}")
    logger.info(config.dump())

    test_datasets_names, test_datasets, test_loaders = build_loader_test(
        config, logger, split=split
    )

    model_checkpoints: list[pathlib.Path] = find_pretrained_checkpoints(config)
    onnx_export_dir: Path = output / "onnx"
    if not onnx_export_dir.exists():
        raise FileNotFoundError(f"No onnx model at {onnx_export_dir}. Use the export-onnx "
                                f"command first.")

    for i, model_ckpt in enumerate(model_checkpoints):
        if i == 0:
            logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_cls_model(config)
        checkpoint_epoch: int = load_pretrained(config, model, logger,
                                                checkpoint_path=model_ckpt, verbose=i == 0)
        model.to(device)
        model.eval()

        patch_encoder: Path = onnx_export_dir / "patch_encoder.onnx"
        patch_aggregator: Path = onnx_export_dir / "patch_aggregator.onnx"

        compare_pytorch_onnx_models(
            model,
            patch_encoder,
            patch_aggregator,
            includes_preprocessing=not exclude_preprocessing,
            device=device
        )

@torch.no_grad()
def validate(
    config,
    data_loader,
    model,
    criterion,
    logger,
    neptune_run,
    verbose: bool = True,
    return_predictions: bool = False
):

    model.eval()
    criterion.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_metrics: metrics.Metrics = metrics.Metrics(metrics=("auc", "ap", "accuracy"))

    predicted_scores: dict[int, tuple[float, Optional[AttentionMask]]] = {}

    end = time.time()
    for idx, (images, target, dataset_idx) in enumerate(data_loader):
        if isinstance(images, list):
            # In case of arbitrary resolution models the batch is provided as a list of tensors.
            images = [img.cuda(non_blocking=True) for img in images]
            # Remove views dimension. Always 1 during inference.
            images = [img.squeeze(dim=1) for img in images]
        else:
            images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # Compute output.
        if isinstance(images, list) and config.TEST.EXPORT_IMAGE_PATCHES:
            export_dirs: list[pathlib.Path] = [
                pathlib.Path(config.OUTPUT)/"images"/f"{dataset_idx.detach().cpu().tolist()[i]}"
                for i in range(len(dataset_idx))
            ]
            output, attention_masks = model(
                images, config.MODEL.FEATURE_EXTRACTION_BATCH, export_dirs
            )
        elif isinstance(images, list):
            output = model(images, config.MODEL.FEATURE_EXTRACTION_BATCH)
            attention_masks = [None] * len(images)
        else:
            if images.size(dim=1) > 1:
                predictions: list[torch.Tensor] = [
                    model(images[:, i]) for i in range(images.size(dim=1))
                ]
                predictions: torch.Tensor = torch.stack(predictions, dim=1)
                if config.TEST.VIEWS_REDUCTION_APPROACH == "max":
                    output: torch.Tensor = predictions.max(dim=1).values
                elif config.TEST.VIEWS_REDUCTION_APPROACH == "mean":
                    output: torch.Tensor = predictions.mean(dim=1)
                else:
                    raise TypeError(f"{config.TEST.VIEWS_REDUCTION_APPROACH} is not a "
                                    f"supported views reduction approach")
            else:
                images = images.squeeze(dim=1)  # Remove views dimension.
                output = model(images)
            attention_masks = [None] * images.size(0)

        loss = criterion(output.squeeze(dim=1), target)

        # Apply sigmoid to output.
        output = torch.sigmoid(output)

        # Update metrics.
        loss_meter.update(loss.item(), target.size(0))
        cls_metrics.update(output[:, 0].cpu(), target.cpu())

        # Keep predictions if requested.
        if return_predictions:
            batch_predictions: list[float] = output.squeeze(dim=1).detach().cpu().tolist()
            batch_dataset_idx: list[int] = dataset_idx.detach().cpu().tolist()
            predicted_scores.update({
                i: (p, m) for i, p, m in zip(batch_dataset_idx, batch_predictions, attention_masks)
            })

        # Measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 and verbose:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}] | '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | '
                f'Mem {memory_used:.0f}MB')

    metric_values: dict[str, np.ndarray] = cls_metrics.compute()
    auc: float = metric_values["auc"].item()
    ap: float = metric_values["ap"].item()
    acc: float = metric_values["accuracy"].item()

    if return_predictions:
        return acc, ap, auc, loss_meter.avg, predicted_scores
    else:
        return acc, ap, auc, loss_meter.avg
