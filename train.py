# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
训练脚本 - SPAI + Effort SVD融合模型
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from timm.utils import AverageMeter
import time

# 添加fusion路径到sys.path，使其可以独立运行
fusion_dir = Path(__file__).parent
if str(fusion_dir) not in sys.path:
    sys.path.insert(0, str(fusion_dir))

from config import get_config
from models.build_model import (
    build_fusion_model,
    load_pretrained_weights,
    apply_svd_to_model_backbone
)
from data.dataset import build_loader, build_loader_test
from utils import create_logger, validate
from models.losses import build_loss


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, device, logger, writer, step, config=None, print_freq=10):
    """训练一个epoch"""
    model.eval()
    criterion.train()
    
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    
    end = time.time()
    for idx, (images, target, dataset_idx) in enumerate(train_loader):
        # 数据移到GPU
        if isinstance(images, list):
            images = [img.to(device, non_blocking=True) for img in images]
            images = [img.squeeze(dim=1) for img in images]
        else:
            images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # 前向传播
        optimizer.zero_grad()
        
        feature_extraction_batch = None
        if config is not None and hasattr(config, 'MODEL') and hasattr(config.MODEL, 'FEATURE_EXTRACTION_BATCH'):
            feature_extraction_batch = config.MODEL.FEATURE_EXTRACTION_BATCH
        
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

        # 反向传播
        loss.backward()
        

        # if config is not None and hasattr(config, 'TRAIN') and hasattr(config.TRAIN, 'CLIP_GRAD') and config.TRAIN.CLIP_GRAD:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        
        optimizer.step()
        
        # 更新统计
        loss_meter.update(loss.item(), target.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        scheduler.step()
        
        # 更新step计数器
        step += 1

        if idx % print_freq == 0:
            logger.info(
                f'Epoch [{epoch}][{idx}/{len(train_loader)}] '
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'
                f"LR {optimizer.param_groups[0]['lr']:.2e}\n"
            )
            # 记录到TensorBoard（使用step作为横坐标）
            if writer is not None:
                writer.add_scalar('Train/Loss', loss_meter.val, step)
                writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], step)
    
    return loss_meter.avg, step


def main():
    # ========== 配置参数 ==========
    # 获取fusion目录和项目根目录
    fusion_dir = Path(__file__).parent
    project_root = fusion_dir.parent
    
    # 配置文件路径（优先使用fusion目录下的，否则使用项目根目录下的）
    config_path = fusion_dir / "configs" / "spai.yaml"
    if not config_path.exists():
        config_path = project_root / "configs" / "spai.yaml"
    config_path = str(config_path)
    
    # 预训练权重路径（优先使用fusion目录下的，否则使用项目根目录下的）
    spai_checkpoint = fusion_dir / "weights" / "spai.pth"
    if not spai_checkpoint.exists():
        spai_checkpoint = project_root / "weights" / "spai.pth"
    spai_checkpoint = str(spai_checkpoint) if spai_checkpoint.exists() else None
    
    effort_checkpoint = fusion_dir / "weights" / "effort_clip_l14.pth"
    if not effort_checkpoint.exists():
        effort_checkpoint = project_root / "weights" / "effort_clip_l14.pth"
    effort_checkpoint = str(effort_checkpoint) if effort_checkpoint.exists() else None
    
    # SVD参数
    svd_rank = 767  # SVD保留的rank（ViT-B/16的embed_dim是768，可以设置为768-1或更小）
    use_svd = True  # 是否使用SVD
    
    # CLIP模型
    clip_model = "ViT-B/16"  # 或 "ViT-L/14"
    
    # 训练参数
    batch_size = 72
    num_epochs = 10
    learning_rate = 5e-7
    weight_decay = 0.05
    max_grad_norm = 1.0
    save_dir = fusion_dir / "output" / "fusion"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir = str(save_dir)
    save_freq = 1  # 每N个epoch保存一次
    
    # 数据路径（优先使用fusion目录下的，否则使用项目根目录下的）
    train_csv = fusion_dir / "data" / "train.csv"
    if not train_csv.exists():
        train_csv = project_root / "data" / "train.csv"
    train_csv = str(train_csv)

    # 使用同一个 CSV，通过 split 字段区分 train/val
    val_csv = train_csv
    
    csv_root = fusion_dir / "datasets"
    if not (csv_root / "train.csv").exists() and (project_root / "data" / "train.csv").exists():
        csv_root = project_root / "data"
    csv_root = str(csv_root)
    
    # ========== 初始化 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = create_logger("fusion_train")
    
    logger.info("=" * 50)
    logger.info("SPAI + Effort SVD Fusion Training")
    logger.info("=" * 50)
    logger.info(f"Device: {device}")
    logger.info(f"SVD Rank: {svd_rank}, Use SVD: {use_svd}")
    logger.info(f"CLIP Model: {clip_model}")
    
    # 加载配置
    config = get_config({
        "cfg": config_path,
        "batch_size": batch_size,
        "data_path": train_csv,
        "csv_root_dir": csv_root,
        "output": save_dir,
        "tag": "fusion_svd",
        "learning_rate": learning_rate
    })

    # 为双阶段微调启用更严格的梯度裁剪
    if hasattr(config, "defrost"):
        config.defrost()
        config.TRAIN.CLIP_GRAD = max_grad_norm
        config.freeze()

    # 保存目录已在上面创建
    
    # TensorBoard
    writer = SummaryWriter(log_dir=save_dir)
    
    # ========== 构建模型 ==========
    logger.info("Building model...")
    model = build_fusion_model(
        config,
        svd_rank=svd_rank,
        use_svd=use_svd,
        clip_model=clip_model
    )
    model = model.to(device)
    
    # 加载预训练权重
    if (spai_checkpoint and os.path.exists(spai_checkpoint)) or (effort_checkpoint and os.path.exists(effort_checkpoint)):
        load_pretrained_weights(
            model,
            spai_checkpoint=spai_checkpoint if (spai_checkpoint and os.path.exists(spai_checkpoint)) else None,
            effort_checkpoint=effort_checkpoint if (effort_checkpoint and os.path.exists(effort_checkpoint)) else None,
            strict=False
        )
    else:
        logger.warning("No pretrained weights found, training from scratch")

    # 在完整加载SPAI权重后再执行SVD降阶，确保backbone来源一致
    if use_svd:
        apply_svd_to_model_backbone(model, svd_rank, device)
    
    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # ========== 构建数据加载器 ==========
    logger.info("Building data loaders...")
    train_loader = build_loader(config, logger, is_train=True)
    
    # 验证集（如果有）
    val_loader = None
    if val_csv and os.path.exists(val_csv):
        val_config = get_config({
            "cfg": config_path,
            "batch_size": batch_size,
            "test_csv": [val_csv],
            "test_csv_root": [csv_root]
        })
        val_datasets_names, val_datasets, val_loaders = build_loader_test(val_config, logger, split="val")
        if len(val_loaders) > 0:
            val_loader = val_loaders[0]
    
    # ========== 构建损失函数和优化器 ==========
    criterion = build_loss(config)
    criterion = criterion.to(device)
    
    # 优化器 - 只优化可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.Adam(
    #     trainable_params,
    #     lr=learning_rate,
    #     betas=(0.9, 0.999),
    #     eps=1e-8,
    #     weight_decay=weight_decay,
    #     amsgrad=False
    # )
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=num_epochs,
    #     eta_min=learning_rate * 0.01
    # )
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=0.0005,  # 直接 ×100！SGD 完全不怕大 lr
        momentum=0.95,
        weight_decay=weight_decay,
        nesterov=True
    )

    total_steps = len(train_loader) * num_epochs
    warmup_steps = 500

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=0)
        ],
        milestones=[warmup_steps]
    )
    # 学习率调度器

    
    # ========== 训练循环 ==========
    logger.info("Starting training...")
    best_acc = 0.0
    global_step = 0  # 全局step计数器
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch [{epoch}/{num_epochs}]")
        
        # 训练
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch, device, logger, writer, global_step, config
        )

        
        # 验证
        if val_loader is not None:
            acc, ap, auc, val_loss = validate(model, val_loader, criterion, device, logger, config)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/ACC', acc, epoch)
            writer.add_scalar('Val/AP', ap, epoch)
            writer.add_scalar('Val/AUC', auc, epoch)
            
            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'acc': acc,
                    'ap': ap,
                    'auc': auc
                }, os.path.join(save_dir, 'best_model.pth'))
                logger.info(f"Saved best model with ACC: {acc:.3f}")
        elif val_loader is not None:
            logger.info("本轮跳过验证")
        
        # 定期保存
        if epoch % save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
            logger.info(f"Saved checkpoint at epoch {epoch}")
    
    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(save_dir, 'final_model.pth'))
    
    logger.info("Training completed!")
    writer.close()


if __name__ == "__main__":
    main()

