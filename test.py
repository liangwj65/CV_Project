# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
测试脚本 - SPAI + Effort SVD融合模型
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np

# 添加fusion路径到sys.path，使其可以独立运行
fusion_dir = Path(__file__).parent
if str(fusion_dir) not in sys.path:
    sys.path.insert(0, str(fusion_dir))
# 如果configs在父目录，也添加父目录
parent_dir = fusion_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fusion.config import get_config
from fusion.models.build_model import build_fusion_model, load_pretrained_weights
from fusion.data.dataset import build_loader_test
from fusion.utils import create_logger, validate
from fusion.models.losses import build_loss


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
    
    # 模型权重路径（训练后的模型，优先使用fusion目录下的）
    model_checkpoint = fusion_dir / "output" / "fusion" / "best_model.pth"
    if not model_checkpoint.exists():
        model_checkpoint = project_root / "output" / "fusion" / "best_model.pth"
    model_checkpoint = str(model_checkpoint)
    
    # 预训练权重路径（如果需要重新加载）
    spai_checkpoint = fusion_dir / "weights" / "spai.pth"
    if not spai_checkpoint.exists():
        spai_checkpoint = project_root / "weights" / "spai.pth"
    spai_checkpoint = str(spai_checkpoint) if spai_checkpoint.exists() else None
    
    effort_checkpoint = fusion_dir / "weights" / "effort_clip_l14.pth"
    if not effort_checkpoint.exists():
        effort_checkpoint = project_root / "weights" / "effort_clip_l14.pth"
    effort_checkpoint = str(effort_checkpoint) if effort_checkpoint.exists() else None
    
    # SVD参数（必须与训练时一致）
    svd_rank = 768
    use_svd = True
    clip_model = "ViT-B/16"
    
    # 测试数据路径（优先使用fusion目录下的，否则使用项目根目录下的）
    test_csv = fusion_dir / "data" / "test.csv"
    if not test_csv.exists():
        test_csv = project_root / "data" / "test.csv"
    test_csv = str(test_csv)
    
    csv_root = fusion_dir / "data"
    if not (csv_root / "test.csv").exists() and (project_root / "data" / "test.csv").exists():
        csv_root = project_root / "data"
    csv_root = str(csv_root)
    
    # 输出路径
    output_dir = fusion_dir / "output" / "fusion" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)
    
    # ========== 初始化 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = create_logger("fusion_test")
    
    logger.info("=" * 50)
    logger.info("SPAI + Effort SVD Fusion Testing")
    logger.info("=" * 50)
    logger.info(f"Device: {device}")
    logger.info(f"Model checkpoint: {model_checkpoint}")
    
    # 加载配置
    config = get_config({
        "cfg": config_path,
        "batch_size": 8,  # 测试时可以用较小的batch size
        "test_csv": [test_csv],
        "test_csv_root": [csv_root],
        "output": output_dir
    })
    
    # 输出目录已在上面创建
    
    # ========== 构建模型 ==========
    logger.info("Building model...")
    model = build_fusion_model(
        config,
        svd_rank=svd_rank,
        use_svd=use_svd,
        clip_model=clip_model
    )
    model = model.to(device)
    
    # 加载模型权重
    if os.path.exists(model_checkpoint):
        logger.info(f"Loading model from {model_checkpoint}")
        checkpoint = torch.load(model_checkpoint, map_location='cpu', weights_only=False)
        
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        if 'epoch' in checkpoint:
            logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'acc' in checkpoint:
            logger.info(f"Checkpoint ACC: {checkpoint['acc']:.3f}")
    else:
        logger.warning(f"Model checkpoint not found: {model_checkpoint}")
        logger.info("Trying to load pretrained weights...")
        
        # 尝试加载预训练权重
        load_pretrained_weights(
            model,
            spai_checkpoint=spai_checkpoint if (spai_checkpoint and os.path.exists(spai_checkpoint)) else None,
            effort_checkpoint=effort_checkpoint if (effort_checkpoint and os.path.exists(effort_checkpoint)) else None,
            strict=False
        )
    
    # ========== 构建测试数据加载器 ==========
    logger.info("Building test data loader...")
    test_datasets_names, test_datasets, test_loaders = build_loader_test(
        config, logger, split="test"
    )
    
    # ========== 构建损失函数 ==========
    criterion = build_loss(config)
    criterion = criterion.to(device)
    
    # ========== 测试 ==========
    logger.info("Starting testing...")
    results = {}
    
    for test_loader, test_dataset, test_name in zip(test_loaders, test_datasets, test_datasets_names):
        logger.info(f"\nTesting on: {test_name}")
        logger.info(f"Dataset size: {len(test_dataset)}")
        
        acc, ap, auc, loss = validate(model, test_loader, criterion, device, logger, config)
        
        results[test_name] = {
            'acc': acc,
            'ap': ap,
            'auc': auc,
            'loss': loss,
            'num_samples': len(test_dataset)
        }
        
        logger.info(f"Results for {test_name}:")
        logger.info(f"  ACC:  {acc:.4f}")
        logger.info(f"  AP:   {ap:.4f}")
        logger.info(f"  AUC:  {auc:.4f}")
        logger.info(f"  Loss: {loss:.4f}")
    
    # ========== 保存结果 ==========
    result_file = os.path.join(output_dir, "test_results.txt")
    with open(result_file, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("Test Results\n")
        f.write("=" * 50 + "\n\n")
        
        for test_name, result in results.items():
            f.write(f"Dataset: {test_name}\n")
            f.write(f"  Samples: {result['num_samples']}\n")
            f.write(f"  ACC:  {result['acc']:.4f}\n")
            f.write(f"  AP:   {result['ap']:.4f}\n")
            f.write(f"  AUC:  {result['auc']:.4f}\n")
            f.write(f"  Loss: {result['loss']:.4f}\n\n")
    
    logger.info(f"\nResults saved to {result_file}")
    logger.info("Testing completed!")


if __name__ == "__main__":
    main()

