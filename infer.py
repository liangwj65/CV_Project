# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
推理脚本 - SPAI + Effort SVD融合模型
"""

import os
import sys
from pathlib import Path
from typing import Optional
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


def infer(
    cfg: Path = None,
    batch_size: int = 1,
    input_paths: list[Path] = None,
    input_csv_root_dir: list[Path] = None,
    split: str = "test",
    model: Path = None,
    output: Path = None,
    tag: str = "fusion",
    svd_rank: int = 768,
    use_svd: bool = True,
    clip_model: str = "ViT-B/16",
) -> None:
    """
    推理函数
    
    Args:
        cfg: 配置文件路径
        batch_size: 批次大小
        input_paths: 输入CSV文件路径列表
        input_csv_root_dir: CSV根目录列表
        split: 数据集分割
        model: 模型权重路径
        output: 输出目录
        tag: 标签
        svd_rank: SVD rank
        use_svd: 是否使用SVD
        clip_model: CLIP模型名称
    """
    # 获取fusion目录和项目根目录
    fusion_dir = Path(__file__).parent
    project_root = fusion_dir.parent
    
    # 默认配置
    if cfg is None:
        cfg = fusion_dir / "configs" / "spai.yaml"
        if not cfg.exists():
            cfg = project_root / "configs" / "spai.yaml"
    
    if model is None:
        model = fusion_dir / "output" / "fusion" / "best_model.pth"
        if not model.exists():
            model = project_root / "output" / "fusion" / "best_model.pth"
    
    if output is None:
        output = fusion_dir / "output" / "inference"
    
    if input_paths is None:
        input_paths = [fusion_dir / "data" / "test.csv"]
        if not input_paths[0].exists():
            input_paths = [project_root / "data" / "test.csv"]
    
    if input_csv_root_dir is None:
        input_csv_root_dir = [fusion_dir / "data"]
        if not (fusion_dir / "data" / "test.csv").exists():
            input_csv_root_dir = [project_root / "data"]
    
    # 确保输出目录存在
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = create_logger("fusion_infer")
    
    logger.info("=" * 50)
    logger.info("SPAI + Effort SVD Fusion Inference")
    logger.info("=" * 50)
    logger.info(f"Device: {device}")
    logger.info(f"Model: {model}")
    logger.info(f"Output: {output}")
    
    # 加载配置
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "test_csv": [str(p) for p in input_paths],
        "test_csv_root": [str(p) for p in input_csv_root_dir],
        "output": str(output),
        "tag": tag
    })
    
    # 构建模型
    logger.info("Building model...")
    model_instance = build_fusion_model(
        config,
        svd_rank=svd_rank,
        use_svd=use_svd,
        clip_model=clip_model
    )
    model_instance = model_instance.to(device)
    
    # 加载模型权重
    if os.path.exists(model):
        logger.info(f"Loading model from {model}")
        checkpoint = torch.load(model, map_location='cpu', weights_only=False)
        
        if 'model' in checkpoint:
            model_instance.load_state_dict(checkpoint['model'], strict=False)
        else:
            model_instance.load_state_dict(checkpoint, strict=False)
        
        if 'epoch' in checkpoint:
            logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'acc' in checkpoint:
            logger.info(f"Checkpoint ACC: {checkpoint['acc']:.3f}")
    else:
        logger.warning(f"Model checkpoint not found: {model}")
        logger.info("Trying to load pretrained weights...")
        
        # 尝试加载预训练权重
        spai_checkpoint = fusion_dir / "weights" / "spai.pth"
        if not spai_checkpoint.exists():
            spai_checkpoint = project_root / "weights" / "spai.pth"
        spai_checkpoint = str(spai_checkpoint) if spai_checkpoint.exists() else None
        
        effort_checkpoint = fusion_dir / "weights" / "effort_clip_l14.pth"
        if not effort_checkpoint.exists():
            effort_checkpoint = project_root / "weights" / "effort_clip_l14.pth"
        effort_checkpoint = str(effort_checkpoint) if effort_checkpoint.exists() else None
        
        load_pretrained_weights(
            model_instance,
            spai_checkpoint=spai_checkpoint,
            effort_checkpoint=effort_checkpoint,
            strict=False
        )
    
    # 构建测试数据加载器
    logger.info("Building test data loader...")
    test_datasets_names, test_datasets, test_loaders = build_loader_test(
        config, logger, split=split
    )
    
    # 构建损失函数
    criterion = build_loss(config)
    criterion = criterion.to(device)
    
    # 推理
    logger.info("Starting inference...")
    results = {}
    
    for test_loader, test_dataset, test_name in zip(test_loaders, test_datasets, test_datasets_names):
        logger.info(f"\nInferencing on: {test_name}")
        logger.info(f"Dataset size: {len(test_dataset)}")
        
        acc, ap, auc, loss = validate(model_instance, test_loader, criterion, device, logger, config)
        
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
    
    # 保存结果
    result_file = output / "inference_results.txt"
    with open(result_file, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("Inference Results\n")
        f.write("=" * 50 + "\n\n")
        
        for test_name, result in results.items():
            f.write(f"Dataset: {test_name}\n")
            f.write(f"  Samples: {result['num_samples']}\n")
            f.write(f"  ACC:  {result['acc']:.4f}\n")
            f.write(f"  AP:   {result['ap']:.4f}\n")
            f.write(f"  AUC:  {result['auc']:.4f}\n")
            f.write(f"  Loss: {result['loss']:.4f}\n\n")
    
    logger.info(f"\nResults saved to {result_file}")
    logger.info("Inference completed!")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser("Fusion Model Inference")
    parser.add_argument("--cfg", type=Path, default=None, help="Config file path")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--input", "--input-paths", dest="input_paths", type=Path, nargs="+", 
                       default=None, help="Input CSV file paths")
    parser.add_argument("--input-csv-root-dir", dest="input_csv_root_dir", type=Path, nargs="*",
                       default=None, help="CSV root directories")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--model", type=Path, default=None, help="Model checkpoint path")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--tag", type=str, default="fusion", help="Tag")
    parser.add_argument("--svd-rank", type=int, default=768, help="SVD rank")
    parser.add_argument("--use-svd", action="store_true", default=True, help="Use SVD")
    parser.add_argument("--no-svd", dest="use_svd", action="store_false", help="Don't use SVD")
    parser.add_argument("--clip-model", type=str, default="ViT-B/16", 
                       choices=["ViT-B/16", "ViT-L/14"], help="CLIP model")
    
    args = parser.parse_args()
    
    infer(
        cfg=args.cfg,
        batch_size=args.batch_size,
        input_paths=args.input_paths,
        input_csv_root_dir=args.input_csv_root_dir,
        split=args.split,
        model=args.model,
        output=args.output,
        tag=args.tag,
        svd_rank=args.svd_rank,
        use_svd=args.use_svd,
        clip_model=args.clip_model
    )


if __name__ == "__main__":
    main()

