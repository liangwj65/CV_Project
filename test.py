

import csv
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch

# 添加fusion路径到sys.path，使其可以独立运行
fusion_dir = Path(__file__).parent
if str(fusion_dir) not in sys.path:
    sys.path.insert(0, str(fusion_dir))

from config import get_config
from models.build_model import (
    build_fusion_model,
    load_pretrained_weights,
    apply_svd_to_model_backbone,
)
from data.dataset import build_loader_test
from utils import create_logger, validate
from models.losses import build_loss
from models.sid import AttentionMask


def _load_trained_weights(
    model: torch.nn.Module,
    checkpoint_path: Optional[Path],
    logger,
) -> Tuple[Optional[int], Optional[float]]:
    """加载训练权重，返回记录的epoch与ACC。"""
    if checkpoint_path is None or not checkpoint_path.exists():
        logger.warning("未找到训练好的checkpoint，尝试回退至预训练权重。")
        return None, None

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    epoch = checkpoint.get("epoch")
    acc = checkpoint.get("acc")
    if epoch is not None:
        logger.info(f"Checkpoint epoch: {epoch}")
    if acc is not None:
        logger.info(f"Checkpoint ACC: {acc:.4f}")
    return epoch, acc


def _export_predictions_csv(
    dataset,
    predictions: Dict[int, Tuple[float, Optional[AttentionMask]]],
    export_path: Path,
) -> None:
    """保存每张图像的预测分数与热力图路径。"""
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_index",
                "image",
                "score",
                "mask",
                "overlay",
                "overlayed_image",
            ],
        )
        writer.writeheader()
        for idx in sorted(predictions.keys()):
            score, attention = predictions[idx]
            entry = dataset.entries[idx]
            writer.writerow({
                "dataset_index": idx,
                "image": entry.get("image", ""),
                "score": f"{score:.6f}",
                "mask": str(attention.mask) if attention and attention.mask else "",
                "overlay": str(attention.overlay) if attention and attention.overlay else "",
                "overlayed_image": (
                    str(attention.overlayed_image) if attention and attention.overlayed_image else ""
                ),
            })


def _build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="运行融合模型测试，支持一个或多个test.csv。")
    parser.add_argument(
        "--test-csv",
        type=Path,
        nargs="+",
        default=[fusion_dir / "spai" / "data" / "real_my.csv"], #, fusion_dir / "spai" / "data" / "real_coco.csv"]
        help="用于评估的CSV文件路径，可重复多次传入。",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=fusion_dir / "output" / "fusion" / "best_model_sd3.pth",
        help="训练后模型的checkpoint路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=fusion_dir / "output" / "fusion_tests",
        help="输出目录，默认 fusion/output/fusion_tests。",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="real_my",
        help="输出子目录标签。",
    )
    return parser


def _merge_test_csvs(
    csv_paths: List[Path],
    export_root: Path,
    tag: str,
    dataset_root: Optional[Path] = None,
) -> Path:
    """将多个CSV合并为一个临时CSV，并保证图像路径指向datasets目录。"""
    combined_dir = export_root / "combined_csv"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_path = combined_dir / f"{tag}_merged.csv"

    combined_rows: list[dict[str, str]] = []
    fieldnames: Optional[list[str]] = None

    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=",")
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError("所有CSV需要有相同的列名。")

            for row in reader:
                row = dict(row)
                image_path = Path(row["image"])
                if not image_path.is_absolute():
                    # CSV相对路径统一指向datasets目录
                    base_dir = dataset_root if dataset_root else csv_path.parent
                    image_path = (base_dir / image_path).resolve()
                if dataset_root:
                    try:
                        image_path = image_path.resolve()
                        relative_path = image_path.relative_to(dataset_root.resolve())
                        row["image"] = str(relative_path)
                    except ValueError:
                        row["image"] = str(image_path)
                else:
                    row["image"] = str(image_path)
                combined_rows.append(row)

    if not combined_rows:
        raise RuntimeError("合并后的CSV为空，请确认输入文件是否包含样本。")

    with combined_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    return combined_path


def main():
    args = _build_arg_parser().parse_args()

    fusion_dir = Path(__file__).parent
    project_root = fusion_dir.parent

    config_path = fusion_dir / "configs" / "spai.yaml"
    if not config_path.exists():
        config_path = project_root / "configs" / "spai.yaml"

    test_csv_paths = [p.resolve() for p in args.test_csv]
    for test_csv in test_csv_paths:
        if not test_csv.exists():
            raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    model_checkpoint = args.checkpoint.resolve()
    if not model_checkpoint.exists():
        alt_checkpoint = project_root / "output" / "fusion" / "best_model.pth"
        model_checkpoint = alt_checkpoint if alt_checkpoint.exists() else args.checkpoint

    datasets_root = fusion_dir / "datasets"
    if not datasets_root.exists():
        alt_datasets = project_root / "datasets"
        datasets_root = alt_datasets if alt_datasets.exists() else datasets_root

    spai_checkpoint = fusion_dir / "weights" / "spai.pth"
    if not spai_checkpoint.exists():
        alt_spai = project_root / "weights" / "spai.pth"
        spai_checkpoint = alt_spai if alt_spai.exists() else None
    else:
        spai_checkpoint = spai_checkpoint

    effort_checkpoint = fusion_dir / "weights" / "effort_clip_l14.pth"
    if not effort_checkpoint.exists():
        alt_effort = project_root / "weights" / "effort_clip_l14.pth"
        effort_checkpoint = alt_effort if alt_effort.exists() else None

    svd_rank = 767
    use_svd = True
    clip_model = "ViT-B/16"
    batch_size = 8
    tag = args.tag
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    # ========== 初始化 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = create_logger("fusion_test")
    logger.info("=" * 50)
    logger.info("SPAI + Effort SVD Fusion Testing (aligned with SPAI test)")
    logger.info("=" * 50)
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {model_checkpoint}")
    logger.info(f"Dataset root: {datasets_root}")
    if len(test_csv_paths) > 1:
        merged_csv = _merge_test_csvs(
            test_csv_paths,
            output_root,
            tag,
            dataset_root=datasets_root,
        )
        test_csv_paths = [merged_csv]
        test_csv_roots = [datasets_root]
    else:
        test_csv_roots = [datasets_root for _ in test_csv_paths]
    logger.info(test_csv_roots)
    config = get_config({
        "cfg": str(config_path),
        "batch_size": batch_size,
        "test_csv": [str(p) for p in test_csv_paths],
        "test_csv_root": [str(root) for root in test_csv_roots],
        "output": str(output_root),
        "tag": tag,
    })

    # 强制开启热力图导出
    config.defrost()
    config.TEST.EXPORT_IMAGE_PATCHES = True
    config.freeze()

    run_output_dir = Path(config.OUTPUT)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be stored in {run_output_dir}")

    # ========== 构建模型 ==========
    logger.info("Building fusion model ...")
    model = build_fusion_model(
        config,
        svd_rank=svd_rank,
        use_svd=use_svd,
        clip_model=clip_model,
    ).to(device)

    checkpoint_epoch, checkpoint_acc = _load_trained_weights(model, model_checkpoint, logger)
    if checkpoint_epoch is None:
        load_pretrained_weights(
            model,
            spai_checkpoint=str(spai_checkpoint) if spai_checkpoint else None,
            effort_checkpoint=str(effort_checkpoint) if effort_checkpoint else None,
            strict=False,
        )
        if use_svd:
            apply_svd_to_model_backbone(model, svd_rank, device)
    model.eval()

    # ========== 数据与损失 ==========
    logger.info("Building test dataloaders ...")
    test_names, test_datasets, test_loaders = build_loader_test(config, logger, split="test")
    criterion = build_loss(config).to(device)

    # ========== 测试 ==========
    logger.info("Running evaluation ...")
    summary: Dict[str, Dict[str, float | int]] = {}
    for test_loader, test_dataset, test_name in zip(test_loaders, test_datasets, test_names):
        logger.info(f"\nTesting on {test_name} | Samples: {len(test_dataset)}")
        acc, ap, auc, loss, predictions = validate(
            config,
            test_loader,
            model,
            criterion,
            logger=logger,
            neptune_run=None,
            verbose=True,
            return_predictions=True,
        )
        summary[test_name] = {
            "acc": acc,
            "ap": ap,
            "auc": auc,
            "loss": loss,
            "num_samples": len(test_dataset),
        }
        logger.info(
            f"[{test_name}] ACC: {acc:.4f} | AP: {ap:.4f} | AUC: {auc:.4f} | Loss: {loss:.4f}"
        )

        if predictions:
            csv_path = run_output_dir / f"{test_name}_predictions.csv"
            _export_predictions_csv(test_dataset, predictions, csv_path)
            logger.info(f"Saved per-image predictions to {csv_path}")
        else:
            logger.warning(f"No predictions returned for {test_name}; CSV skipped.")

    result_file = run_output_dir / "test_results.txt"
    with result_file.open("w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("Fusion Test Results\n")
        f.write("=" * 50 + "\n\n")
        for name, stats in summary.items():
            f.write(f"Dataset: {name}\n")
            f.write(f"  Samples: {stats['num_samples']}\n")
            f.write(f"  ACC:  {stats['acc']:.4f}\n")
            f.write(f"  AP:   {stats['ap']:.4f}\n")
            f.write(f"  AUC:  {stats['auc']:.4f}\n")
            f.write(f"  Loss: {stats['loss']:.4f}\n\n")
    logger.info(f"Aggregated metrics saved to {result_file}")
    logger.info(f"Heatmaps exported under {run_output_dir / 'images'}")
    logger.info("Testing completed.")


if __name__ == "__main__":
    main()