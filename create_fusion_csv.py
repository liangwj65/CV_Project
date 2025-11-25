#!/usr/bin/env python3
"""
辅助脚本：为fusion训练创建CSV文件
支持latent_diffusion_trainingset + COCO + LSUN数据集
"""

import sys
from pathlib import Path
from typing import Any

# 添加spai路径（支持spai在fusion目录下或父目录下）
fusion_dir = Path(__file__).parent
if str(fusion_dir) not in sys.path:
    sys.path.insert(0, str(fusion_dir))
# 如果spai在fusion目录下，也添加fusion/spai路径
spai_in_fusion = fusion_dir / "spai"
if spai_in_fusion.exists() and str(fusion_dir) not in sys.path:
    sys.path.insert(0, str(fusion_dir))

from spai.spai.tools.create_dmid_ldm_train_val_csv import (
    find_coco_samples,
    find_lsun_samples,
    write_csv_file
)
import click
from tqdm import tqdm
import random


@click.command()
@click.option("--train_dir", 
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True,
              help="latent_diffusion_trainingset的train目录路径")
@click.option("--val_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True,
              help="latent_diffusion_trainingset的val目录路径")
@click.option("--coco_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True,
              help="COCO数据集根目录（包含train2017文件夹）")
@click.option("--lsun_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=False,
              default=None,
              help="LSUN/CNNDetect数据集根目录（可选，暂时不使用）")
@click.option("--output_csv",
              type=click.Path(dir_okay=False, path_type=Path),
              default=Path("./fusion/data/train.csv"),
              help="输出的CSV文件路径（默认: ./fusion/data/train.csv）")
@click.option("--csv_root",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="CSV根目录（默认: output_csv的父目录的父目录）")
@click.option("--real_coco_filename",
              type=str,
              default="real_coco.txt",
              help="COCO文件名列表文件（默认: real_coco.txt）")
@click.option("--real_lsun_filename",
              type=str,
              default="real_lsun.txt",
              help="LSUN文件名列表文件（默认: real_lsun.txt）")
def main(
    train_dir: Path,
    val_dir: Path,
    coco_dir: Path,
    lsun_dir: Path,
    output_csv: Path,
    csv_root: Path,
    real_coco_filename: str,
    real_lsun_filename: str
):
    """为fusion训练创建CSV文件"""
    
    # 如果未提供lsun_dir，暂时跳过LSUN数据集
    use_lsun = lsun_dir is not None
    if not use_lsun:
        print("⚠️  注意: 未提供 --lsun_dir，将跳过LSUN数据集")
        # 创建一个临时目录用于调用原始函数（不会被实际使用）
        import tempfile
        lsun_dir = Path(tempfile.mkdtemp())
        # 创建空的LSUN文件列表，这样原始函数会跳过LSUN处理
        train_lsun_file = train_dir / real_lsun_filename
        val_lsun_file = val_dir / real_lsun_filename
        if not train_lsun_file.exists():
            train_lsun_file.touch()  # 创建空文件
        if not val_lsun_file.exists():
            val_lsun_file.touch()  # 创建空文件
    
    # 确保输出目录存在
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果没有指定csv_root，使用output_csv的父目录的父目录
    if csv_root is None:
        # 如果output_csv在fusion/data/下，csv_root应该是datasets目录
        if "fusion" in str(output_csv):
            csv_root = output_csv.parent.parent.parent / "datasets"
            if not csv_root.exists():
                csv_root = output_csv.parent.parent
        else:
            csv_root = output_csv.parent
    
    print("=" * 60)
    print("Fusion训练数据集CSV生成工具")
    print("=" * 60)
    print(f"Train目录: {train_dir}")
    print(f"Val目录: {val_dir}")
    print(f"COCO目录: {coco_dir}")
    if not use_lsun:
        print(f"LSUN目录: 未提供（跳过LSUN数据集）")
    else:
        print(f"LSUN目录: {lsun_dir}")
    print(f"输出CSV: {output_csv}")
    print(f"CSV根目录: {csv_root}")
    print(f"COCO文件名列表: {real_coco_filename}")
    print(f"LSUN文件名列表: {real_lsun_filename}")
    print("=" * 60)
    
    # 检查必要的文件是否存在
    print("\n检查必要文件...")
    
    train_coco_file = train_dir / real_coco_filename
    val_coco_file = val_dir / real_coco_filename
    
    missing_files = []
    if not train_coco_file.exists():
        missing_files.append(f"训练集COCO列表: {train_coco_file}")
    if not val_coco_file.exists():
        missing_files.append(f"验证集COCO列表: {val_coco_file}")
    
    # 只有在提供了lsun_dir时才检查LSUN文件
    if use_lsun:
        train_lsun_file = train_dir / real_lsun_filename
        val_lsun_file = val_dir / real_lsun_filename
        if not train_lsun_file.exists():
            missing_files.append(f"训练集LSUN列表: {train_lsun_file}")
        if not val_lsun_file.exists():
            missing_files.append(f"验证集LSUN列表: {val_lsun_file}")
    
    if missing_files:
        print("\n⚠️  警告: 以下文件不存在:")
        for f in missing_files:
            print(f"  - {f}")
        print("\n这些文件应该包含要使用的COCO和LSUN图像文件名列表。")
        print("如果数据集不包含这些文件，您可能需要手动创建。")
        response = input("\n是否继续? (y/n): ")
        if response.lower() != 'y':
            print("已取消。")
            return
    
    # 检查COCO和LSUN目录
    coco_train_dir = coco_dir / "train2017"
    if not coco_train_dir.exists():
        print(f"\n⚠️  警告: COCO train2017目录不存在: {coco_train_dir}")
        print("请确保COCO数据集已正确下载和解压。")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\n开始生成CSV文件...")
    
    # 直接实现CSV生成逻辑（不通过click命令）
    try:
        entries: list[dict[str, Any]] = []
        
        coco_copy_dir_name: str = "real_coco"
        lsun_copy_dir_name: str = "real_lsun"
        
        split_dirs: list[Path] = []
        split_labels: list[str] = []
        if train_dir is not None:
            split_dirs.append(train_dir)
            split_labels.append("train")
        if val_dir is not None:
            split_dirs.append(val_dir)
            split_labels.append("val")
        
        for s_dir, s_label in tqdm(zip(split_dirs, split_labels),
                                   desc="Finding synthetic images", unit="image"):
            # Make entries for the synthetic LDM data.
            data_gen = s_dir.rglob("*")
            for p in data_gen:
                path_parts: list[str] = p.parts
                if (p.is_file() and p.suffix == ".png"
                        and coco_copy_dir_name not in path_parts
                        and lsun_copy_dir_name not in path_parts):
                    entries.append({
                        "image": str(p.relative_to(csv_root)),
                        "class": 1,
                        "split": s_label
                    })
            
            # Make entries for COCO real data.
            real_coco_file: Path = s_dir / real_coco_filename
            if real_coco_file.exists():
                coco_samples: list[Path] = find_coco_samples(real_coco_file, coco_dir, s_label)
                for p in coco_samples:
                    entries.append({
                        "image": str(p.relative_to(csv_root)),
                        "class": 0,
                        "split": s_label
                    })
            
            # Make entries for LSUN real data (only if use_lsun is True).
            if use_lsun:
                real_lsun_file: Path = s_dir / real_lsun_filename
                if real_lsun_file.exists() and real_lsun_file.stat().st_size > 0:
                    try:
                        lsun_samples: list[Path] = find_lsun_samples(real_lsun_file, lsun_dir, s_label)
                        for p in lsun_samples:
                            entries.append({
                                "image": str(p.relative_to(csv_root)),
                                "class": 0,
                                "split": s_label
                            })
                    except Exception as e:
                        print(f"⚠️  警告: 处理LSUN数据时出错: {e}，跳过LSUN数据")
        
        # Write CSV file
        write_csv_file(entries, output_csv, delimiter=",")
        print(f"Exported CSV to {output_csv}")
        
        print(f"\n✅ CSV文件已成功生成: {output_csv}")
        print(f"   总共 {len(entries)} 条记录")
        print(f"\n📝 下一步:")
        print(f"1. 检查生成的CSV文件: {output_csv}")
        print(f"2. 在fusion/train.py中配置:")
        print(f"   train_csv = '{output_csv}'")
        print(f"   csv_root = '{csv_root}'")
        print(f"3. 开始训练: cd fusion && python train.py")
        
    except Exception as e:
        print(f"\n❌ 生成CSV时出错: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

