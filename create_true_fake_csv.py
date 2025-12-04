#!/usr/bin/env python3
"""
终极版：所有 image 路径都相对于 ./datasets 根目录
输出格式示例：
synthbuster/dalle3/xxx.png,1,train
COCO/train2017/xxx.jpg,0,val
"""

import csv
import math
from pathlib import Path
from typing import Sequence
import click
from tqdm import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# 关键：所有路径最终都相对于这个根目录
DATASETS_ROOT = Path("./datasets").resolve()

def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p

def list_images(root: Path, desc: str):
    return sorted(list(tqdm(iter_images(root), desc=desc, unit="img")))

def make_relative_to_datasets(path: Path) -> str:
    """强制把路径变成相对于 ./datasets 的 posix 路径"""
    try:
        return path.resolve().relative_to(DATASETS_ROOT).as_posix()
    except ValueError:
        # 如果路径根本不在 datasets 下（极少见），直接报错提醒你
        raise click.ClickException(
            f"错误：图片路径不在 datasets 根目录下！\n"
            f"   图片路径: {path}\n"
            f"   datasets根目录: {DATASETS_ROOT}\n"
            f"   请确保所有数据都在 ./datasets/ 下"
        )

def split_paths(paths: Sequence[Path], ratio: float):
    n = len(paths)
    if n == 0: return [], []
    train_cnt = max(1, math.floor(n * ratio))
    if train_cnt == n and n > 1:
        train_cnt -= 1
    return paths[:train_cnt], paths[train_cnt:]

def build_entries(paths: Sequence[Path], label: int, ratio: float):
    train_p, val_p = split_paths(paths, ratio)
    entries = []
    for p in train_p:
        entries.append({"image": make_relative_to_datasets(p), "class": label, "split": "train"})
    for p in val_p:
        entries.append({"image": make_relative_to_datasets(p), "class": label, "split": "val"})
    return entries

def sample_coco(train_dir: Path, val_dir: Path, n: int):
    train_imgs = list_images(train_dir, "扫描 COCO train2017")
    val_imgs   = list_images(val_dir,   "扫描 COCO val2017")
    all_real = train_imgs + val_imgs
    if len(all_real) < n:
        raise click.ClickException(f"COCO 图像不足，需要 {n} 张，只有 {len(all_real)} 张")
    return all_real[:n]

@click.command()
@click.option("--fake_dir",   required=True,  type=click.Path(exists=True, file_okay=False, path_type=Path), help="fake 数据根目录（如 ./datasets/synthbuster/dalle3 或 ./datasets/stable-diffusion-3）")
@click.option("--coco_train", default=Path("./datasets/COCO/train2017"), type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--coco_val",   default=Path("./datasets/COCO/val2017"),   type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output",     default=Path("./data/train.csv"), type=click.Path(dir_okay=False, path_type=Path))
@click.option("--train_ratio", default=0.8, type=click.FloatRange(0.01, 0.99))
def main(fake_dir: Path, coco_train: Path, coco_val: Path, output: Path, train_ratio: float):
    output.parent.mkdir(parents=True, exist_ok=True)

    print("True/Fake CSV 生成器（路径统一相对于 datasets/）".center(70, "="))
    print(f"Fake 目录     → {fake_dir}")
    print(f"datasets 根目录 → {DATASETS_ROOT}")
    print(f"输出 CSV      → {output}")

    # 1. 扫描 fake
    fake_images = list_images(fake_dir, "扫描 Fake 图像（递归）")
    print(f"找到 Fake 图像: {len(fake_images)} 张")

    # 2. 采样 real
    real_images = sample_coco(coco_train, coco_val, len(fake_images))
    print(f"采样 Real 图像: {len(real_images)} 张")

    # 3. 生成 entries（所有路径都相对于 datasets/）
    entries = []
    entries += build_entries(fake_images, label=1, ratio=train_ratio)
    entries += build_entries(real_images,  label=0, ratio=train_ratio)

    # 4. 写入 CSV
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "class", "split"])
        w.writeheader()
        w.writerows(entries)

    print(f"\n成功！共 {len(entries)} 条记录，已写入：")
    print(f"   {output}")
    print(f"   示例路径：")
    print(f"   {entries[0]['image']}")
    print(f"   {entries[-1]['image']}")
    print("现在可以直接喂给任何训练脚本了！")

if __name__ == "__main__":
    main()