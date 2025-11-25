#!/usr/bin/env python3
"""
辅助脚本：根据单个真实数据集与单个伪造数据集生成train.csv
默认将CSV写入 data/train.csv，可通过参数自定义。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import click
from tqdm import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def iter_images(root: Path) -> Iterable[Path]:
    """Yield image files under *root* that match supported extensions."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def to_relative_path(path: Path, csv_root: Path, allow_absolute: bool) -> str:
    """
    Convert *path* to a relative string based on *csv_root*.
    Falls back to absolute paths when necessary (if allow_absolute is True).
    """
    try:
        return str(path.relative_to(csv_root))
    except ValueError:
        if allow_absolute:
            return str(path.resolve())
        raise click.ClickException(
            f"{path} 不在 CSV 根目录 {csv_root} 下，请使用 --csv_root 指定共同父目录 "
            "或使用 --allow_absolute"
        )


def collect_entries(
    dataset_dir: Path,
    label: int,
    split: str,
    csv_root: Path,
    allow_absolute: bool,
    desc: str,
) -> list[dict[str, str | int]]:
    entries: list[dict[str, str | int]] = []
    for img_path in tqdm(iter_images(dataset_dir), desc=desc, unit="image"):
        relative_image = to_relative_path(img_path, csv_root, allow_absolute)
        entries.append({"image": relative_image, "class": label, "split": split})
    return entries


@click.command()
@click.option(
    "--real_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="真实数据集根目录（label=0）",
)
@click.option(
    "--fake_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="伪造数据集根目录（label=1）",
)
@click.option(
    "--output_csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("./data/train_.csv"),
    show_default=True,
    help="输出CSV路径",
)
@click.option(
    "--csv_root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="CSV根目录（默认：output_csv的父目录）",
)
@click.option("--real_label", type=int, default=0, show_default=True, help="真实数据标签")
@click.option("--fake_label", type=int, default=1, show_default=True, help="伪造数据标签")
@click.option("--split", type=str, default="train", show_default=True, help="split列取值")
@click.option(
    "--allow_absolute",
    is_flag=True,
    default=False,
    show_default=True,
    help="若文件不在csv_root下，允许写入绝对路径",
)
def main(
    real_dir: Path,
    fake_dir: Path,
    output_csv: Path,
    csv_root: Path | None,
    real_label: int,
    fake_label: int,
    split: str,
    allow_absolute: bool,
) -> None:
    """根据真实/伪造数据集生成训练CSV。"""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if csv_root is None:
        csv_root = output_csv.parent
    csv_root = csv_root.resolve()

    print("=" * 50)
    print("True/Fake 训练CSV生成工具")
    print("=" * 50)
    print(f"真实数据集: {real_dir}")
    print(f"伪造数据集: {fake_dir}")
    print(f"输出CSV: {output_csv}")
    print(f"CSV根目录: {csv_root}")
    print(f"split: {split}")
    print("=" * 50)

    entries: list[dict[str, str | int]] = []
    entries.extend(
        collect_entries(
            real_dir, real_label, split, csv_root, allow_absolute, desc="扫描真实图像"
        )
    )
    entries.extend(
        collect_entries(
            fake_dir, fake_label, split, csv_root, allow_absolute, desc="扫描伪造图像"
        )
    )

    if not entries:
        raise click.ClickException("未找到任何图像，CSV未生成。")

    fieldnames = ["image", "class", "split"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)

    print(f"\n✅ 已写入 {len(entries)} 条记录到 {output_csv}")
    print("请在训练配置中设置 train_csv 与 csv_root 即可使用。")


if __name__ == "__main__":
    main()

