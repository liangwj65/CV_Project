import logging
from pathlib import Path
from typing import Optional

from spai.config import get_config
from spai.data import build_loader_test
from spai.models import build_cls_model
from spai.models.losses import build_loss as build_loss_fn
from spai.utils import load_pretrained, find_pretrained_checkpoints
from spai.models.sid import AttentionMask  # noqa: F401  (kept for type parity with package infer)
from spai_main_functions import validate  # reuse the existing evaluation logic


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
    """
    Mimics the package's infer command with identical defaults.
    It supports both CSV inputs (with optional labels) and directory inputs.
    """
    if not input_paths:
        raise ValueError("input_paths 不能为空：需要至少提供一个目录或CSV路径")
    if input_csv_root_dir is None:
        input_csv_root_dir = []

    # Build runtime config
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
        "opts": extra_options,
    })

    output.mkdir(exist_ok=True, parents=True)

    # Console logger (match package behavior)
    global logger  # align with package logging usage
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Dataloaders
    test_datasets_names, test_datasets, test_loaders = build_loader_test(
        config, logger, split=split, dummy_csv_dir=output
    )

    # Load checkpoint & build model/loss
    model_ckpt: Path = find_pretrained_checkpoints(config)[0]
    criterion = build_loss_fn(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model_instance = build_cls_model(config)
    model_instance.cuda()
    load_pretrained(config, model_instance, logger, checkpoint_path=model_ckpt, verbose=False)

    # Inference + optional metrics update (for CSV with labels)
    for test_loader, test_dataset, test_name, in_path in zip(
        test_loaders, test_datasets, test_datasets_names, input_paths
    ):
        acc, ap, auc, loss, predictions = validate(
            config, test_loader, model_instance, criterion, logger, None, return_predictions=True
        )

        # Only CSV inputs may contain ground-truth labels
        if in_path.is_file():
            logger.info(f"Test | {test_name} | Images: {len(test_dataset)} | loss: {loss:.4f}")
            logger.info(f"Test | {test_name} | Images: {len(test_dataset)} | ACC: {acc:.3f}")
            if test_dataset.get_classes_num() > 1:  # AUC/AP not meaningful for single class
                logger.info(f"Test | {test_name} | Images: {len(test_dataset)} | AP: {ap:.3f}")
                logger.info(f"Test | {test_name} | Images: {len(test_dataset)} | AUC: {auc:.3f}")

        # Update output CSV with predictions and optional attention masks
        if predictions is not None:
            column_name: str = f"{tag}"
            scores: dict[int, float] = {i: t[0] for i, t in predictions.items()}
            attention_masks: dict[int, Path] = {
                i: t[1].mask for i, t in predictions.items() if t[1] is not None
            }
            test_dataset.update_dataset_csv(column_name, scores, export_dir=Path(output))
            if len(attention_masks) == len(scores):
                test_dataset.update_dataset_csv(
                    f"{column_name}_mask", attention_masks, export_dir=Path(output)
                )


__all__ = ["infer"]

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser("SPAI infer (debug entry)")
    parser.add_argument("--cfg", type=Path, default=Path("./configs/spai.yaml"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input", "--input-paths", dest="input_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--input-csv-root-dir", dest="input_csv_root_dir", type=Path, nargs="*", default=[])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--lmdb", dest="lmdb_path", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=Path("./weights/spai.pth"))
    parser.add_argument("--output", type=Path, default=Path("./output"))
    parser.add_argument("--tag", type=str, default="spai")
    parser.add_argument("--resize-to", dest="resize_to", type=int, default=None)
    parser.add_argument("--opt", dest="extra_options", nargs=2, action="append", default=[])

    args = parser.parse_args()
    extra_opts: tuple[tuple[str, str], ...] = tuple((str(k), str(v)) for k, v in args.extra_options)
    infer(
        cfg=args.cfg,
        batch_size=args.batch_size,
        input_paths=list(args.input_paths),
        input_csv_root_dir=list(args.input_csv_root_dir),
        split=args.split,
        lmdb_path=args.lmdb_path,
        model=args.model,
        output=args.output,
        tag=args.tag,
        resize_to=args.resize_to,
        extra_options=extra_opts,
    )


if __name__ == "__main__":
    main()


