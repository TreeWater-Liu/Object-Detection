#!/usr/bin/env python
"""Fine-tune RTMDet-m on a prepared multimodal-composite COCO dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared dataset root")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rtmdet_multimodal.py"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/rtmdet/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"),
    )
    parser.add_argument("--output", type=Path, default=Path("work_dirs/rtmdet_m"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    config_path = args.config.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()

    required = (
        dataset / "annotations" / "train.json",
        dataset / "annotations" / "valid.json",
        dataset / "manifest.json",
        config_path,
        weights,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(path) for path in missing))

    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
        from mmdet.utils import register_all_modules
    except ImportError as exc:
        raise SystemExit(
            "MMDetection is not installed. Follow the RTMDet installation commands in README.md"
        ) from exc

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(config_path)
    dataset_root = str(dataset) + "/"
    cfg.work_dir = str(output)
    cfg.load_from = str(weights)
    cfg.randomness = dict(seed=args.seed, deterministic=False)

    for loader_name, split_name, annotation_name in (
        ("train_dataloader", "train", "train.json"),
        ("val_dataloader", "valid", "valid.json"),
        ("test_dataloader", "valid", "valid.json"),
    ):
        loader = cfg[loader_name]
        loader.dataset.data_root = dataset_root
        loader.dataset.ann_file = f"annotations/{annotation_name}"
        loader.dataset.data_prefix = dict(img=f"{split_name}/images/")
        loader.num_workers = args.num_workers
        loader.persistent_workers = args.num_workers > 0
    cfg.train_dataloader.batch_size = args.batch_size

    annotation_path = str(dataset / "annotations" / "valid.json")
    cfg.val_evaluator.ann_file = annotation_path
    cfg.test_evaluator.ann_file = annotation_path

    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    means = [float(value) for value in manifest["channel_mean_0_255"]]
    stds = [max(float(value), 1.0) for value in manifest["channel_std_0_255"]]
    cfg.model.data_preprocessor.mean = means
    cfg.model.data_preprocessor.std = stds

    cfg.train_cfg.max_epochs = args.epochs
    cfg.optim_wrapper.optimizer.lr = args.lr
    cosine = cfg.param_scheduler[-1]
    cosine.begin = 0
    cosine.end = args.epochs
    cosine.T_max = args.epochs
    cosine.eta_min = args.lr * 0.05
    if args.amp:
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.loss_scale = "dynamic"

    output.mkdir(parents=True, exist_ok=True)
    cfg.dump(output / "effective_config.py")
    print(f"Dataset channel mean={means}, std={stds}")
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
