#!/usr/bin/env python
"""Fine-tune RF-DETR-Medium on a prepared multimodal-composite dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared dataset root")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/rfdetr/rf-detr-medium.pth"),
    )
    parser.add_argument("--output", type=Path, default=Path("work_dirs/rfdetr_medium"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()

    if not (dataset / "data.yaml").is_file():
        raise FileNotFoundError(f"Prepared YOLO dataset not found: {dataset / 'data.yaml'}")
    if not weights.is_file():
        raise FileNotFoundError(f"RF-DETR weights not found: {weights}")
    output.mkdir(parents=True, exist_ok=True)

    try:
        from rfdetr import RFDETRMedium
    except ImportError as exc:
        raise SystemExit(
            "RF-DETR is not installed. Run: pip install -r requirements-rfdetr.txt"
        ) from exc

    model = RFDETRMedium(
        pretrain_weights=str(weights),
        device=args.device,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model.train(
        dataset_dir=str(dataset),
        dataset_file="yolo",
        output_dir=str(output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        num_workers=args.num_workers,
        use_ema=True,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        run_test=False,
        tensorboard=True,
    )


if __name__ == "__main__":
    main()
