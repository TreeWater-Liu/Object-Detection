#!/usr/bin/env python
"""Run RTMDet on raw multimodal triplets and write competition TXT files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multimodal_detection.fusion import discover_triplets, load_composite  # noqa: E402
from multimodal_detection.labels import (  # noqa: E402
    write_competition_predictions,
    zip_prediction_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Raw root with visible/infrared/depth")
    parser.add_argument("--config", type=Path, default=Path("configs/rtmdet_multimodal.py"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Fine-tuned RTMDet checkpoint")
    parser.add_argument("--output", type=Path, default=Path("predictions/rtmdet"))
    parser.add_argument("--fusion", choices=("yid", "rgb", "infrared", "depth"), default="yid")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--zip-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be in [0, 1]")
    if not 1 <= args.max_detections <= 100:
        raise ValueError("Competition output requires --max-detections in [1, 100]")

    try:
        from mmdet.apis import inference_detector, init_detector
        from mmdet.utils import register_all_modules
    except ImportError as exc:
        raise SystemExit(
            "MMDetection is not installed. Follow the RTMDet installation commands in README.md"
        ) from exc

    register_all_modules(init_default_scope=True)
    samples = discover_triplets(args.source, require_labels=False)
    output.mkdir(parents=True, exist_ok=True)
    model = init_detector(str(config), str(checkpoint), device=args.device)

    for sample in tqdm(samples, desc="RTMDet inference"):
        composite = load_composite(sample, fusion=args.fusion)
        height, width = composite.shape[:2]
        # inference_detector interprets ndarray input as BGR. Training files are
        # decoded as BGR and the config then applies bgr_to_rgb=True, so reverse
        # here to reproduce the identical preprocessing path.
        composite_bgr = np.ascontiguousarray(composite[..., ::-1])
        result = inference_detector(model, composite_bgr)
        instances = result.pred_instances.cpu()
        write_competition_predictions(
            output / f"{sample.sample_id}.txt",
            instances.bboxes.numpy(),
            instances.scores.numpy(),
            instances.labels.numpy(),
            image_width=width,
            image_height=height,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
        )

    if args.zip_output:
        zip_prediction_directory(output, args.zip_output.resolve())
        print(f"Submission archive: {args.zip_output.resolve()}")
    print(f"Wrote {len(samples)} prediction files to {output}")


if __name__ == "__main__":
    main()
