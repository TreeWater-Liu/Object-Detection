#!/usr/bin/env python
"""Validate raw triplets and create model-ready YOLO and COCO datasets."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multimodal_detection.constants import CLASS_NAMES  # noqa: E402
from multimodal_detection.fusion import TripletSample, discover_triplets, load_composite  # noqa: E402
from multimodal_detection.labels import (  # noqa: E402
    build_coco_annotations,
    read_yolo_labels,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Raw root with visible/infrared/depth/labels")
    parser.add_argument("--output", type=Path, required=True, help="New or empty prepared dataset directory")
    parser.add_argument("--fusion", choices=("yid", "rgb", "infrared", "depth"), default="yid")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-valid-depth-mm", type=int, default=300)
    parser.add_argument("--max-depth-mm", type=int, default=20_000)
    return parser.parse_args()


def multilabel_split(
    samples: list[TripletSample],
    labels_by_id: dict[str, list[tuple[int, float, float, float, float]]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[TripletSample], list[TripletSample]]:
    """Make a deterministic split while keeping singleton classes in training."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if len(samples) < 2:
        raise ValueError("At least two labelled samples are required for train/validation splitting")

    target_size = min(max(1, round(len(samples) * val_ratio)), len(samples) - 1)
    classes_by_id = {
        sample.sample_id: {row[0] for row in labels_by_id[sample.sample_id]}
        for sample in samples
    }
    total_presence = Counter(class_id for classes in classes_by_id.values() for class_id in classes)
    target_presence = {
        class_id: (max(1, round(count * val_ratio)) if count >= 2 else 0)
        for class_id, count in total_presence.items()
    }

    rng = random.Random(seed)
    remaining = samples.copy()
    rng.shuffle(remaining)
    validation: list[TripletSample] = []
    validation_presence: Counter[int] = Counter()
    remaining_presence = total_presence.copy()

    while remaining and len(validation) < target_size:
        eligible = [
            sample
            for sample in remaining
            if all(remaining_presence[class_id] > 1 for class_id in classes_by_id[sample.sample_id])
        ]
        candidates = eligible or remaining

        def gain(sample: TripletSample) -> tuple[float, float]:
            classes = classes_by_id[sample.sample_id]
            deficit_gain = sum(
                max(target_presence[class_id] - validation_presence[class_id], 0)
                / max(target_presence[class_id], 1)
                for class_id in classes
            )
            rarity = sum(1.0 / total_presence[class_id] for class_id in classes)
            return deficit_gain, rarity

        selected = max(candidates, key=gain)
        remaining.remove(selected)
        validation.append(selected)
        for class_id in classes_by_id[selected.sample_id]:
            validation_presence[class_id] += 1
            remaining_presence[class_id] -= 1

    return sorted(remaining, key=lambda sample: sample.sample_id), sorted(
        validation, key=lambda sample: sample.sample_id
    )


def ensure_empty_output(output: Path) -> None:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    ensure_empty_output(output)

    samples = discover_triplets(source, require_labels=True)
    labels_by_id = {
        sample.sample_id: read_yolo_labels(sample.label)  # type: ignore[arg-type]
        for sample in samples
    }
    train_samples, valid_samples = multilabel_split(
        samples,
        labels_by_id,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    split_map = {"train": train_samples, "valid": valid_samples}
    coco_records: dict[str, list[tuple[str, int, int, list[tuple[int, float, float, float, float]]]]] = {
        "train": [],
        "valid": [],
    }
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for split_name, split_samples in split_map.items():
        image_dir = output / split_name / "images"
        label_dir = output / split_name / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for sample in tqdm(split_samples, desc=f"Preparing {split_name}"):
            composite = load_composite(
                sample,
                fusion=args.fusion,
                min_valid_depth_mm=args.min_valid_depth_mm,
                max_depth_mm=args.max_depth_mm,
            )
            image_path = image_dir / f"{sample.sample_id}.png"
            Image.fromarray(composite, mode="RGB").save(image_path, compress_level=3)

            label_path = label_dir / f"{sample.sample_id}.txt"
            source_text = sample.label.read_text(encoding="utf-8")  # type: ignore[union-attr]
            label_path.write_text(source_text, encoding="utf-8")

            height, width = composite.shape[:2]
            coco_records[split_name].append(
                (image_path.name, width, height, labels_by_id[sample.sample_id])
            )
            values = composite.astype(np.float64)
            channel_sum += values.sum(axis=(0, 1))
            channel_square_sum += np.square(values).sum(axis=(0, 1))
            pixel_count += width * height

    for split_name, records in coco_records.items():
        save_json(build_coco_annotations(records), output / "annotations" / f"{split_name}.json")

    channel_mean = channel_sum / pixel_count
    channel_variance = np.maximum(channel_square_sum / pixel_count - np.square(channel_mean), 0.0)
    channel_std = np.sqrt(channel_variance)

    yaml_data = {
        "path": output.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    class_box_counts = Counter(row[0] for rows in labels_by_id.values() for row in rows)
    manifest = {
        "source": str(source),
        "output": str(output),
        "fusion": args.fusion,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "min_valid_depth_mm": args.min_valid_depth_mm,
        "max_depth_mm": args.max_depth_mm,
        "classes": list(CLASS_NAMES),
        "sample_count": len(samples),
        "train_ids": [sample.sample_id for sample in train_samples],
        "valid_ids": [sample.sample_id for sample in valid_samples],
        "box_counts": {CLASS_NAMES[index]: class_box_counts[index] for index in range(len(CLASS_NAMES))},
        "channel_mean_0_255": channel_mean.tolist(),
        "channel_std_0_255": channel_std.tolist(),
    }
    save_json(manifest, output / "manifest.json")

    print(f"Prepared {len(train_samples)} train and {len(valid_samples)} validation samples in {output}")
    print(f"Fusion={args.fusion}; channel mean={channel_mean.round(3).tolist()}")
    print(f"Channel std={channel_std.round(3).tolist()}")


if __name__ == "__main__":
    main()
