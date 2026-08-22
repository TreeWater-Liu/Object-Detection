"""Competition label validation, conversion, and prediction export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .constants import CLASS_NAMES, NUM_CLASSES


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields, got {len(fields)}")
        class_id = int(fields[0])
        values = tuple(float(value) for value in fields[1:])
        if not 0 <= class_id < NUM_CLASSES:
            raise ValueError(f"{path}:{line_number}: invalid class id {class_id}")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{path}:{line_number}: normalized coordinates must be in [0, 1]")
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError(f"{path}:{line_number}: width and height must be positive")
        rows.append((class_id, *values))
    return rows


def build_coco_annotations(
    records: Iterable[tuple[str, int, int, list[tuple[int, float, float, float, float]]]],
) -> dict:
    images: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1

    for image_id, (file_name, width, height, labels) in enumerate(records, start=1):
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        for class_id, center_x, center_y, norm_w, norm_h in labels:
            box_w = norm_w * width
            box_h = norm_h * height
            x_min = max(0.0, (center_x - norm_w / 2.0) * width)
            y_min = max(0.0, (center_y - norm_h / 2.0) * height)
            box_w = min(box_w, width - x_min)
            box_h = min(box_h, height - y_min)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [x_min, y_min, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": class_id + 1, "name": name, "supercategory": "object"}
            for class_id, name in enumerate(CLASS_NAMES)
        ],
    }


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_competition_predictions(
    path: Path,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    score_threshold: float,
    max_detections: int = 100,
) -> None:
    """Write [class cx cy w h confidence] rows in competition format."""

    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if not (len(boxes) == len(scores) == len(class_ids)):
        raise ValueError("boxes, scores, and class_ids must have equal length")

    valid = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(scores)
        & (scores >= score_threshold)
        & (class_ids >= 0)
        & (class_ids < NUM_CLASSES)
    )
    boxes = boxes[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    order = np.argsort(-scores, kind="stable")[:max_detections]
    boxes = boxes[order]
    scores = scores[order]
    class_ids = class_ids[order]

    lines: list[str] = []
    for box, score, class_id in zip(boxes, scores, class_ids, strict=True):
        x1 = float(np.clip(box[0], 0.0, image_width))
        y1 = float(np.clip(box[1], 0.0, image_height))
        x2 = float(np.clip(box[2], 0.0, image_width))
        y2 = float(np.clip(box[3], 0.0, image_height))
        if x2 <= x1 or y2 <= y1:
            continue
        center_x = ((x1 + x2) / 2.0) / image_width
        center_y = ((y1 + y2) / 2.0) / image_height
        norm_w = (x2 - x1) / image_width
        norm_h = (y2 - y1) / image_height
        lines.append(
            f"{int(class_id)} {center_x:.8f} {center_y:.8f} "
            f"{norm_w:.8f} {norm_h:.8f} {float(score):.8f}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def zip_prediction_directory(prediction_dir: Path, zip_path: Path) -> None:
    txt_files = sorted(prediction_dir.glob("*.txt"))
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for txt_file in txt_files:
            archive.write(txt_file, arcname=txt_file.name)
