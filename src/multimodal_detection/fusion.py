"""Read aligned RGB/infrared/depth samples and build model-compatible images.

The first baseline deliberately keeps the published detectors unchanged. It
encodes the aligned modalities into a three-channel composite:

    channel 0: visible-image luminance
    channel 1: infrared intensity
    channel 2: inverse depth (nearer valid pixels are brighter)

This is an early-fusion baseline, not a claim that the three channels have RGB
semantics. It provides a reproducible starting point while retaining the COCO
pretrained input stems of RF-DETR and RTMDet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class TripletSample:
    """Paths belonging to one spatially aligned competition sample."""

    sample_id: str
    visible: Path
    infrared: Path
    depth: Path
    label: Path | None = None


def _files_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing modality directory: {directory}")

    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in result:
            raise ValueError(
                f"Duplicate sample stem {path.stem!r} in {directory}: "
                f"{result[path.stem].name}, {path.name}"
            )
        result[path.stem] = path
    return result


def discover_triplets(root: str | Path, *, require_labels: bool) -> list[TripletSample]:
    """Discover samples from visible/infrared/depth[/labels] directories."""

    root = Path(root).resolve()
    visible = _files_by_stem(root / "visible")
    infrared = _files_by_stem(root / "infrared")
    depth = _files_by_stem(root / "depth")

    all_stems = set(visible) | set(infrared) | set(depth)
    complete_stems = set(visible) & set(infrared) & set(depth)
    incomplete = sorted(all_stems - complete_stems)
    if incomplete:
        preview = ", ".join(incomplete[:10])
        raise ValueError(f"Incomplete modality triplets ({len(incomplete)}): {preview}")

    label_dir = root / "labels"
    labels = {path.stem: path for path in label_dir.glob("*.txt")} if label_dir.is_dir() else {}
    if require_labels:
        missing_labels = sorted(complete_stems - set(labels))
        if missing_labels:
            preview = ", ".join(missing_labels[:10])
            raise ValueError(f"Missing label files ({len(missing_labels)}): {preview}")

    return [
        TripletSample(
            sample_id=stem,
            visible=visible[stem],
            infrared=infrared[stem],
            depth=depth[stem],
            label=labels.get(stem),
        )
        for stem in sorted(complete_stems)
    ]


def _visible_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _single_channel(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return np.rint(array[..., :3].astype(np.float32).mean(axis=2)).astype(array.dtype)
    raise ValueError(f"Unsupported image shape {array.shape} for {path}")


def _visible_luminance(rgb: np.ndarray) -> np.ndarray:
    values = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    return np.rint(values).clip(0, 255).astype(np.uint8)


def _infrared_u8(path: Path) -> np.ndarray:
    infrared = _single_channel(path).astype(np.float32)
    if infrared.size == 0:
        raise ValueError(f"Empty infrared image: {path}")
    if infrared.max(initial=0) <= 255:
        return np.rint(infrared).clip(0, 255).astype(np.uint8)
    maximum = float(infrared.max())
    return np.rint(infrared * (255.0 / maximum)).clip(0, 255).astype(np.uint8)


def _inverse_depth_u8(
    path: Path,
    *,
    min_valid_depth_mm: int,
    max_depth_mm: int,
) -> np.ndarray:
    depth = _single_channel(path)
    if depth.size == 0:
        raise ValueError(f"Empty depth image: {path}")

    # Official PNG depth maps are uint16 millimetres. The provided sample also
    # contains a JPG visualization; for an 8-bit depth image we preserve its
    # ordering and invert it so nearer values are brighter.
    if np.issubdtype(depth.dtype, np.integer) and depth.dtype.itemsize >= 2:
        depth_f = depth.astype(np.float32)
        valid = (depth_f >= float(min_valid_depth_mm)) & (depth_f <= float(max_depth_mm))
        clipped = np.clip(depth_f, float(min_valid_depth_mm), float(max_depth_mm))
        denominator = max(float(max_depth_mm - min_valid_depth_mm), 1.0)
        near = 1.0 - (clipped - float(min_valid_depth_mm)) / denominator
        result = np.rint(near * 255.0).clip(0, 255).astype(np.uint8)
        result[~valid] = 0
        return result

    depth_u8 = np.rint(depth.astype(np.float32)).clip(0, 255).astype(np.uint8)
    valid = depth_u8 > 0
    result = 255 - depth_u8
    result[~valid] = 0
    return result


def load_composite(
    sample: TripletSample,
    *,
    fusion: str = "yid",
    min_valid_depth_mm: int = 300,
    max_depth_mm: int = 20_000,
) -> np.ndarray:
    """Load a triplet as an HxWx3 uint8 model input.

    Supported modes:
      - yid: visible luminance, infrared intensity, inverse depth
      - rgb: visible image only (ablation)
      - infrared: replicated infrared only (ablation)
      - depth: replicated inverse depth only (ablation)
    """

    rgb = _visible_rgb(sample.visible)
    infrared = _infrared_u8(sample.infrared)
    inverse_depth = _inverse_depth_u8(
        sample.depth,
        min_valid_depth_mm=min_valid_depth_mm,
        max_depth_mm=max_depth_mm,
    )

    height, width = rgb.shape[:2]
    expected_shape = (height, width)
    if infrared.shape != expected_shape or inverse_depth.shape != expected_shape:
        raise ValueError(
            f"Spatial mismatch for {sample.sample_id}: visible={rgb.shape[:2]}, "
            f"infrared={infrared.shape}, depth={inverse_depth.shape}"
        )

    if fusion == "yid":
        return np.stack((_visible_luminance(rgb), infrared, inverse_depth), axis=2)
    if fusion == "rgb":
        return rgb
    if fusion == "infrared":
        return np.repeat(infrared[..., None], 3, axis=2)
    if fusion == "depth":
        return np.repeat(inverse_depth[..., None], 3, axis=2)
    raise ValueError(f"Unsupported fusion mode: {fusion}")
