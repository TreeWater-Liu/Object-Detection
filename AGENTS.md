# Agent execution guide

This repository contains a reproducible sample workflow for the multimodal
object-detection competition. Work from the repository root and read
`README.md` before changing model or data settings.

## Assets and data scope

- Run `git lfs install` and `git lfs pull` after cloning.
- `data/raw/sample` is the official **sample**, not the full competition set.
- `data/prepared/sample` is the reproducible `yid` preparation of that sample.
- The two files under `weights/` are official pretrained weights, not
  competition-fine-tuned checkpoints.
- Never claim a sample run is a full competition training result.

## Environment rules

- Use separate Python 3.10 or 3.11 virtual environments for RF-DETR and
  RTMDet. Do not install both stacks into one environment unless compatibility
  has been verified.
- Install a PyTorch build matching the host NVIDIA driver/CUDA before the
  model-specific requirements.
- Verify the GPU with `nvidia-smi` and
  `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
- Do not silently fall back to CPU for a requested training run. Report the
  missing CUDA capability instead.

## Reproducible smoke training

RF-DETR:

```powershell
python scripts/train_rfdetr.py `
  --dataset data/prepared/sample `
  --weights weights/rfdetr/rf-detr-medium.pth `
  --output work_dirs/rfdetr_medium `
  --epochs 1 --batch-size 1 --grad-accum-steps 1 --device cuda
```

RTMDet:

```powershell
python scripts/train_rtmdet.py `
  --dataset data/prepared/sample `
  --weights weights/rtmdet/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth `
  --output work_dirs/rtmdet_m `
  --epochs 1 --batch-size 1 --num-workers 0 --amp
```

After smoke tests pass, select realistic epochs, batch size, accumulation, and
workers for the host GPU. Preserve the training/preparation fusion mode for
inference.

## Required verification

- Confirm checkpoints were written under `work_dirs/`.
- Use RF-DETR's `checkpoint_best_total.pth` or the selected best checkpoint.
- Use RTMDet's `effective_config.py` together with the matching best `.pth`.
- Run inference on `data/raw/sample` and verify that one TXT file is generated
  for every source triplet and that the submission ZIP contains only TXT files.
- Do not commit `work_dirs/`, `predictions/`, private datasets, or newly trained
  checkpoints without explicit approval.
