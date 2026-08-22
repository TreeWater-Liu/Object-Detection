"""RTMDet-m configuration for the three-channel Y/IR/inverse-depth composite."""

_base_ = "mmdet::rtmdet/rtmdet_m_8xb32-300e_coco.py"

class_names = (
    "person",
    "boat",
    "animal",
    "seat",
    "sign",
    "bicycle",
    "car",
    "ball",
    "light",
    "garbage can",
    "uav",
    "tricycle",
)
metainfo = dict(classes=class_names)

data_root = "data/prepared/sample/"
input_size = (1024, 1024)

# Modality-aware baseline: avoid HSV and MixUp because the second and third
# channels represent infrared and inverse depth rather than colour channels.
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="RandomResize",
        scale=input_size,
        ratio_range=(0.8, 1.2),
        keep_ratio=True,
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Pad", size=input_size, pad_val=dict(img=(0, 0, 0))),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=input_size, keep_ratio=True),
    dict(type="Pad", size=input_size, pad_val=dict(img=(0, 0, 0))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

model = dict(
    data_preprocessor=dict(
        mean=[127.5, 127.5, 127.5],
        std=[64.0, 64.0, 64.0],
        # MMDetection decodes files as BGR; convert back to the saved
        # Y/IR/inverse-depth channel order before normalization.
        bgr_to_rgb=True,
    ),
    bbox_head=dict(num_classes=12),
    test_cfg=dict(score_thr=0.001, max_per_img=100),
)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="annotations/train.json",
        data_prefix=dict(img="train/images/"),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="annotations/valid.json",
        data_prefix=dict(img="valid/images/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/valid.json",
    metric="bbox",
    format_only=False,
    proposal_nums=(100, 1, 10),
)
test_evaluator = val_evaluator

max_epochs = 100
base_lr = 5e-4
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
    clip_grad=dict(max_norm=35, norm_type=2),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, by_epoch=False, begin=0, end=100),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
    ),
]
custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    )
]
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=3,
        save_best="coco/bbox_mAP",
        rule="greater",
    )
)
