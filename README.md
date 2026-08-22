# 面向城市场景的视觉多模态目标检测

本项目面向“全球校园人工智能算法精英大赛·面向城市场景的视觉多模态目标检测”赛题，提供：

- RGB、Infrared、Depth 三模态样本校验与对齐；
- 官方五字段 YOLO 标签校验；
- RF-DETR-Medium 和 RTMDet-m 的训练入口；
- 符合竞赛六字段格式的推理与 ZIP 提交文件生成；
- RGB-only、Infrared-only、Depth-only 消融模式。

## 1. 当前基线的三模态适配方式

两个官方模型都以三通道图像作为默认输入。为了在不破坏 COCO 预训练输入层的前提下先建立可复现基线，本项目将空间对齐的三模态编码为一个三通道复合图像：

```text
通道 0 = RGB 可见光的亮度 Y
通道 1 = Infrared 红外强度 I
通道 2 = 反向深度 D（有效像素越近越亮）
```

代码中称为 `yid` early-fusion。它能够同时使用三种模态，但不是最终的可学习特征融合网络。后续可以在此基线上增加三分支 stem、门控融合或中层特征融合，并与当前结果公平比较。

几何增强只作用于已经对齐的复合图像，因此三模态不会错位。RTMDet 配置特意移除了 HSV 和 MixUp：第二、第三通道不是颜色，使用彩色增强会破坏红外和深度语义。

## 2. 目录结构

```text
Object-Detection/
├─ configs/
│  └─ rtmdet_multimodal.py
├─ data/
│  ├─ raw/
│  │  └─ sample/
│  │     ├─ visible/
│  │     ├─ infrared/
│  │     ├─ depth/
│  │     └─ labels/
│  └─ prepared/
├─ scripts/
│  ├─ prepare_dataset.py
│  ├─ train_rfdetr.py
│  ├─ infer_rfdetr.py
│  ├─ train_rtmdet.py
│  └─ infer_rtmdet.py
├─ src/multimodal_detection/
├─ weights/
│  ├─ rfdetr/rf-detr-medium.pth
│  └─ rtmdet/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth
└─ work_dirs/
```

仓库通过 Git LFS 提供官方样例、样例的 `yid` 预处理结果以及两份固定版本的官方预训练权重。其他比赛数据、新权重、`work_dirs/` 和 `predictions/` 仍被忽略，避免意外提交完整数据集和训练产物。

## 3. 克隆后准备

本仓库包含约 775 MiB 的 Git LFS 资源。合作伙伴安装 Git LFS 后执行：

```powershell
git clone git@github.com:TreeWater-Liu/Object-Detection.git
cd Object-Detection
git lfs install
git lfs pull
git lfs ls-files
```

完成后，`data/raw/sample/`、`data/prepared/sample/` 和 `weights/` 已经可以直接使用，不需要再次复制样例或预训练权重。仓库中的数据只是官方样例，不是正式训练集；正式训练时仍需将官方完整数据按相同目录结构放到本机，再运行数据准备脚本。

## 4. 已下载资源

### 官方示例数据

示例数据已经解压到 `data/raw/sample/`：

- 18 组完整的 visible/infrared/depth/labels；
- 86 个标注框；
- 标签均为 `class_id center_x center_y width height` 五字段；
- PNG Depth 为单通道 `uint16` 毫米数据；
- 示例中的 JPG Depth 为三通道 `uint8` 可视化数据，代码同时兼容这两种形式。

原始 ZIP SHA256：

```text
075E52A4957A717D33FE4644B0B6CC39CECD1E26B1A08F06F51FC707D897CF15
```

### 官方预训练权重

```text
RF-DETR-Medium
路径: weights/rfdetr/rf-detr-medium.pth
大小: 404,992,918 bytes
MD5 : 7223F764A87B863F02EB8D52BF0CE2EE

RTMDet-m
路径: weights/rtmdet/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth
大小: 224,299,609 bytes
SHA256: 229F527CA88498E8894A778A62A878A322B4A3EA2CAE09EA537D34B7E907792B
```

比赛规则允许 ImageNet、COCO、Objects365 等公开预训练权重。RF-DETR 还包含 DINOv2 预训练先验，正式参赛前建议向组委会书面确认其属于允许的“等公开预训练权重”。

## 5. 准备数据

原始数据目录必须使用以下名称：

```text
RAW_ROOT/
├─ visible/
├─ infrared/
├─ depth/
└─ labels/
```

四个目录通过不含扩展名的文件名匹配。例如：

```text
visible/000016.png
infrared/000016.png
depth/000016.png
labels/000016.txt
```

安装公共依赖：

```powershell
python -m pip install -r requirements.txt
```

准备示例数据：

```powershell
python scripts/prepare_dataset.py `
  --source data/raw/sample `
  --output data/prepared/sample `
  --fusion yid `
  --val-ratio 0.2 `
  --seed 42
```

脚本会：

1. 检查三模态是否完整且空间尺寸一致；
2. 校验类别编号、标签字段和归一化坐标；
3. 生成 Y/IR/Inverse-Depth 复合 PNG；
4. 生成 RF-DETR 可读取的 YOLO 目录和 `data.yaml`；
5. 生成 RTMDet 可读取的 COCO `train.json`、`valid.json`；
6. 保存可复现划分、类别统计和三通道均值/标准差到 `manifest.json`。

目标目录必须是新目录或空目录，脚本不会静默覆盖已有数据。

可使用以下模式做消融：

```text
--fusion yid        三模态基线，默认
--fusion rgb        仅 RGB
--fusion infrared   仅红外
--fusion depth      仅深度
```

训练和推理必须使用相同的 `--fusion`。

## 6. RF-DETR-Medium

### 6.1 安装

建议为 RF-DETR 单独创建 Python 3.10 或 3.11 环境，并先安装与本机 CUDA 匹配的 PyTorch：

```powershell
python -m venv .venv-rfdetr
.\.venv-rfdetr\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-rfdetr.txt
```

### 6.2 训练

```powershell
python scripts/train_rfdetr.py `
  --dataset data/prepared/sample `
  --weights weights/rfdetr/rf-detr-medium.pth `
  --output work_dirs/rfdetr_medium `
  --epochs 100 `
  --batch-size 4 `
  --grad-accum-steps 4 `
  --device cuda
```

显存较小时加入：

```text
--batch-size 1 --grad-accum-steps 16 --gradient-checkpointing
```

常用输出：

```text
work_dirs/rfdetr_medium/checkpoint_best_total.pth
work_dirs/rfdetr_medium/checkpoint_best_ema.pth
work_dirs/rfdetr_medium/checkpoint_best_regular.pth
```

### 6.3 推理并生成提交文件

测试集不需要 `labels/`，但必须包含 `visible/infrared/depth/`。

```powershell
python scripts/infer_rfdetr.py `
  --source path/to/raw_test `
  --checkpoint work_dirs/rfdetr_medium/checkpoint_best_total.pth `
  --output predictions/rfdetr `
  --fusion yid `
  --score-threshold 0.001 `
  --max-detections 100 `
  --zip-output predictions/rfdetr_submission.zip
```

## 7. RTMDet-m

### 7.1 安装

MMCV 必须与 PyTorch/CUDA 匹配，建议使用独立环境：

```powershell
python -m venv .venv-rtmdet
.\.venv-rtmdet\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 先按本机 CUDA 安装官方 PyTorch，然后执行：
python -m pip install -r requirements-rtmdet.txt
mim install "mmcv>=2.0.0,<2.2.0"
```

### 7.2 训练

```powershell
python scripts/train_rtmdet.py `
  --dataset data/prepared/sample `
  --config configs/rtmdet_multimodal.py `
  --weights weights/rtmdet/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth `
  --output work_dirs/rtmdet_m `
  --epochs 100 `
  --batch-size 4 `
  --lr 0.0005 `
  --amp
```

训练脚本会读取 `manifest.json` 中的通道统计量，并写入：

```text
work_dirs/rtmdet_m/effective_config.py
```

推理时应使用这个有效配置，而不是原始模板，否则归一化参数可能不一致。

### 7.3 推理并生成提交文件

```powershell
python scripts/infer_rtmdet.py `
  --source path/to/raw_test `
  --config work_dirs/rtmdet_m/effective_config.py `
  --checkpoint work_dirs/rtmdet_m/best_coco_bbox_mAP_epoch_XX.pth `
  --output predictions/rtmdet `
  --fusion yid `
  --score-threshold 0.001 `
  --max-detections 100 `
  --device cuda:0 `
  --zip-output predictions/rtmdet_submission.zip
```

## 8. 训练输出是什么

RF-DETR 的 `--output work_dirs/rfdetr_medium` 通常包含：

```text
checkpoint_best_total.pth     综合指标最优权重，优先用于推理
checkpoint_best_ema.pth       EMA 权重
checkpoint_best_regular.pth   非 EMA 权重
checkpoint*.pth               周期性或最终检查点（具体名称随版本变化）
events.out.tfevents.*          TensorBoard 训练曲线
```

RTMDet 的 `--output work_dirs/rtmdet_m` 通常包含：

```text
effective_config.py                         本次训练的完整有效配置
epoch_*.pth                                 周期性检查点
best_coco_bbox_mAP_epoch_*.pth              验证集 mAP 最优权重
last_checkpoint                             最后检查点路径记录
*.log / vis_data/                           日志、指标和可视化数据
```

训练日志会持续打印分类损失、框回归损失、学习率、显存和 ETA；验证阶段会输出 COCO 风格的 `mAP、AP50、AP75` 等指标。最终推理输出不是模型权重，而是每张图对应的六字段 TXT，以及可选的提交 ZIP。

训练后的 `work_dirs/` 默认不会上传 GitHub，因为检查点体积很大且属于具体实验结果。合作伙伴应自行备份，或在明确选择某个最佳检查点后再通过 Git LFS/Release 单独发布。

## 9. CPU 与 GPU

两套模型理论上都能在 CPU 上执行部分流程，但训练面向 GPU 设计：

- GPU 能并行执行卷积、Transformer 注意力和矩阵运算，并使用 CUDA、混合精度与显存；
- CPU 训练通常慢一个到两个数量级，内存充足也无法代替 GPU 的并行吞吐；
- CPU 适合数据校验、数据准备、少量图片的功能性推理，不适合完整训练和大规模调参；
- GPU 显存不足时应降低 batch size、使用梯度累积、混合精度或梯度检查点，而不是直接改用 CPU 完整训练。

建议至少使用一张支持 CUDA 的 NVIDIA GPU。开始训练前执行：

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

只有 `torch.cuda.is_available()` 为 `True` 才应按 README 中的 CUDA 命令开始训练。

## 10. 推理输出保证

两个推理脚本都会：

- 为每个测试样本生成一个同名 TXT；
- 无检测结果时生成空 TXT；
- 按 confidence 从高到低排序；
- 最多保留 100 个合法框；
- 将坐标裁剪到图像范围；
- 输出：

```text
class_id norm_center_x norm_center_y norm_w norm_h confidence
```

- 可选生成只包含 TXT 文件的 ZIP 压缩包。

建议正式提交前统计：TXT 数量是否等于测试样本数、是否存在非法类别/坐标、是否有文件超过 100 行。

## 11. 推荐实验顺序

在固定训练/验证划分下依次运行：

1. `rgb`：RGB-only 基线；
2. `infrared` 和 `depth`：单模态贡献；
3. `yid`：三模态 early-fusion；
4. 比较 RF-DETR 与 RTMDet 的 `mAP@50-95、AP50、AP75、各类别 AP`；
5. 再开发可学习的中层融合结构。

竞赛禁止多个模型结果的简单投票、平均等集成。本项目的训练和推理入口每次只加载一个模型。
