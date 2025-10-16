# RT-DETRv3 PyTorch Implementation

PyTorch implementation of RT-DETRv3 (Real-Time Detection Transformer v3), migrated from the official PaddlePaddle implementation.

## Features

- ✅ Complete RT-DETRv3 architecture (ResNet backbone + HybridEncoder + Transformer)
- ✅ Multi-scale deformable attention
- ✅ One-to-one and one-to-many matching
- ✅ PaddlePaddle checkpoint conversion utility
- ✅ Training and inference pipelines
- ✅ ONNX and TorchScript export support

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.5.1
- CUDA >= 11.8 or >= 12.1

### Install with uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd rtdetrv3_pytorch

# Install dependencies with uv
uv sync

# Or install manually
pip install -e .
```

### CUDA Support

For CUDA 12.1:
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Inference

```python
from rtdetrv3_pytorch.models import RTDETRv3
import torch

# Load model
model = RTDETRv3(config)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# Run inference
image = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    outputs = model(image)
```

### Training

```bash
# Single GPU
python tools/train.py -c configs/rtdetrv3_r50_6x_coco.yml

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetrv3_r50_6x_coco.yml --distributed
```

### Evaluation

```bash
python tools/eval.py -c configs/rtdetrv3_r50_6x_coco.yml --checkpoint checkpoints/best.pth
```

### Weight Conversion

Convert PaddlePaddle checkpoints to PyTorch format:

```bash
python tools/convert_weights.py \
    --paddle_checkpoint path/to/paddle.pdparams \
    --config configs/rtdetrv3_r50_6x_coco.yml \
    --output converted.pth
```

## Model Zoo

| Model | Backbone | mAP (COCO val2017) | FPS (T4) | Checkpoint |
|-------|----------|-------------------|----------|------------|
| RT-DETRv3-R18 | ResNet-18 | 48.1% | 217 | [download](#) |
| RT-DETRv3-R50 | ResNet-50 | 53.4% | 108 | [download](#) |
| RT-DETRv3-R101 | ResNet-101 | TBD | TBD | [download](#) |

## Project Structure

```
rtdetrv3_pytorch/
├── models/              # Model components
│   ├── backbones/      # ResNet variants
│   ├── necks/          # HybridEncoder (FPN-PAN)
│   ├── transformers/   # Transformer encoder/decoder
│   ├── heads/          # Detection heads
│   └── losses/         # Loss functions
├── data/               # Dataset loaders and transforms
├── engine/             # Training and evaluation logic
├── utils/              # Utilities (config, logger, distributed)
├── tools/              # Scripts (train, eval, export, convert)
├── configs/            # Configuration files
└── tests/              # Unit and integration tests
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{rtdetrv3,
  title={RT-DETRv3: Real-Time End-to-End Object Detection with Transformers},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This project is released under the Apache 2.0 license.

## Acknowledgments

- Original PaddlePaddle implementation: [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- Deformable attention implementation
- PyTorch team for excellent framework support
