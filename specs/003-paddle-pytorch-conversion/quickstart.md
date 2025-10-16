# Quick Start Guide: Weight Conversion Tool

**Feature**: 003-paddle-pytorch-conversion
**Date**: 2025-10-16
**Target Audience**: Developers and researchers migrating RT-DETRv3 models

## Overview

This guide helps you quickly convert RT-DETRv3 model weights from PaddlePaddle format to PyTorch format. The conversion process is automated and takes ~2 minutes for typical models.

## Prerequisites

### System Requirements
- **Python**: 3.8+ (3.11 recommended)
- **RAM**: 8GB minimum (16GB recommended for large models)
- **Disk Space**: 2x the size of source checkpoint

### Required Dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch>=2.0.0 torchvision

# Install PaddlePaddle (for loading source checkpoints)
pip install paddlepaddle>=2.4.0

# Install additional dependencies
pip install numpy>=1.21.0 pyyaml>=6.0
```

### Optional Dependencies

```bash
# For progress bars and better UX
pip install tqdm>=4.60.0

# For testing (if developing/contributing)
pip install pytest>=7.0.0 pytest-xdist
```

## Quick Start (5 Minutes)

### Step 1: Verify Installation

```bash
python -c "import torch; import paddle; print('PyTorch:', torch.__version__); print('PaddlePaddle:', paddle.__version__)"
```

**Expected Output**:
```
PyTorch: 2.1.0+cu118
PaddlePaddle: 2.5.1
```

---

### Step 2: Convert Your First Model

**Basic Conversion** (no validation):

```bash
python tools/convert_weights.py \
    --input pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams \
    --output pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth \
    --no-validate
```

**Conversion Time**: ~90 seconds for r50vd model (182MB)

**Output**: `pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth`

---

### Step 3: Verify Converted Weights

```python
import torch

# Load converted checkpoint
checkpoint = torch.load('pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth')

print(f"Parameters: {len(checkpoint['model'])}")
print(f"Conversion stats: {checkpoint['metadata']['conversion_stats']}")

# Verify a few parameters
for name, param in list(checkpoint['model'].items())[:5]:
    print(f"{name}: {param.shape}, {param.dtype}")
```

**Expected Output**:
```
Parameters: 315
Conversion stats: {'total': 315, 'converted': 312, 'skipped': 0, 'shape_mismatches': []}
backbone.conv1.weight: torch.Size([64, 3, 7, 7]), torch.float32
backbone.bn1.weight: torch.Size([64]), torch.float32
backbone.bn1.bias: torch.Size([64]), torch.float32
backbone.bn1.running_mean: torch.Size([64]), torch.float32
backbone.bn1.running_var: torch.Size([64]), torch.float32
```

---

## Common Use Cases

### Use Case 1: Convert with Validation

Validate that converted weights match your PyTorch model structure:

```bash
python tools/convert_weights.py \
    --input pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams \
    --output pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth \
    --model-config configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml
```

**When to use**: First-time conversion, verifying compatibility

**Benefit**: Catches shape mismatches and missing parameters early

---

### Use Case 2: Convert with Custom Mapping

For models with custom layers or non-standard naming:

```bash
# Create manual mapping file
cat > my_mapping.json << EOF
{
  "version": "1.0",
  "mappings": {
    "backbone.custom_layer.w_0": "backbone.custom_layer.weight",
    "neck.special_module._param": "neck.special_module.param"
  }
}
EOF

# Convert with manual mapping
python tools/convert_weights.py \
    --input pretrained_models/paddle/my_custom_model.pdparams \
    --output pretrained_models/pytorch/my_custom_model.pth \
    --model-config configs/my_custom_model.yml \
    --manual-mapping my_mapping.json \
    --save-mapping outputs/final_mapping.json
```

**When to use**: Custom models, non-standard architectures

**Benefit**: Full control over parameter name mapping

---

### Use Case 3: Batch Convert Multiple Models

Convert all available model variants at once:

```bash
python tools/convert_weights.py \
    --batch "pretrained_models/paddle/rtdetrv3_*.pdparams" \
    --output-dir pretrained_models/pytorch/ \
    --model-config configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml
```

**When to use**: Converting multiple model variants (r18, r34, r50)

**Benefit**: Automated batch processing, time-saving

---

### Use Case 4: Strict Mode (Fail-Fast)

Fail immediately on any conversion issue:

```bash
python tools/convert_weights.py \
    --input pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams \
    --output pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth \
    --model-config configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml \
    --strict
```

**When to use**: Production deployments, CI/CD pipelines

**Benefit**: Ensures 100% conversion success or fail

---

### Use Case 5: Numerical Validation

Verify converted values match source (takes longer):

```bash
python tools/convert_weights.py \
    --input pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams \
    --output pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth \
    --model-config configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml \
    --validate-numerical \
    --tolerance 1e-5
```

**When to use**: Critical applications, research reproducibility

**Benefit**: Guarantees numerical equivalence (within tolerance)

---

## Loading Converted Weights

### Option 1: Load into PyTorch Model

```python
import torch
from models.rtdetrv3 import RTDETRv3

# Initialize model
model = RTDETRv3(config)

# Load converted checkpoint
checkpoint = torch.load('pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth')
model.load_state_dict(checkpoint['model'], strict=True)

# Set to evaluation mode
model.eval()

print("Model loaded successfully!")
```

---

### Option 2: Load with Custom Missing/Unexpected Handling

```python
import torch
from models.rtdetrv3 import RTDETRv3

model = RTDETRv3(config)
checkpoint = torch.load('pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth')

# Load with non-strict mode (allows missing/unexpected keys)
missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint['model'],
    strict=False
)

if missing_keys:
    print(f"Warning: Missing keys: {missing_keys}")
if unexpected_keys:
    print(f"Warning: Unexpected keys: {unexpected_keys}")
```

---

### Option 3: Inspect Checkpoint Before Loading

```python
import torch

# Load checkpoint
checkpoint = torch.load('pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth')

# Inspect metadata
print("Source:", checkpoint['metadata']['source'])
print("Conversion timestamp:", checkpoint['metadata']['conversion_timestamp'])
print("Conversion stats:", checkpoint['metadata']['conversion_stats'])

# List all parameters
print("\nParameters:")
for name in checkpoint['model'].keys():
    param = checkpoint['model'][name]
    print(f"  {name}: {param.shape}, {param.dtype}")
```

---

## Troubleshooting

### Problem 1: Shape Mismatch Error

**Error Message**:
```
ERROR: Shape mismatch for parameter 'backbone.layer.weight'
  Source shape: (256, 128, 3, 3)
  Target shape: (256, 64, 3, 3)
```

**Solution**:
1. Verify you're using the correct model config file
2. Check if PaddlePaddle and PyTorch model architectures match
3. Use `--permissive` mode to skip problematic parameters:
   ```bash
   python tools/convert_weights.py --input input.pdparams --output output.pth --permissive
   ```

---

### Problem 2: Unmapped Parameters

**Warning Message**:
```
WARNING: Unmapped source parameter: backbone.custom_layer.extra_param
WARNING: Unmapped target parameter: backbone.new_module.weight
```

**Solution**:
1. Export mapping to inspect unmapped parameters:
   ```bash
   python tools/convert_weights.py ... --save-mapping mapping.json
   ```
2. Create manual mapping file for unmapped parameters:
   ```json
   {
     "version": "1.0",
     "mappings": {
       "backbone.custom_layer.extra_param": "backbone.custom_layer.extra_param_torch"
     }
   }
   ```
3. Retry conversion with manual mapping:
   ```bash
   python tools/convert_weights.py ... --manual-mapping my_mapping.json
   ```

---

### Problem 3: Out of Memory

**Error Message**:
```
RuntimeError: CUDA out of memory. Tried to allocate 1.50 GiB
```

**Solution**:
1. Use memory-efficient mode:
   ```bash
   python tools/convert_weights.py ... --memory-efficient
   ```
2. Close other programs to free RAM
3. Use CPU-only conversion (slower but lower memory):
   ```bash
   export CUDA_VISIBLE_DEVICES=""  # Disable GPU
   python tools/convert_weights.py ...
   ```

---

### Problem 4: PaddlePaddle Not Installed

**Error Message**:
```
ImportError: No module named 'paddle'
```

**Solution**:
```bash
# CPU version
pip install paddlepaddle

# GPU version (CUDA 11.8)
pip install paddlepaddle-gpu==2.5.1 -f https://paddlepaddle.org.cn/whl/linux/cudnn/stable.html
```

---

### Problem 5: Numerical Validation Failed

**Error Message**:
```
ERROR: Numerical validation FAILED
  Max absolute difference: 2.3e-4
  Tolerance: 1e-5
```

**Solution**:
1. Increase tolerance if small differences are acceptable:
   ```bash
   python tools/convert_weights.py ... --validate-numerical --tolerance 1e-4
   ```
2. Check if PaddlePaddle and PyTorch models are truly equivalent
3. Verify random seed handling for stochastic layers

---

## Performance Tips

### Tip 1: Batch Processing
Convert multiple models in one command:
```bash
python tools/convert_weights.py --batch "pretrained_models/paddle/*.pdparams" --output-dir pytorch/
```
**Speed-up**: ~3x faster than individual conversions (shared initialization overhead)

---

### Tip 2: Skip Validation for Fast Conversion
If you trust the conversion tool:
```bash
python tools/convert_weights.py ... --no-validate
```
**Speed-up**: ~20% faster (skips shape validation)

---

### Tip 3: Use SSD for I/O-Bound Operations
Store checkpoints on SSD (not HDD) for faster loading/saving.

**Speed-up**: ~2x faster for large models (>500MB)

---

### Tip 4: Parallel Batch Conversion
Convert multiple models in parallel using GNU parallel:
```bash
ls pretrained_models/paddle/*.pdparams | \
parallel -j4 python tools/convert_weights.py --input {} --output pytorch/{/} --no-validate
```
**Speed-up**: ~4x faster on multi-core systems

---

## Next Steps

### For Researchers
1. **Validate Inference**: Run inference on COCO validation set with converted weights
2. **Compare Results**: Verify predictions match PaddlePaddle version (mAP, latency)
3. **Document Differences**: Note any numerical differences for reproducibility

**Example Validation Script**:
```python
import torch
from models.rtdetrv3 import RTDETRv3
from pycocotools.coco import COCO

# Load model
model = RTDETRv3(config)
checkpoint = torch.load('pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth')
model.load_state_dict(checkpoint['model'])
model.eval()

# Run inference on COCO val
coco = COCO('data/coco/annotations/instances_val2017.json')
# ... (inference code)

print(f"mAP: {map_score:.4f}")
```

---

### For Developers
1. **Integrate into Pipeline**: Add conversion step to training/evaluation workflows
2. **Automate Testing**: Create unit tests for custom models
3. **CI/CD Integration**: Add conversion validation to continuous integration

**Example CI Script** (`.github/workflows/test_conversion.yml`):
```yaml
name: Test Weight Conversion
on: [push, pull_request]
jobs:
  test-conversion:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install torch paddlepaddle pytest
      - name: Convert weights
        run: |
          python tools/convert_weights.py \
            --input tests/fixtures/sample.pdparams \
            --output tests/outputs/sample.pth \
            --strict --validate-numerical
      - name: Run tests
        run: pytest tests/test_weight_conversion/
```

---

## Resources

### Documentation
- **Feature Specification**: `specs/003-paddle-pytorch-conversion/spec.md`
- **Implementation Plan**: `specs/003-paddle-pytorch-conversion/plan.md`
- **Data Model**: `specs/003-paddle-pytorch-conversion/data-model.md`
- **CLI Contract**: `specs/003-paddle-pytorch-conversion/contracts/cli-interface.md`

### Related Tools
- **Model Zoo**: `pretrained_models/` (source PaddlePaddle checkpoints)
- **Test Suite**: `tests/test_weight_conversion/` (validation tests)
- **Example Configs**: `configs/rtdetrv3/` (model configuration files)

### External References
- **PyTorch State Dict**: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html
- **PaddlePaddle Model I/O**: https://www.paddlepaddle.org.cn/documentation/docs/en/guides/model_convert/save_load_en.html
- **RT-DETRv3 Paper**: https://arxiv.org/abs/2407.17140

---

## FAQ

### Q: Can I convert weights without a PyTorch model?
**A**: Yes, use `--no-validate` mode. However, you won't have shape validation, so ensure your PyTorch model matches the PaddlePaddle architecture.

---

### Q: Are optimizer states converted?
**A**: No, only model weights are converted. This is a design choice (see specification scope).

---

### Q: Can I use converted weights for training?
**A**: Yes, converted weights are suitable for both inference and fine-tuning. For full training from scratch, you may want to train natively in PyTorch.

---

### Q: What if my model has custom operators?
**A**: Custom operators require manual porting to PyTorch. The conversion tool only handles standard layers. See `tech-report.md` for RT-DETRv3-specific operator mappings.

---

### Q: How do I verify conversion correctness?
**A**: Use `--validate-numerical` flag for automated validation, or run inference on test images and compare predictions with PaddlePaddle outputs.

---

**Quick Start Complete**: You're ready to convert RT-DETRv3 weights! For advanced usage, see the full CLI contract documentation.
