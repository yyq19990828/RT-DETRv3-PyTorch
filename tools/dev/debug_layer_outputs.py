#!/usr/bin/env python
"""Debug layer-by-layer outputs to find where differences start"""

import sys
import torch
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "third-party" / "RT-DETRv3-paddle"))

def load_paddle_model():
    """Load Paddle model"""
    import paddle
    from ppdet.core.workspace import load_config, create

    cfg = load_config(str(project_root / "third-party/RT-DETRv3-paddle/configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml"))
    model = create(cfg.architecture)

    # Load weights
    state = paddle.load(str(project_root / "pretrained_models/paddle/rtdetrv3_r18vd_6x_coco.pdparams"))
    model.set_state_dict(state)
    model.eval()

    return model

def load_pytorch_model():
    """Load PyTorch model"""
    from ppdet_pytorch.core.workspace import load_config, create

    cfg = load_config(str(project_root / "configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml"))
    model = create(cfg.architecture)

    # Load weights
    checkpoint = torch.load(
        str(project_root / "pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth"),
        map_location='cpu'
    )
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    return model

def compare_layer_output(paddle_out, torch_out, layer_name):
    """Compare outputs from Paddle and PyTorch"""
    if isinstance(paddle_out, (list, tuple)):
        print(f"\n{layer_name} (list/tuple with {len(paddle_out)} elements):")
        for i, (p, t) in enumerate(zip(paddle_out, torch_out)):
            compare_layer_output(p, t, f"{layer_name}[{i}]")
        return

    if isinstance(paddle_out, dict):
        print(f"\n{layer_name} (dict with keys: {list(paddle_out.keys())}):")
        for key in paddle_out.keys():
            compare_layer_output(paddle_out[key], torch_out[key], f"{layer_name}.{key}")
        return

    # Convert to numpy
    import paddle
    if isinstance(paddle_out, paddle.Tensor):
        paddle_np = paddle_out.numpy()
    else:
        paddle_np = paddle_out

    if isinstance(torch_out, torch.Tensor):
        torch_np = torch_out.detach().cpu().numpy()
    else:
        torch_np = torch_out

    # Compare
    abs_diff = np.abs(paddle_np - torch_np)
    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()

    # Relative difference
    mask = np.abs(paddle_np) > 1e-7
    if mask.any():
        rel_diff = abs_diff[mask] / (np.abs(paddle_np[mask]) + 1e-10)
        max_rel_diff = rel_diff.max()
    else:
        max_rel_diff = 0.0

    status = "✅" if max_abs_diff < 1e-3 else "❌"
    print(f"{status} {layer_name}: shape={paddle_np.shape}, max_abs={max_abs_diff:.2e}, mean_abs={mean_abs_diff:.2e}, max_rel={max_rel_diff:.2e}")

def main():
    print("Loading models...")
    import paddle
    paddle_model = load_paddle_model()
    torch_model = load_pytorch_model()

    # Create same input
    np.random.seed(42)
    input_np = np.random.randn(1, 3, 640, 640).astype(np.float32)

    paddle_input = paddle.to_tensor(input_np)
    torch_input = torch.from_numpy(input_np)

    print("\nComparing layer outputs...")
    print("="*80)

    # Backbone
    with paddle.no_grad():
        paddle_backbone_out = paddle_model.backbone({'image': paddle_input})
    with torch.no_grad():
        torch_backbone_out = torch_model.backbone(torch_input)

    compare_layer_output(paddle_backbone_out, torch_backbone_out, "backbone")

    # Neck
    with paddle.no_grad():
        paddle_neck_out = paddle_model.neck(paddle_backbone_out)
    with torch.no_grad():
        torch_neck_out = torch_model.neck(torch_backbone_out)

    compare_layer_output(paddle_neck_out, torch_neck_out, "neck")

    # Transformer input_proj
    with paddle.no_grad():
        paddle_input_proj_out = [proj(feat) for proj, feat in zip(paddle_model.transformer.input_proj, paddle_neck_out)]
    with torch.no_grad():
        torch_input_proj_out = [proj(feat) for proj, feat in zip(torch_model.transformer.input_proj, torch_neck_out)]

    compare_layer_output(paddle_input_proj_out, torch_input_proj_out, "transformer.input_proj")

    # Check first transformer decoder layer self_attn
    print("\n\nTesting first decoder layer self_attn...")

    # Prepare input for decoder layer (needs proper format)
    # For now, just check if the weight loading is correct by comparing model parameters
    print("\nChecking parameter alignment:")
    paddle_param = paddle_model.transformer.decoder.layers[0].self_attn.in_proj_weight
    torch_param = torch_model.transformer.decoder.layers[0].self_attn.in_proj_weight

    paddle_np = paddle_param.numpy()
    torch_np = torch_param.detach().cpu().numpy()

    abs_diff = np.abs(paddle_np - torch_np)
    print(f"in_proj_weight: paddle shape={paddle_np.shape}, torch shape={torch_np.shape}")
    print(f"  max_abs_diff={abs_diff.max():.2e}, mean_abs_diff={abs_diff.mean():.2e}")

    # Check if they're transposed
    if paddle_np.shape != torch_np.shape:
        print(f"  ⚠️  Shape mismatch!")
        torch_np_t = torch_np.T
        abs_diff_t = np.abs(paddle_np - torch_np_t)
        print(f"  After transpose: max_abs_diff={abs_diff_t.max():.2e}")

    print("\n" + "="*80)
    print("Layer comparison complete")

if __name__ == "__main__":
    main()
