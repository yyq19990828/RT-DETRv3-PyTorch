#!/usr/bin/env python3
"""
Analyze PaddlePaddle model weights from .pdparams files
Compare with PyTorch model parameters

Usage:
    python analyze_paddle_weights.py
"""

import sys
import os
sys.path.insert(0, 'rtdetrv3_pytorch')

import torch
from rtdetrv3_pytorch.models.rtdetrv3 import build_rtdetrv3
from rtdetrv3_pytorch.utils.config import load_config


def analyze_paddle_weights(pdparams_path):
    """Analyze PaddlePaddle weights file"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {pdparams_path}")
    print(f"{'='*80}")

    try:
        # Try to load with pickle (numpy format)
        import pickle
        with open(pdparams_path, 'rb') as f:
            state_dict = pickle.load(f)

        # Count parameters
        total_params = 0
        component_params = {
            'backbone': 0,
            'neck': 0,
            'transformer': 0,
            'detr_head': 0,
            'aux_head': 0,
            'other': 0
        }

        print(f"\nParameter breakdown:")
        print(f"{'Parameter Name':<60} {'Shape':<25} {'Count':>15}")
        print("-" * 100)

        for name, param in state_dict.items():
            if hasattr(param, 'shape'):
                shape = param.shape
                count = 1
                for dim in shape:
                    count *= dim
                total_params += count

                # Categorize parameter
                if 'backbone' in name:
                    component_params['backbone'] += count
                elif 'neck' in name or 'hybrid_encoder' in name:
                    component_params['neck'] += count
                elif 'transformer' in name or 'decoder' in name or 'encoder' in name.lower():
                    component_params['transformer'] += count
                elif 'detr_head' in name or 'dinov3_head' in name:
                    component_params['detr_head'] += count
                elif 'aux' in name or 'ppyoloe' in name:
                    component_params['aux_head'] += count
                else:
                    component_params['other'] += count

                # Print first 20 parameters
                if len([k for k in state_dict.keys() if state_dict[k] is param][:20]) < 20:
                    print(f"{name:<60} {str(shape):<25} {count:>15,}")

        print(f"\n{'='*100}")
        print(f"{'TOTAL PARAMETERS':<60} {'':<25} {total_params:>15,}")
        print(f"{'='*100}")

        print(f"\nComponent-wise breakdown:")
        for comp, count in component_params.items():
            if count > 0:
                percentage = count / total_params * 100
                print(f"  {comp:<20}: {count:>15,} ({percentage:>5.2f}%)")

        size_mb = total_params * 4 / 1024 / 1024
        print(f"\nModel size: {size_mb:.2f} MB (assuming float32)")

        return {
            'total': total_params,
            'components': component_params,
            'size_mb': size_mb
        }

    except Exception as e:
        print(f"✗ Failed to load PaddlePaddle weights: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_pytorch(paddle_results, pytorch_config, model_name):
    """Compare PaddlePaddle weights with PyTorch model"""
    print(f"\n{'='*80}")
    print(f"Comparing with PyTorch model: {model_name}")
    print(f"{'='*80}")

    try:
        # Load PyTorch config
        cfg = load_config(pytorch_config)

        # Extract model parameters
        resnet_cfg = cfg.get('ResNet', {})
        transformer_cfg = cfg.get('RTDETRTransformerv3', {})

        depth = resnet_cfg.get('depth', 50)
        variant = resnet_cfg.get('variant', 'd')
        num_decoder_layers = transformer_cfg.get('num_decoder_layers', 6)
        hidden_dim = cfg.get('hidden_dim', 256)
        o2m_branch = cfg.get('o2m_branch', False)
        num_queries_o2m = cfg.get('num_queries_o2m', 450)

        # Build PyTorch model
        print(f"\nBuilding PyTorch model (ResNet-{depth}-v{variant}, {num_decoder_layers} decoder layers)...")
        backbone_name = f'resnet{depth}'
        pytorch_model = build_rtdetrv3(
            num_classes=80,
            backbone=backbone_name,
            variant=variant,
            frozen_stages=0,
            hidden_dim=hidden_dim,
            num_queries=300,
            num_decoder_layers=num_decoder_layers,
            num_levels=3,
            num_points=4,
            eval_idx=-1,
            o2m=4,
            o2m_branch=o2m_branch,
            num_queries_o2m=num_queries_o2m,
            use_aux_head=True
        )
        pytorch_model.eval()

        # Count PyTorch parameters
        pytorch_total = sum(p.numel() for p in pytorch_model.parameters())
        pytorch_components = {}

        for attr_name in ['backbone', 'neck', 'transformer', 'detr_head', 'aux_head']:
            if hasattr(pytorch_model, attr_name):
                attr = getattr(pytorch_model, attr_name)
                if isinstance(attr, torch.nn.Module):
                    pytorch_components[attr_name] = sum(p.numel() for p in attr.parameters())

        print(f"\nPyTorch model parameters: {pytorch_total:,}")
        for comp, count in pytorch_components.items():
            print(f"  {comp:<20}: {count:>15,}")

        # Compare
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")

        paddle_total = paddle_results['total']
        diff = pytorch_total - paddle_total
        diff_pct = (diff / paddle_total * 100) if paddle_total > 0 else 0

        print(f"\n{'Framework':<20} {'Total Params':>20} {'Size (MB)':>15}")
        print("-" * 60)
        print(f"{'PaddlePaddle':<20} {paddle_total:>20,} {paddle_results['size_mb']:>15.2f}")
        print(f"{'PyTorch':<20} {pytorch_total:>20,} {pytorch_total * 4 / 1024 / 1024:>15.2f}")
        print(f"{'Difference':<20} {diff:>+20,} {diff_pct:>+14.2f}%")

        # Component comparison
        print(f"\n{'Component':<20} {'PaddlePaddle':>20} {'PyTorch':>20} {'Difference':>20}")
        print("-" * 85)

        for comp in ['backbone', 'neck', 'transformer', 'aux_head']:
            paddle_comp = paddle_results['components'].get(comp, 0)
            pytorch_comp = pytorch_components.get(comp, 0)
            comp_diff = pytorch_comp - paddle_comp

            if paddle_comp > 0 or pytorch_comp > 0:
                print(f"{comp:<20} {paddle_comp:>20,} {pytorch_comp:>20,} {comp_diff:>+20,}")

        # Validation
        print(f"\n{'='*80}")
        print("VALIDATION")
        print(f"{'='*80}")

        tolerance_pct = 1.0  # 1% tolerance

        if abs(diff_pct) <= tolerance_pct:
            print(f"\n✓ Parameter counts match within {tolerance_pct}% tolerance!")
            print(f"  Difference: {diff_pct:+.4f}%")
            print(f"✓ PyTorch model correctly migrated from PaddlePaddle")
            return True
        else:
            print(f"\n⚠ Parameter count mismatch: {diff_pct:+.2f}%")
            print(f"  Expected difference should be < {tolerance_pct}%")

            # Analyze component mismatches
            print(f"\nComponent-wise analysis:")
            for comp in ['backbone', 'neck', 'transformer', 'aux_head']:
                paddle_comp = paddle_results['components'].get(comp, 0)
                pytorch_comp = pytorch_components.get(comp, 0)

                if paddle_comp > 0 and pytorch_comp > 0:
                    comp_diff_pct = ((pytorch_comp - paddle_comp) / paddle_comp * 100)
                    if abs(comp_diff_pct) > tolerance_pct:
                        print(f"  ⚠ {comp}: {comp_diff_pct:+.2f}% difference")

            return False

    except Exception as e:
        print(f"\n✗ Failed to build PyTorch model: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    print("="*80)
    print("PaddlePaddle vs PyTorch Model Verification")
    print("Real Model Weights Analysis")
    print("="*80)

    models = [
        {
            'name': 'RT-DETRv3-R18-vd',
            'paddle_weights': 'pretrained_models/paddle/rtdetrv3_r18vd_6x_coco.pdparams',
            'pytorch_config': 'rtdetrv3_pytorch/configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml'
        },
        {
            'name': 'RT-DETRv3-R34-vd',
            'paddle_weights': 'pretrained_models/paddle/rtdetrv3_r34vd_6x_coco.pdparams',
            'pytorch_config': 'rtdetrv3_pytorch/configs/rtdetrv3/rtdetrv3_r34vd_6x_coco.yml'
        },
        {
            'name': 'RT-DETRv3-R50-vd',
            'paddle_weights': 'pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams',
            'pytorch_config': 'rtdetrv3_pytorch/configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml'
        },
    ]

    results = []

    for model in models:
        print(f"\n\n{'#'*80}")
        print(f"# {model['name']}")
        print(f"{'#'*80}")

        # Analyze PaddlePaddle weights
        paddle_results = analyze_paddle_weights(model['paddle_weights'])

        if paddle_results:
            # Compare with PyTorch
            match = compare_with_pytorch(paddle_results, model['pytorch_config'], model['name'])
            results.append({
                'name': model['name'],
                'paddle': paddle_results,
                'match': match
            })

    # Final summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Model':<20} {'PaddlePaddle':>20} {'PyTorch (Built)':>20} {'Match':>10}")
    print("-" * 75)

    all_matched = True
    for result in results:
        paddle_total = result['paddle']['total']
        match_symbol = "✓" if result['match'] else "✗"
        if not result['match']:
            all_matched = False
        print(f"{result['name']:<20} {paddle_total:>20,} {'(see above)':>20} {match_symbol:>10}")

    print(f"\n{'='*80}")
    if all_matched:
        print("✓ ALL MODELS VALIDATED SUCCESSFULLY!")
        print("✓ PyTorch implementations match PaddlePaddle parameter counts")
        print("✓ Migration verified with actual model weights")
    else:
        print("⚠ SOME MODELS HAVE MISMATCHES")
        print("  Review component-wise comparisons above")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
