"""
Compare parameter names between PaddlePaddle and PyTorch RT-DETRv3 models

This script creates both Paddle and PyTorch models and compares their parameter
naming conventions to help validate weight conversion.
"""

import sys
import argparse
from pathlib import Path

# Get project root (parent of tools directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_paddle_model(config_path):
    """Load PaddlePaddle model from config"""
    import paddle

    # Add PaddleDetection to path (in RT-DETRv3-paddle directory)
    paddle_codebase = project_root / 'RT-DETRv3-paddle'
    if not paddle_codebase.exists():
        raise FileNotFoundError(f"Paddle codebase not found at: {paddle_codebase}")

    if str(paddle_codebase) not in sys.path:
        sys.path.insert(0, str(paddle_codebase))

    from ppdet.core.workspace import load_config, create

    print(f"Loading Paddle config: {config_path}")
    cfg = load_config(config_path)

    print("Building PaddlePaddle model...")
    model = create(cfg.architecture)

    return model


def load_pytorch_model(config_path):
    """Load PyTorch model from config"""
    import torch

    # Add PyTorch codebase to path
    pytorch_codebase = project_root / 'rtdetrv3_pytorch'
    if str(pytorch_codebase) not in sys.path:
        sys.path.insert(0, str(pytorch_codebase))

    from ppdet_pytorch.core.workspace import load_config, create

    print(f"Loading PyTorch config: {config_path}")
    cfg = load_config(config_path)

    print("Building PyTorch model...")
    model = create(cfg.architecture)

    return model


def get_model_parameters(model, framework='paddle'):
    """Extract parameter names and shapes from model"""
    params = []
    buffers = []

    if framework == 'paddle':
        for name, param in model.named_parameters():
            params.append((name, list(param.shape)))
        for name, buf in model.named_buffers():
            buffers.append((name, list(buf.shape)))
    else:  # pytorch
        for name, param in model.named_parameters():
            params.append((name, list(param.shape)))
        for name, buf in model.named_buffers():
            buffers.append((name, list(buf.shape)))

    return params, buffers


def compare_parameters(paddle_params, torch_params, paddle_buffers, torch_buffers):
    """Compare parameter names between Paddle and PyTorch"""

    print("\n" + "="*80)
    print("PARAMETER COMPARISON SUMMARY")
    print("="*80)

    print(f"\nPaddle: {len(paddle_params)} parameters, {len(paddle_buffers)} buffers")
    print(f"PyTorch: {len(torch_params)} parameters, {len(torch_buffers)} buffers")

    # Create sets for comparison
    paddle_param_names = {name for name, _ in paddle_params}
    torch_param_names = {name for name, _ in torch_params}
    paddle_buffer_names = {name for name, _ in paddle_buffers}
    torch_buffer_names = {name for name, _ in torch_buffers}

    # Find matches and differences
    exact_matches = paddle_param_names & torch_param_names
    only_in_paddle = paddle_param_names - torch_param_names
    only_in_torch = torch_param_names - paddle_param_names

    print(f"\nExact name matches: {len(exact_matches)}")
    print(f"Only in Paddle: {len(only_in_paddle)}")
    print(f"Only in PyTorch: {len(only_in_torch)}")

    # Analyze naming pattern differences
    print("\n" + "-"*80)
    print("NAMING PATTERN ANALYSIS")
    print("-"*80)

    # Check for common transformations
    transformations = {
        '._mean -> .running_mean': [],
        '._variance -> .running_var': [],
        'Other differences': []
    }

    for paddle_name in only_in_paddle:
        if paddle_name.endswith('._mean'):
            expected_torch = paddle_name.replace('._mean', '.running_mean')
            if expected_torch in torch_buffer_names:
                transformations['._mean -> .running_mean'].append((paddle_name, expected_torch))
        elif paddle_name.endswith('._variance'):
            expected_torch = paddle_name.replace('._variance', '.running_var')
            if expected_torch in torch_buffer_names:
                transformations['._variance -> .running_var'].append((paddle_name, expected_torch))
        else:
            transformations['Other differences'].append(paddle_name)

    for key, items in transformations.items():
        if items:
            print(f"\n{key}: {len(items)}")
            if key != 'Other differences':
                for paddle_name, torch_name in items[:5]:
                    print(f"  {paddle_name} -> {torch_name}")
                if len(items) > 5:
                    print(f"  ... and {len(items) - 5} more")
            else:
                for name in items[:10]:
                    print(f"  {name}")
                if len(items) > 10:
                    print(f"  ... and {len(items) - 10} more")

    return exact_matches, only_in_paddle, only_in_torch


def print_side_by_side_comparison(paddle_params, torch_params, num_lines=50):
    """Print side-by-side parameter comparison"""

    print("\n" + "="*80)
    print("SIDE-BY-SIDE PARAMETER COMPARISON (First {} parameters)".format(num_lines))
    print("="*80)

    max_lines = min(num_lines, max(len(paddle_params), len(torch_params)))

    # Print header
    print(f"\n{'PaddlePaddle':<60} | {'PyTorch':<60}")
    print("-"*60 + "-+-" + "-"*60)

    for i in range(max_lines):
        paddle_str = ""
        torch_str = ""

        if i < len(paddle_params):
            name, shape = paddle_params[i]
            paddle_str = f"{name}: {shape}"

        if i < len(torch_params):
            name, shape = torch_params[i]
            torch_str = f"{name}: {shape}"

        # Highlight if names match
        marker = "✓" if paddle_str.split(':')[0] == torch_str.split(':')[0] else " "

        print(f"{paddle_str:<60} {marker} {torch_str:<60}")


def save_full_comparison(paddle_params, paddle_buffers, torch_params, torch_buffers, output_file):
    """Save full parameter list to file"""

    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("COMPLETE PARAMETER COMPARISON: PaddlePaddle vs PyTorch RT-DETRv3\n")
        f.write("="*100 + "\n\n")

        f.write(f"PaddlePaddle: {len(paddle_params)} parameters, {len(paddle_buffers)} buffers\n")
        f.write(f"PyTorch: {len(torch_params)} parameters, {len(torch_buffers)} buffers\n\n")

        f.write("-"*100 + "\n")
        f.write("PADDLEPADDLE PARAMETERS\n")
        f.write("-"*100 + "\n")
        for name, shape in paddle_params:
            f.write(f"{name}: {shape}\n")

        f.write("\n" + "-"*100 + "\n")
        f.write("PADDLEPADDLE BUFFERS\n")
        f.write("-"*100 + "\n")
        for name, shape in paddle_buffers:
            f.write(f"{name}: {shape}\n")

        f.write("\n" + "-"*100 + "\n")
        f.write("PYTORCH PARAMETERS\n")
        f.write("-"*100 + "\n")
        for name, shape in torch_params:
            f.write(f"{name}: {shape}\n")

        f.write("\n" + "-"*100 + "\n")
        f.write("PYTORCH BUFFERS\n")
        f.write("-"*100 + "\n")
        for name, shape in torch_buffers:
            f.write(f"{name}: {shape}\n")

    print(f"\nFull comparison saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare parameter names between Paddle and PyTorch RT-DETRv3 models"
    )
    parser.add_argument(
        '--paddle-config',
        type=str,
        required=True,
        help='Path to PaddlePaddle config file'
    )
    parser.add_argument(
        '--torch-config',
        type=str,
        required=True,
        help='Path to PyTorch config file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='logs/model_params_comparison.txt',
        help='Output file for full comparison (default: /tmp/model_params_comparison.txt)'
    )
    parser.add_argument(
        '--num-lines',
        type=int,
        default=50,
        help='Number of lines to print in terminal (default: 50)'
    )

    args = parser.parse_args()

    try:
        # Load Paddle model
        print("\n" + "="*80)
        print("LOADING PADDLEPADDLE MODEL")
        print("="*80)
        paddle_model = load_paddle_model(args.paddle_config)
        paddle_params, paddle_buffers = get_model_parameters(paddle_model, 'paddle')
        print(f"Loaded: {len(paddle_params)} parameters, {len(paddle_buffers)} buffers")

        # Load PyTorch model
        print("\n" + "="*80)
        print("LOADING PYTORCH MODEL")
        print("="*80)
        torch_model = load_pytorch_model(args.torch_config)
        torch_params, torch_buffers = get_model_parameters(torch_model, 'pytorch')
        print(f"Loaded: {len(torch_params)} parameters, {len(torch_buffers)} buffers")

        # Compare
        compare_parameters(paddle_params, torch_params, paddle_buffers, torch_buffers)

        # Print side-by-side comparison
        print_side_by_side_comparison(paddle_params, torch_params, args.num_lines)

        # Save full comparison
        save_full_comparison(paddle_params, paddle_buffers, torch_params, torch_buffers, args.output)

        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
