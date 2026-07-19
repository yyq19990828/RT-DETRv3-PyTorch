"""Create test fixtures for weight conversion tests

This script generates a sample PaddlePaddle checkpoint file for testing purposes.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def create_sample_paddle_checkpoint():
    """Create sample PaddlePaddle checkpoint file"""
    try:
        import paddle

        # Create sample state dict with typical parameter naming conventions
        state_dict = {
            # Conv2D layer
            "backbone.conv1.w_0": paddle.randn([64, 3, 7, 7], dtype="float32"),
            # BatchNorm layer
            "backbone.bn1._mean": paddle.randn([64], dtype="float32"),
            "backbone.bn1._variance": paddle.abs(paddle.randn([64], dtype="float32"))
            + 0.1,  # Must be positive
            "backbone.bn1._scale": paddle.randn([64], dtype="float32"),
            "backbone.bn1._offset": paddle.randn([64], dtype="float32"),
            # Linear layer
            "encoder.layer.0.self_attn.q_proj.w_0": paddle.randn(
                [256, 128], dtype="float32"
            ),
            "encoder.layer.0.self_attn.q_proj.b_0": paddle.randn(
                [256], dtype="float32"
            ),
            # Output layer
            "decoder.output.w_0": paddle.randn([10, 256], dtype="float32"),
            "decoder.output.b_0": paddle.randn([10], dtype="float32"),
        }

        # Save checkpoint
        output_path = Path(__file__).parent / "sample_paddle.pdparams"
        paddle.save(state_dict, str(output_path))
        print(f"Created sample PaddlePaddle checkpoint at: {output_path}")
        print(f"  Parameters: {len(state_dict)}")
        print(f"  Total size: {output_path.stat().st_size / 1024:.2f} KB")

        return True
    except ImportError:
        print("ERROR: PaddlePaddle is not installed. Run: uv sync --extra dev")
        return False
    except Exception as e:
        print(f"ERROR: Failed to create sample checkpoint: {e}")
        return False


if __name__ == "__main__":
    success = create_sample_paddle_checkpoint()
    sys.exit(0 if success else 1)
