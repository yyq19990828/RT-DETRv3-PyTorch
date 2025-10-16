"""
Weight conversion utility: PaddlePaddle to PyTorch

This script is a backward-compatible entry point that delegates to the new
modular weight_conversion package.

**DEPRECATED**: This file is kept for backward compatibility.
New code should use: python -m tools.weight_conversion.cli

Usage:
    # New recommended way:
    python -m tools.weight_conversion.cli \
        --input path/to/model.pdparams \
        --output converted.pth \
        --save-mapping mapping.json

    # Old way (still works):
    python tools/convert_weights.py \
        --paddle_checkpoint path/to/model.pdparams \
        --output converted.pth \
        --save_mapping mapping.json

For full documentation, see: specs/003-paddle-pytorch-conversion/quickstart.md
"""

import sys
import warnings

# Emit deprecation warning
warnings.warn(
    "tools/convert_weights.py is deprecated. "
    "Please use: python -m tools.weight_conversion.cli instead. "
    "This compatibility wrapper will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new modular implementation
from tools.weight_conversion.cli import main

if __name__ == '__main__':
    main()
