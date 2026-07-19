"""Tensor conversion utilities

This module provides utilities for converting tensors between PaddlePaddle and PyTorch formats,
with support for shape validation and dtype detection.
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def paddle_to_numpy(paddle_tensor) -> np.ndarray:
    """Convert PaddlePaddle tensor to NumPy array

    Args:
        paddle_tensor: PaddlePaddle tensor

    Returns:
        NumPy array (contiguous copy)

    Raises:
        ValueError: If tensor conversion fails
    """
    try:
        # Convert to numpy (creates contiguous copy)
        numpy_array = paddle_tensor.numpy()

        # Ensure contiguous memory layout
        if not numpy_array.flags["C_CONTIGUOUS"]:
            numpy_array = np.ascontiguousarray(numpy_array)

        return numpy_array
    except Exception as e:
        raise ValueError(f"Failed to convert PaddlePaddle tensor to NumPy: {e}")


def numpy_to_torch(numpy_array: np.ndarray):
    """Convert NumPy array to PyTorch tensor

    Args:
        numpy_array: NumPy array

    Returns:
        PyTorch tensor (shares memory with numpy array if contiguous)

    Raises:
        ValueError: If tensor conversion fails
    """
    try:
        import torch

        # Ensure contiguous memory layout
        if not numpy_array.flags["C_CONTIGUOUS"]:
            numpy_array = np.ascontiguousarray(numpy_array)

        # Convert to torch tensor (shares memory)
        torch_tensor = torch.from_numpy(numpy_array)

        return torch_tensor
    except Exception as e:
        raise ValueError(f"Failed to convert NumPy array to PyTorch tensor: {e}")


def validate_tensor_shape(
    tensor_shape: Tuple[int, ...],
    expected_shape: Tuple[int, ...],
    param_name: str,
    strict: bool = False,
) -> bool:
    """Validate tensor shape against expected shape

    Args:
        tensor_shape: Actual tensor shape
        expected_shape: Expected tensor shape
        param_name: Parameter name (for logging)
        strict: If True, raise exception on mismatch; if False, log warning

    Returns:
        True if shapes match, False otherwise

    Raises:
        ValueError: If strict=True and shapes don't match
    """
    shapes_match = tensor_shape == expected_shape

    if not shapes_match:
        error_msg = (
            f"Shape mismatch for parameter '{param_name}': "
            f"got {tensor_shape}, expected {expected_shape}"
        )

        if strict:
            raise ValueError(error_msg)
        else:
            logger.warning(error_msg)

    return shapes_match


def detect_dtype(tensor) -> str:
    """Detect dtype of a tensor (PaddlePaddle or PyTorch)

    Args:
        tensor: PaddlePaddle or PyTorch tensor

    Returns:
        String representation of dtype (e.g., 'float32', 'float16')

    Raises:
        ValueError: If dtype detection fails
    """
    try:
        # Try PaddlePaddle tensor
        if hasattr(tensor, "dtype"):
            dtype_str = str(tensor.dtype).split(".")[-1]  # Extract dtype name
            return dtype_str

        # Try NumPy array
        if isinstance(tensor, np.ndarray):
            return str(tensor.dtype)

        raise ValueError(f"Unknown tensor type: {type(tensor)}")
    except Exception as e:
        raise ValueError(f"Failed to detect tensor dtype: {e}")


def check_shape_compatibility(
    source_shape: Tuple[int, ...], target_shape: Tuple[int, ...]
) -> Tuple[bool, str]:
    """Check if source and target shapes are compatible

    Args:
        source_shape: Source tensor shape
        target_shape: Target tensor shape

    Returns:
        Tuple of (is_compatible, message):
            - is_compatible: True if shapes can be reconciled
            - message: Description of compatibility or suggested fix
    """
    # Exact match
    if source_shape == target_shape:
        return True, "Shapes match exactly"

    source_elements = int(np.prod(source_shape))
    target_elements = int(np.prod(target_shape))

    # A reversed 2-D shape is the known Paddle Linear -> PyTorch Linear case.
    if len(source_shape) == 2 and source_shape[::-1] == target_shape:
        return True, "Shapes are compatible by a 2-D transpose"

    if source_elements == target_elements:
        return (
            False,
            "Element counts match, but reshape/layout compatibility requires "
            "explicit semantic validation",
        )

    # Incompatible shapes
    return (
        False,
        f"Shapes are incompatible: {source_elements} != {target_elements} elements",
    )


def should_transpose_weight(param_name: str) -> bool:
    """Determine if a parameter weight should be transposed for PyTorch

    PaddlePaddle Linear layers use (in_features, out_features) format,
    while PyTorch uses (out_features, in_features). This function identifies
    Linear layer weights that need transposition.

    Args:
        param_name: Parameter name

    Returns:
        True if weight should be transposed, False otherwise
    """
    # Linear layer weights that need transposition
    # Patterns: *.weight for Linear layers (but not Conv layers)
    # Conv layers have 4D weights, so we'll check dimensionality later

    # Known Linear layer patterns in RT-DETRv3
    linear_patterns = [
        ".linear1.weight",  # FFN first linear
        ".linear2.weight",  # FFN second linear
        ".fc.weight",  # Fully connected layers (MLPs in heads)
        "_head.weight",  # Classification/regression heads (ends with _head.weight)
        "enc_score_head",  # Encoder score head
        "dec_score_head",  # Decoder score head
        "enc_bbox_head",  # Encoder bbox head (MLP last layer)
        "dec_bbox_head",  # Decoder bbox head (MLP last layer)
        "query_pos_head",  # Query position head
        "sampling_offsets.weight",  # Deformable attention
        "attention_weights.weight",  # Deformable attention
        "out_proj.weight",  # MultiHeadAttention output projection
        # NOTE: in_proj_weight should NOT be transposed - Paddle and PyTorch use same format [embed_dim, 3*embed_dim]
    ]

    # Check if parameter name matches any linear pattern
    for pattern in linear_patterns:
        if pattern in param_name:
            return True

    return False


def convert_paddle_to_torch_tensor(
    paddle_tensor,
    param_name: str = "unknown",
    auto_transpose: bool = True,
    transpose: Optional[bool] = None,
):
    """Convert PaddlePaddle tensor to PyTorch tensor (full pipeline)

    Args:
        paddle_tensor: PaddlePaddle tensor
        param_name: Parameter name (for logging)
        auto_transpose: Automatically transpose Linear layer weights if True
        transpose: Explicit target-aware transpose decision. When provided,
            this takes priority over name-based detection.

    Returns:
        PyTorch tensor

    Raises:
        ValueError: If conversion fails
    """
    try:
        # Step 1: PaddlePaddle -> NumPy
        numpy_array = paddle_to_numpy(paddle_tensor)
        logger.debug(
            f"Converted {param_name} to NumPy: shape={numpy_array.shape}, dtype={numpy_array.dtype}"
        )

        # Step 2: Check if this is a Linear layer weight that needs transposition
        transpose_weight = (
            transpose
            if transpose is not None
            else auto_transpose and should_transpose_weight(param_name)
        )
        if transpose_weight and numpy_array.ndim == 2:
            logger.debug(
                f"Transposing Linear layer weight: {param_name} from {numpy_array.shape} to {numpy_array.T.shape}"
            )
            numpy_array = numpy_array.T  # Transpose from (in, out) to (out, in)

        # Step 3: NumPy -> PyTorch
        torch_tensor = numpy_to_torch(numpy_array)
        logger.debug(
            f"Converted {param_name} to PyTorch: shape={torch_tensor.shape}, dtype={torch_tensor.dtype}"
        )

        return torch_tensor
    except Exception as e:
        raise ValueError(f"Failed to convert {param_name}: {e}")
