"""Model output validation for weight conversion

This module provides utilities to validate numerical consistency between
Paddle and PyTorch models by comparing forward pass outputs.
"""

import logging
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ForwardPassResult:
    """Result of forward pass comparison"""

    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    output_shape: Tuple[int, ...]
    details: str


class ModelOutputValidator:
    """Validator for model output consistency between Paddle and PyTorch"""

    def __init__(self, rtol: float = 1e-4, atol: float = 1e-5):
        """Initialize validator

        Args:
            rtol: Relative tolerance for numerical comparison
            atol: Absolute tolerance for numerical comparison
        """
        self.rtol = rtol
        self.atol = atol

    def validate_forward_pass(
        self,
        paddle_model: Any,
        torch_model: Any,
        sample_input: np.ndarray,
    ) -> ForwardPassResult:
        """Validate numerical consistency of forward pass outputs

        Args:
            paddle_model: PaddlePaddle model
            torch_model: PyTorch model
            sample_input: Sample input as numpy array (B, C, H, W)

        Returns:
            ForwardPassResult with comparison details
        """
        import paddle
        import torch

        logger.info("Validating forward pass numerical consistency...")

        # Set models to eval mode
        paddle_model.eval()
        torch_model.eval()

        # Convert input to appropriate format
        # Both Paddle and PyTorch models expect dict input with 'image' key (Paddle mode)
        # Paddle also needs 'im_shape' and 'scale_factor' for post-processing
        batch_size = sample_input.shape[0]
        img_h, img_w = sample_input.shape[2], sample_input.shape[3]

        paddle_tensor = paddle.to_tensor(sample_input, dtype="float32")
        # im_shape: original image shape [H, W]
        # scale_factor: scale factor used to resize image [scale_y, scale_x]
        # For consistency test, we assume no scaling (scale_factor = [1.0, 1.0])
        paddle_input = {
            "image": paddle_tensor,
            "im_shape": paddle.to_tensor(
                [[img_h, img_w]] * batch_size, dtype="float32"
            ),
            "scale_factor": paddle.to_tensor(
                [[1.0, 1.0]] * batch_size, dtype="float32"
            ),
        }

        torch_tensor = torch.from_numpy(sample_input).float()
        torch_input = {
            "image": torch_tensor,
            "im_shape": torch.tensor(
                [[img_h, img_w]] * batch_size, dtype=torch.float32
            ),
            "scale_factor": torch.tensor(
                [[1.0, 1.0]] * batch_size, dtype=torch.float32
            ),
        }

        # RT-DETRv3 needs raw transformer outputs for a like-for-like comparison.
        # Generic models are invoked directly so this validator also works for
        # unit tests and other Paddle/PyTorch model pairs.
        is_rtdetr = all(
            hasattr(paddle_model, name) for name in ("backbone", "neck", "transformer")
        )
        with paddle.no_grad():
            if is_rtdetr:
                body_feats = paddle_model.backbone(paddle_input)
                neck_feats = paddle_model.neck(body_feats)
                transformer_output = paddle_model.transformer(neck_feats, None, None)
                if (
                    isinstance(transformer_output, (tuple, list))
                    and len(transformer_output) >= 4
                ):
                    paddle_output = {
                        "pred_boxes": transformer_output[2],
                        "pred_logits": transformer_output[3],
                    }
                else:
                    paddle_output = transformer_output
            else:
                paddle_output = paddle_model(paddle_tensor)

        with torch.no_grad():
            torch_output = torch_model(torch_input if is_rtdetr else torch_tensor)

        # Handle dict outputs (RT-DETRv3 returns dict with pred_boxes and pred_logits)
        if isinstance(paddle_output, dict) and isinstance(torch_output, dict):
            return self._compare_dict_outputs(paddle_output, torch_output)
        if isinstance(paddle_output, dict) or isinstance(torch_output, dict):
            raise TypeError(
                "Paddle and PyTorch models returned incompatible output structures"
            )

        # Handle tensor outputs
        if isinstance(paddle_output, (list, tuple)):
            paddle_output = paddle_output[0]
        if isinstance(torch_output, (list, tuple)):
            torch_output = torch_output[0]

        paddle_out_np = paddle_output.numpy()
        torch_out_np = torch_output.detach().cpu().numpy()

        return self._compare_tensors(paddle_out_np, torch_out_np, "output")

    def _compare_dict_outputs(
        self, paddle_output: dict, torch_output: dict
    ) -> ForwardPassResult:
        """Compare dictionary outputs (for RT-DETRv3)"""

        all_passed = True
        max_abs_diff_overall = 0.0
        mean_abs_diff_overall = 0.0
        max_rel_diff_overall = 0.0
        details_list = []

        # Compare each output tensor
        for key in paddle_output.keys():
            if key not in torch_output:
                details_list.append(f"❌ Key '{key}' missing in PyTorch output")
                all_passed = False
                continue

            paddle_tensor = paddle_output[key].numpy()
            torch_tensor = torch_output[key].detach().cpu().numpy()

            result = self._compare_tensors(paddle_tensor, torch_tensor, key)

            max_abs_diff_overall = max(max_abs_diff_overall, result.max_abs_diff)
            mean_abs_diff_overall = max(mean_abs_diff_overall, result.mean_abs_diff)
            max_rel_diff_overall = max(max_rel_diff_overall, result.max_rel_diff)

            status = "✅ MATCH" if result.passed else "❌ MISMATCH"
            details_list.append(
                f"{status} {key}:\n"
                f"  Shape: {paddle_tensor.shape}\n"
                f"  Max abs diff: {result.max_abs_diff:.2e}\n"
                f"  Mean abs diff: {result.mean_abs_diff:.2e}\n"
                f"  Max rel diff: {result.max_rel_diff:.2e}"
            )

            if not result.passed:
                all_passed = False

        details = "\n".join(details_list)
        output_shape = tuple(paddle_output[list(paddle_output.keys())[0]].shape)

        return ForwardPassResult(
            passed=all_passed,
            max_abs_diff=max_abs_diff_overall,
            mean_abs_diff=mean_abs_diff_overall,
            max_rel_diff=max_rel_diff_overall,
            output_shape=output_shape,
            details=details,
        )

    def _compare_tensors(
        self, paddle_array: np.ndarray, torch_array: np.ndarray, name: str
    ) -> ForwardPassResult:
        """Compare two numpy arrays"""

        # Check shape
        if paddle_array.shape != torch_array.shape:
            logger.error(
                f"Shape mismatch for {name}: Paddle {paddle_array.shape} != PyTorch {torch_array.shape}"
            )
            return ForwardPassResult(
                passed=False,
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                output_shape=paddle_array.shape,
                details=f"Shape mismatch for {name}: Paddle {paddle_array.shape} != PyTorch {torch_array.shape}",
            )

        # Check for NaN/Inf
        paddle_has_nan = np.isnan(paddle_array).any()
        paddle_has_inf = np.isinf(paddle_array).any()
        torch_has_nan = np.isnan(torch_array).any()
        torch_has_inf = np.isinf(torch_array).any()

        if paddle_has_nan or paddle_has_inf:
            logger.error(
                f"Paddle output for {name} contains NaN={paddle_has_nan}, Inf={paddle_has_inf}"
            )
        if torch_has_nan or torch_has_inf:
            logger.error(
                f"PyTorch output for {name} contains NaN={torch_has_nan}, Inf={torch_has_inf}"
            )

        # Compare outputs
        abs_diff = np.abs(paddle_array - torch_array)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = abs_diff / (np.abs(paddle_array) + 1e-10)
            max_rel_diff = np.max(rel_diff)

        passed = np.allclose(paddle_array, torch_array, rtol=self.rtol, atol=self.atol)

        details = (
            f"Comparison for {name}:\n"
            f"  Output shape: {paddle_array.shape}\n"
            f"  Max abs diff: {max_abs_diff:.2e}\n"
            f"  Mean abs diff: {mean_abs_diff:.2e}\n"
            f"  Max rel diff: {max_rel_diff:.2e}\n"
            f"  Tolerance: rtol={self.rtol}, atol={self.atol}\n"
            f"  Status: {'✅ PASSED' if passed else '❌ FAILED'}"
        )

        return ForwardPassResult(
            passed=passed,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
            max_rel_diff=max_rel_diff,
            output_shape=paddle_array.shape,
            details=details,
        )

    def print_validation_report(self, result: ForwardPassResult) -> None:
        """Print detailed validation report

        Args:
            result: ForwardPassResult to report
        """
        print("\n" + "=" * 80)
        print("MODEL OUTPUT VALIDATION REPORT")
        print("=" * 80)

        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(f"\nStatus: {status}")

        print("\nNumerical Statistics:")
        print(f"  Max absolute difference: {result.max_abs_diff:.2e}")
        print(f"  Mean absolute difference: {result.mean_abs_diff:.2e}")
        print(f"  Max relative difference: {result.max_rel_diff:.2e}")

        print("\nTolerance thresholds:")
        print(f"  Relative tolerance (rtol): {self.rtol}")
        print(f"  Absolute tolerance (atol): {self.atol}")

        if result.details:
            print("\nDetails:")
            print(result.details)

        print("\n" + "=" * 80)
