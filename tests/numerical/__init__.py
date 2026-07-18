"""
Numerical Equivalence Tests

This package contains tests to verify numerical equivalence between
PyTorch and PaddlePaddle implementations of RT-DETRv3 components.

Test Strategy:
1. Load identical weights into both implementations
2. Run inference on fixed random inputs (seed=42)
3. Compare outputs with strict tolerances:
   - Activations: max_diff < 1e-4
   - Predictions: ±0.01 for scores, ±2 pixels for bboxes
   - mAP: ±0.005 tolerance

Test Coverage:
- test_numerical_backbone.py: ResNet backbone equivalence
- test_numerical_neck.py: HybridEncoder neck equivalence
- test_numerical_decoder.py: TransformerDecoder equivalence
- test_numerical_e2e.py: End-to-end model equivalence
"""
