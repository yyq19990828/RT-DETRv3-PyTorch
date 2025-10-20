"""
Integration Tests for Backward Compatibility (User Story 4)

This test suite verifies that existing code continues to work after adding
the registry system. Tests cover common usage patterns that users might have
in their existing code.

Requirements (from tasks.md T065):
- Direct instantiation still works
- No warnings or errors
- All existing test patterns continue to work
- No breaking changes to public APIs

Following User Story 4 (US4) requirements
"""

import torch
import pytest
import warnings
from pathlib import Path
import sys
import io
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import components (existing usage pattern - direct imports)
from rtdetrv3_pytorch.models.backbones.resnet import ResNet
from rtdetrv3_pytorch.models.necks.hybrid_encoder import HybridEncoder
from rtdetrv3_pytorch.models.transformers.rtdetr_transformer import RTDETRTransformerv3
from rtdetrv3_pytorch.models.heads.detr_head import DINOv3Head
from rtdetrv3_pytorch.models.rtdetrv3 import RTDETRv3


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.mark.integration
class TestBackwardCompatibility:
    """Test that existing usage patterns continue to work"""

    def test_direct_instantiation_resnet(self):
        """
        Test Pattern 1: Direct class instantiation (ResNet)

        Users may have code like:
            backbone = ResNet(depth=50, variant='d')
        """
        # This should work without any issues
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            model = ResNet(
                depth=50,
                variant='d',
                frozen_stages=1,
                return_idx=[1, 2, 3]
            )

            # Check no warnings were raised
            assert len(w) == 0, f"Unexpected warnings: {[str(x.message) for x in w]}"

        # Should be able to use the model normally
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = model(x)

        assert len(out) == 3, "Expected 3 output features"
        print("✓ Direct ResNet instantiation - PASSED")

    def test_direct_instantiation_hybrid_encoder(self):
        """
        Test Pattern 2: Direct class instantiation (HybridEncoder)
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            model = HybridEncoder(
                in_channels=[512, 1024, 2048],
                feat_strides=[8, 16, 32],
                hidden_dim=256,
                use_encoder_idx=[2],
                num_encoder_layers=1
            )

            assert len(w) == 0, f"Unexpected warnings: {[str(x.message) for x in w]}"

        model.eval()
        x = [
            torch.randn(1, 512, 80, 80),
            torch.randn(1, 1024, 40, 40),
            torch.randn(1, 2048, 20, 20)
        ]
        with torch.no_grad():
            out = model(x)

        assert len(out) == 3, "Expected 3 output features"
        print("✓ Direct HybridEncoder instantiation - PASSED")

    def test_direct_instantiation_transformer(self):
        """
        Test Pattern 3: Direct class instantiation (RTDETRTransformerv3)
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            model = RTDETRTransformerv3(
                num_queries=300,
                num_decoder_layers=6,
                hidden_dim=256,
                num_classes=80
            )

            assert len(w) == 0, f"Unexpected warnings: {[str(x.message) for x in w]}"

        model.eval()
        # Transformer expects multi-scale features
        feats = [
            torch.randn(1, 256, 80, 80),
            torch.randn(1, 256, 40, 40),
            torch.randn(1, 256, 20, 20)
        ]
        with torch.no_grad():
            out = model(feats, targets=None)

        assert isinstance(out, tuple), "Expected tuple output"
        assert len(out) == 5, "Expected 5 outputs"
        print("✓ Direct RTDETRTransformerv3 instantiation - PASSED")

    def test_direct_instantiation_head(self):
        """
        Test Pattern 4: Direct class instantiation (DINOv3Head)
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            model = DINOv3Head(
                num_classes=80,
                hidden_dim=256
            )

            assert len(w) == 0, f"Unexpected warnings: {[str(x.message) for x in w]}"

        model.eval()
        # Head expects transformer output tuple
        dec_out_bboxes = torch.randn(6, 1, 300, 4)
        dec_out_logits = torch.randn(6, 1, 300, 80)
        enc_topk_bboxes = torch.randn(1, 300, 4)
        enc_topk_logits = torch.randn(1, 300, 80)
        dn_meta = None
        out_transformer = (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta)

        with torch.no_grad():
            out = model(out_transformer)

        assert isinstance(out, tuple), "Expected tuple output"
        assert len(out) == 3, "Expected 3 outputs"
        print("✓ Direct DINOv3Head instantiation - PASSED")

    def test_direct_instantiation_full_model(self):
        """
        Test Pattern 5: Direct instantiation of full RTDETRv3 model
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Build components using direct instantiation
            backbone = ResNet(
                depth=50,
                variant='d',
                return_idx=[1, 2, 3]
            )
            neck = HybridEncoder(
                in_channels=[512, 1024, 2048],
                feat_strides=[8, 16, 32],
                hidden_dim=256,
                use_encoder_idx=[2],
                num_encoder_layers=1
            )
            transformer = RTDETRTransformerv3(
                num_queries=300,
                num_decoder_layers=6,
                hidden_dim=256,
                num_classes=80
            )
            head = DINOv3Head(num_classes=80, hidden_dim=256)

            # Assemble model
            model = RTDETRv3(
                backbone=backbone,
                neck=neck,
                transformer=transformer,
                detr_head=head,
                num_classes=80
            )

            assert len(w) == 0, f"Unexpected warnings: {[str(x.message) for x in w]}"

        model.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = model(x)

        assert isinstance(out, dict), "Expected dict output"
        assert 'pred_logits' in out and 'pred_boxes' in out
        print("✓ Direct RTDETRv3 instantiation - PASSED")

    def test_registry_create_functions_work(self):
        """
        Test Pattern 6: Using new create() function with registry

        The new unified create() function replaces old build_* functions
        """
        from rtdetrv3_pytorch.models import create

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Build components using create() function
            backbone = create('ResNet', depth=50, variant='d', return_idx=[1, 2, 3])
            neck = create('HybridEncoder',
                in_channels=[512, 1024, 2048],
                feat_strides=[8, 16, 32],
                hidden_dim=256,
                use_encoder_idx=[2],
                num_encoder_layers=1
            )
            transformer = create('RTDETRTransformerv3',
                num_queries=300,
                num_decoder_layers=6,
                hidden_dim=256,
                num_classes=80
            )
            head = create('DINOv3Head', num_classes=80, hidden_dim=256)

            assert len(w) == 0, f"Unexpected warnings: {[str(x.message) for x in w]}"

        # Verify they work
        assert backbone is not None
        assert neck is not None
        assert transformer is not None
        assert head is not None

        print("✓ Registry create() function works - PASSED")

    def test_no_import_side_effects(self):
        """
        Test Pattern 7: Importing modules doesn't cause errors

        Users should be able to import modules without issues:
            from rtdetrv3_pytorch.models.backbones.resnet import ResNet
            from rtdetrv3_pytorch.models.necks.hybrid_encoder import HybridEncoder
        """
        # Capture any stderr/warnings during import
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Re-import to check for side effects
            import importlib
            import models.backbones.resnet
            import models.necks.hybrid_encoder
            import models.transformers.rtdetr_transformer
            import models.heads.detr_head
            import models.rtdetrv3

            importlib.reload(models.backbones.resnet)
            importlib.reload(models.necks.hybrid_encoder)
            importlib.reload(models.transformers.rtdetr_transformer)
            importlib.reload(models.heads.detr_head)
            importlib.reload(models.rtdetrv3)

            # Filter out deprecation warnings from other libraries
            user_warnings = [x for x in w if 'models' in str(x.filename)]

            assert len(user_warnings) == 0, (
                f"Import caused warnings: {[str(x.message) for x in user_warnings]}"
            )

        print("✓ No import side effects - PASSED")

    def test_model_serialization(self):
        """
        Test Pattern 8: Model saving and loading still works

        Users need to be able to save and load models:
            torch.save(model.state_dict(), 'checkpoint.pth')
            model.load_state_dict(torch.load('checkpoint.pth'))
        """
        import tempfile

        # Create a model
        model = ResNet(depth=50, variant='d', return_idx=[1, 2, 3])
        model.eval()

        # Save state dict
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
            torch.save(model.state_dict(), temp_path)

        # Load into a new model
        model2 = ResNet(depth=50, variant='d', return_idx=[1, 2, 3])
        model2.load_state_dict(torch.load(temp_path))
        model2.eval()

        # Verify they produce same output
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)

        for f1, f2 in zip(out1, out2):
            assert torch.allclose(f1, f2), "Loaded model produces different output"

        # Cleanup
        Path(temp_path).unlink()

        print("✓ Model serialization - PASSED")

    def test_no_logging_pollution(self):
        """
        Test Pattern 9: No excessive logging

        Registry system should not spam logs during normal usage
        """
        # Capture logs
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)

        logger = logging.getLogger('models')
        logger.addHandler(handler)
        old_level = logger.level
        logger.setLevel(logging.WARNING)

        try:
            # Create several models (normal usage)
            model1 = ResNet(depth=50, variant='d', return_idx=[1, 2, 3])
            model2 = HybridEncoder(
                in_channels=[512, 1024, 2048],
                feat_strides=[8, 16, 32],
                hidden_dim=256,
                use_encoder_idx=[2],
                num_encoder_layers=1
            )

            # Check no warnings/errors were logged
            log_output = log_capture.getvalue()
            assert len(log_output) == 0, f"Unexpected log output: {log_output}"

        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

        print("✓ No logging pollution - PASSED")

    def test_class_attributes_preserved(self):
        """
        Test Pattern 10: Class attributes are not modified by decorator

        The @register decorator should not modify class behavior:
        - __init__ signature unchanged
        - Methods unchanged
        - Attributes unchanged (except __category__, __inject__, __shared__)
        """
        # Check ResNet class
        import inspect

        # Get __init__ signature
        sig = inspect.signature(ResNet.__init__)
        params = list(sig.parameters.keys())

        # Essential parameters should exist
        assert 'depth' in params, "depth parameter missing"
        assert 'variant' in params, "variant parameter missing"
        assert 'return_idx' in params, "return_idx parameter missing"

        # Check methods exist
        assert hasattr(ResNet, 'forward'), "forward method missing"

        # Check that new registry attributes exist but don't interfere
        assert hasattr(ResNet, '__category__'), "__category__ should be added"
        assert hasattr(ResNet, '__inject__'), "__inject__ should be added"
        assert hasattr(ResNet, '__shared__'), "__shared__ should be added"

        # But they should be class-level, not instance-level interference
        model = ResNet(depth=50, variant='d', return_idx=[1, 2, 3])
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = model(x)

        assert len(out) == 3, "Model behavior changed"

        print("✓ Class attributes preserved - PASSED")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-m', 'integration'])
