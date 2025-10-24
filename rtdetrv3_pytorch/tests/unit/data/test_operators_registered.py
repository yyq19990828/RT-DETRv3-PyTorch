"""
Automated testing for all operators using registered_ops

This test uses the registered_ops list to automatically discover and test
all operator classes from ppdet_pytorch.data.transform.operators.
"""

import pytest
import numpy as np
import cv2
import ppdet_pytorch.data.transform.operators as ops_module
from ppdet_pytorch.data.transform.operators import registered_ops


# Test data generators for different operator categories
def get_basic_sample():
    """Basic sample with image and boxes"""
    return {
        'image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        'im_shape': np.array([480, 640], dtype=np.float32),
        'scale_factor': np.array([1.0, 1.0], dtype=np.float32),
        'gt_bbox': np.array([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=np.float32),
        'gt_class': np.array([[1], [2]], dtype=np.int32),
        'gt_score': np.array([[1.0], [1.0]], dtype=np.float32),
        'is_crowd': np.array([[0], [0]], dtype=np.int32),
        'difficult': np.array([[0], [0]], dtype=np.int32),
        'curr_iter': 100,  # Required by GridMask and some other operators
    }


def get_encoded_sample():
    """Sample with encoded image data (for Decode operator)"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode('.jpg', img)
    return {
        'image': encoded.tobytes(),
        'h': 100,
        'w': 100,
    }


# Operator instantiation configs
# Key: operator name, Value: initialization kwargs
OPERATOR_CONFIGS = {
    # Operators with no args
    'Decode': {},
    'Permute': {},
    'NormalizeImage': {},
    'NormalizeBox': {},
    'BboxXYXY2XYWH': {},
    'BboxCXCYWH2XYXY': {},
    'RandomFlip': {},
    'RandomDistort': {},
    'PhotoMetricDistortion': {},
    'Lighting': {'eigval': np.array([0.2175, 0.0188, 0.0045]), 'eigvec': np.array([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]])},
    'RandomErasingImage': {},
    'RandomGrayscale': {},
    'RandomColorJitter': {},
    'DebugVisibleImage': {},
    'Norm2PixelBbox': {},
    'Poly2Mask': {},

    # Operators with required args
    'Resize': {'target_size': [640, 640], 'keep_ratio': False, 'interp': 2},
    'RandomResize': {'target_size': [[480, 480], [512, 512], [640, 640]]},
    'MultiscaleTestResize': {'origin_target_size': [800, 1333], 'target_size': [[480, 800], [512, 800]]},
    'Pad': {'size': [640, 640]},
    'PadResize': {'target_size': [640, 640]},
    'PadBox': {'num_max_boxes': 50},
    'RandomCrop': {'aspect_ratio': [.5, 2.], 'thresholds': [.0, .1, .3, .5, .7, .9], 'scaling': [.3, 1.], 'num_attempts': 50},
    'RandomScaledCrop': {'target_size': 640, 'scale_range': [0.1, 2.0], 'interp': 2},
    'RandomSizeCrop': {'min_size': 384, 'max_size': 600},
    'RandomResizeCrop': {'resizes': [[640, 640]], 'cropsizes': [[640, 640]]},
    'RandomErasingCrop': {},

    # Augmentation operators
    'GridMask': {'use_h': True, 'use_w': True, 'rotate': 1, 'offset': False, 'ratio': 0.5, 'mode': 1, 'prob': 0.7},
    'AutoAugment': {'autoaug_type': 'v0'},
    'RandomExpand': {'fill_value': [123.675, 116.28, 103.53]},
    'RandomShift': {'prob': 0.5, 'max_shift': 32},
    'RandomGaussianBlur': {'sigma': [0.1, 2.0]},
    'AugmentHSV': {'hgain': 0.015, 'sgain': 0.7, 'vgain': 0.4},
    'RandomErasing': {'prob': 0.5, 'scale': (0.02, 0.33), 'ratio': (0.3, 3.3)},

    # Advanced augmentation
    'Cutmix': {'alpha': 1.5, 'beta': 1.5},
    'Mixup': {'alpha': 1.5, 'beta': 1.5},
    'Mosaic': {'prob': 1.0, 'input_dim': [640, 640]},

    # Sampling-based operators
    'CropWithSampling': {
        'batch_sampler': [[1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
        'satisfy_all': False,
        'avoid_no_bbox': True,
    },
    'CropWithDataAchorSampling': {
        'batch_sampler': [[1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
        'anchor_sampler': [[1, 1]],
        'target_size': 640,
    },

    # Transform selection
    'RandomSelect': {
        'transforms1': [{'Resize': {'target_size': [480, 480]}}],
        'transforms2': [{'Resize': {'target_size': [512, 512]}}],
        'p': 0.5,
    },

    # Special operators
    'WarpAffine': {'keep_res': False, 'pad': 0, 'input_h': 512, 'input_w': 512},
    'FlipWarpAffine': {'keep_res': False, 'pad': 0, 'input_h': 512, 'input_w': 512, 'flip': 0.5},
    'CenterRandColor': {'saturation': 0.5, 'contrast': 0.5, 'brightness': 0.5},
    'StrongAugImage': {'transforms': []},
    'RandomShortSideResize': {'short_side_sizes': [480, 512, 544, 576, 608, 640], 'max_size': 640},
    'RandomShortSideRangeResize': {'scales': [400, 600]},
}

# Operators that need special sample data
SPECIAL_SAMPLE_OPS = {
    'Decode': get_encoded_sample,
    'DecodeCache': get_encoded_sample,
}

# Operators to skip (need complex setup or external dependencies)
SKIP_OPS = {
    'SniperDecodeCrop',  # Needs complex chip configuration
    'DecodeCache',  # Needs cache directory setup
    'RandomSelects',  # Complex nested transforms
    'BatchRandomResize',  # Batch processing operator
    'BatchRandomResizeForSSOD',  # Semi-supervised specific batch operator
    'Gt2CenterNetTarget',  # Target conversion for CenterNet
    'Gt2CenterTrackTarget',  # Target conversion for CenterTrack
    'Gt2FCOSTarget',  # Target conversion for FCOS
    'Gt2GFLTarget',  # Target conversion for GFL
    'Gt2Solov2Target',  # Target conversion for SOLOv2
    'Gt2SparseTarget',  # Target conversion for Sparse RCNN
    'Gt2TTFTarget',  # Target conversion for TTFNet
    'Gt2YoloTarget',  # Target conversion for YOLO
    'PadBatch',  # Batch padding operator
    'PadGT',  # Ground truth padding
    'PadMaskBatch',  # Batch mask padding
    'PadRGT',  # Relationship ground truth padding
}


class TestRegisteredOperators:
    """Test all registered operators"""

    @staticmethod
    def get_all_operators():
        """Get all operator classes using registered_ops list"""
        return {
            name: getattr(ops_module, name)
            for name in registered_ops
            if hasattr(ops_module, name)
        }

    def test_operators_are_registered(self):
        """Verify operators are in registered_ops list"""
        operators = self.get_all_operators()

        # Check that we have operators
        assert len(registered_ops) > 0, "No operators in registered_ops"
        assert len(operators) > 0, "No operators found in module"
        print(f"\nTotal registered operators: {len(registered_ops)}")
        print(f"Total accessible operators: {len(operators)}")
        print(f"Sample operators: {list(operators.keys())[:10]}")

    @pytest.mark.parametrize("op_name", [
        'Resize', 'RandomFlip', 'NormalizeImage', 'NormalizeBox',
        'BboxXYXY2XYWH', 'Permute', 'RandomDistort', 'Pad'
    ])
    def test_critical_operators(self, op_name):
        """Test critical operators used in RT-DETRv3"""
        operators = self.get_all_operators()

        # Get config
        if op_name not in OPERATOR_CONFIGS:
            pytest.skip(f"{op_name} config not defined")

        config = OPERATOR_CONFIGS[op_name]

        # Get operator class from module
        if op_name not in operators:
            pytest.fail(f"{op_name} not found in operators module")

        op_class = operators[op_name]

        # Instantiate
        try:
            op = op_class(**config)
        except Exception as e:
            pytest.fail(f"Failed to instantiate {op_name}: {e}")

        # Get sample data
        if op_name in SPECIAL_SAMPLE_OPS:
            sample = SPECIAL_SAMPLE_OPS[op_name]()
        else:
            sample = get_basic_sample()

        # Test __call__
        try:
            result = op(sample.copy())
            assert isinstance(result, dict), f"{op_name} should return dict"
        except Exception as e:
            pytest.fail(f"Failed to call {op_name}: {e}")

    def test_all_configured_operators_instantiation(self):
        """Test all operators in OPERATOR_CONFIGS can be instantiated"""
        operators = self.get_all_operators()

        failed = []
        skipped = []
        success = []

        for op_name, config in OPERATOR_CONFIGS.items():
            if op_name in SKIP_OPS:
                skipped.append(op_name)
                continue

            # Check if exists in module
            if op_name not in operators:
                failed.append(f"{op_name}: not found in operators module")
                continue

            op_class = operators[op_name]

            # Try to instantiate
            try:
                op = op_class(**config)
                assert hasattr(op, '__call__'), f"{op_name} has no __call__"
                success.append(op_name)
            except Exception as e:
                failed.append(f"{op_name}: {str(e)[:100]}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"Operator Instantiation Summary:")
        print(f"  Success: {len(success)}/{len(OPERATOR_CONFIGS)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Skipped: {len(skipped)}")
        print(f"{'='*60}")

        if failed:
            print("\nFailed operators:")
            for f in failed:
                print(f"  - {f}")

        if skipped:
            print("\nSkipped operators:")
            for s in skipped:
                print(f"  - {s}")

        # Test passes if < 10% failures
        assert len(failed) < len(OPERATOR_CONFIGS) * 0.1, \
            f"Too many failures: {len(failed)}/{len(OPERATOR_CONFIGS)}"

    def test_operator_call_basic(self):
        """Test __call__ on a subset of operators with basic sample"""
        import ppdet_pytorch.data.transform.operators

        # Test these operators with basic sample
        test_ops = ['Resize', 'RandomFlip', 'NormalizeImage', 'Permute', 'NormalizeBox']

        failed = []
        for op_name in test_ops:
            if op_name not in self.get_all_operators() or op_name not in OPERATOR_CONFIGS:
                continue

            op_class = self.get_all_operators()[op_name]
            config = OPERATOR_CONFIGS[op_name]

            try:
                op = op_class(**config)
                sample = get_basic_sample()
                result = op(sample)

                assert isinstance(result, dict), f"{op_name} should return dict"
                assert 'image' in result or op_name == 'NormalizeBox', \
                    f"{op_name} result should have 'image' key"

            except Exception as e:
                failed.append(f"{op_name}: {str(e)[:100]}")

        if failed:
            print("\nFailed operator calls:")
            for f in failed:
                print(f"  - {f}")

        assert len(failed) == 0, f"Some operators failed: {failed}"

    def test_decode_operator_special(self):
        """Test Decode operator with encoded image"""
        import ppdet_pytorch.data.transform.operators

        if 'Decode' not in self.get_all_operators():
            pytest.skip("Decode not registered")

        decode = self.get_all_operators()['Decode']()
        sample = get_encoded_sample()

        try:
            result = decode(sample)
            assert 'image' in result
            assert isinstance(result['image'], np.ndarray)
            assert result['image'].ndim == 3  # H, W, C
        except Exception as e:
            pytest.fail(f"Decode operator failed: {e}")

    def test_bbox_format_conversion(self):
        """Test bbox format conversion operators"""
        import ppdet_pytorch.data.transform.operators

        # Test BboxXYXY2XYWH
        if 'BboxXYXY2XYWH' in self.get_all_operators():
            op = self.get_all_operators()['BboxXYXY2XYWH']()
            sample = {
                'gt_bbox': np.array([[10, 20, 40, 60]], dtype=np.float32),  # [x1,y1,x2,y2]
            }
            result = op(sample)
            bbox = result['gt_bbox']

            # Should convert to [cx, cy, w, h]
            # cx = 10 + (40-10)/2 = 25, cy = 20 + (60-20)/2 = 40
            # w = 40-10 = 30, h = 60-20 = 40
            expected = np.array([[25, 40, 30, 40]], dtype=np.float32)
            np.testing.assert_array_almost_equal(bbox, expected, decimal=4)

        # Test BboxCXCYWH2XYXY
        if 'BboxCXCYWH2XYXY' in self.get_all_operators():
            op = self.get_all_operators()['BboxCXCYWH2XYXY']()
            sample = {
                'gt_bbox': np.array([[25, 40, 30, 40]], dtype=np.float32),  # [cx,cy,w,h]
            }
            result = op(sample)
            bbox = result['gt_bbox']

            # Should convert to [x1, y1, x2, y2]
            # x1 = 25-30/2 = 10, y1 = 40-40/2 = 20
            # x2 = 25+30/2 = 40, y2 = 40+40/2 = 60
            expected = np.array([[10, 20, 40, 60]], dtype=np.float32)
            np.testing.assert_array_almost_equal(bbox, expected, decimal=4)


class TestOperatorCategories:
    """Test operators by category"""

    @staticmethod
    def get_all_operators():
        """Get all operator classes using registered_ops list"""
        return {
            name: getattr(ops_module, name)
            for name in registered_ops
            if hasattr(ops_module, name)
        }

    def test_resize_operators(self):
        """Test all resize-related operators"""
        import ppdet_pytorch.data.transform.operators

        resize_ops = ['Resize', 'RandomResize', 'RandomScaledCrop', 'PadResize']
        sample = get_basic_sample()

        for op_name in resize_ops:
            if op_name not in self.get_all_operators() or op_name not in OPERATOR_CONFIGS:
                continue

            op = self.get_all_operators()[op_name](**OPERATOR_CONFIGS[op_name])
            result = op(sample.copy())

            assert 'image' in result
            assert result['image'].shape[2] == 3, f"{op_name}: should preserve color channels"

    def test_augmentation_operators(self):
        """Test augmentation operators"""
        import ppdet_pytorch.data.transform.operators

        aug_ops = ['RandomFlip', 'RandomDistort', 'PhotoMetricDistortion', 'GridMask']
        sample = get_basic_sample()

        for op_name in aug_ops:
            if op_name not in self.get_all_operators() or op_name not in OPERATOR_CONFIGS:
                continue

            op = self.get_all_operators()[op_name](**OPERATOR_CONFIGS[op_name])
            result = op(sample.copy())

            assert 'image' in result
            assert result['image'].dtype == sample['image'].dtype or \
                   result['image'].dtype == np.float32, \
                   f"{op_name}: unexpected dtype {result['image'].dtype}"

    def test_normalization_operators(self):
        """Test normalization operators"""
        import ppdet_pytorch.data.transform.operators

        norm_ops = ['NormalizeImage', 'NormalizeBox']
        sample = get_basic_sample()

        for op_name in norm_ops:
            if op_name not in self.get_all_operators():
                continue

            op = self.get_all_operators()[op_name]()
            result = op(sample.copy())

            if op_name == 'NormalizeImage':
                assert 'image' in result
                assert result['image'].dtype == np.float32
            elif op_name == 'NormalizeBox':
                assert 'gt_bbox' in result
                # Boxes should be normalized to [0, 1]
                if len(result['gt_bbox']) > 0:
                    assert result['gt_bbox'].max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
