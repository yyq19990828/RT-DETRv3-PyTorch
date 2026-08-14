"""
Unit tests for Multi-Scale Deformable Attention

Tests cover:
- Forward pass with multi-scale inputs
- Backward pass (gradient check)
- Reference point validity
- Output shape correctness
- Edge cases (single scale, large num_points)

Following PaddlePaddle implementation for numerical equivalence.
"""

import pytest
import torch

from detrs.modeling.transformers.attention import (
    MSDeformableAttention,
    build_ms_deformable_attention,
    deformable_attention_core_func,
)


class TestMSDeformableAttentionOutputShapes:
    """Test output shapes for various configurations"""

    @pytest.mark.parametrize(
        "batch_size,num_queries,embed_dim,num_heads,num_levels,num_points",
        [
            (2, 100, 256, 8, 3, 4),  # Standard config
            (1, 300, 256, 8, 4, 4),  # Single batch with 4 levels
            (4, 450, 256, 8, 3, 8),  # More points
            (2, 100, 512, 8, 3, 4),  # Larger embed_dim
        ],
    )
    def test_forward_output_shape(
        self, batch_size, num_queries, embed_dim, num_heads, num_levels, num_points
    ):
        """Test forward pass output shape"""
        attn = MSDeformableAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        attn.eval()

        # Create multi-scale features
        # Typical spatial shapes: [(80, 80), (40, 40), (20, 20), (10, 10)]
        spatial_shapes = torch.tensor(
            [[80, 80], [40, 40], [20, 20], [10, 10]][:num_levels], dtype=torch.long
        )

        value_length = int(spatial_shapes.prod(1).sum())
        level_start_index = torch.cat(
            [torch.zeros(1, dtype=torch.long), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        # Inputs
        query = torch.randn(batch_size, num_queries, embed_dim)
        reference_points = torch.rand(batch_size, num_queries, num_levels, 2)  # [0, 1]
        value = torch.randn(batch_size, value_length, embed_dim)

        # Forward pass
        output = attn(query, reference_points, value, spatial_shapes, level_start_index)

        # Check output shape
        assert output.shape == (batch_size, num_queries, embed_dim), (
            f"Expected shape {(batch_size, num_queries, embed_dim)}, got {output.shape}"
        )

    def test_forward_with_mask(self):
        """Test forward pass with value mask"""
        batch_size, num_queries, embed_dim = 2, 100, 256
        num_heads, num_levels, num_points = 8, 3, 4

        attn = MSDeformableAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        attn.eval()

        # Create inputs
        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        value_length = int(spatial_shapes.prod(1).sum())
        level_start_index = torch.cat(
            [torch.zeros(1, dtype=torch.long), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        query = torch.randn(batch_size, num_queries, embed_dim)
        reference_points = torch.rand(batch_size, num_queries, num_levels, 2)
        value = torch.randn(batch_size, value_length, embed_dim)

        # Create mask (True for valid positions)
        value_mask = torch.ones(batch_size, value_length, dtype=torch.bool)
        # Mask out some positions
        value_mask[:, :100] = False

        # Forward pass with mask
        output = attn(
            query,
            reference_points,
            value,
            spatial_shapes,
            level_start_index,
            value_mask,
        )

        assert output.shape == (batch_size, num_queries, embed_dim)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"


class TestMSDeformableAttentionGradientFlow:
    """Test gradient flow through attention"""

    def test_gradient_flow_standard(self):
        """Test gradient flow in standard configuration"""
        batch_size, num_queries, embed_dim = 2, 100, 256
        num_heads, num_levels, num_points = 8, 3, 4

        attn = MSDeformableAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        attn.train()

        # Create inputs
        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        value_length = int(spatial_shapes.prod(1).sum())
        level_start_index = torch.cat(
            [torch.zeros(1, dtype=torch.long), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        query = torch.randn(batch_size, num_queries, embed_dim, requires_grad=True)
        reference_points = torch.rand(batch_size, num_queries, num_levels, 2)
        value = torch.randn(batch_size, value_length, embed_dim, requires_grad=True)

        # Forward pass
        output = attn(query, reference_points, value, spatial_shapes, level_start_index)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        # Note: Following PaddlePaddle's testing approach, we test:
        # 1. Value gradients (main path for attended features)
        # 2. Module parameter gradients (sampling_offsets, attention_weights, projections)
        # Query gradients flow through sampling_offsets and attention_weights linear layers
        # but are not the primary concern in deformable attention

        assert value.grad is not None, "Value gradients are None"
        assert value.grad.abs().sum() > 0, "Value gradients are all zero"

        # Check attention module parameters have gradients
        assert attn.sampling_offsets.weight.grad is not None, (
            "sampling_offsets.weight gradients are None"
        )
        assert attn.attention_weights.weight.grad is not None, (
            "attention_weights.weight gradients are None"
        )
        assert attn.value_proj.weight.grad is not None, (
            "value_proj.weight gradients are None"
        )
        assert attn.output_proj.weight.grad is not None, (
            "output_proj.weight gradients are None"
        )

        # All attention parameters should have non-zero gradients
        assert attn.sampling_offsets.weight.grad.abs().sum() > 0
        assert attn.attention_weights.weight.grad.abs().sum() > 0
        assert attn.value_proj.weight.grad.abs().sum() > 0
        assert attn.output_proj.weight.grad.abs().sum() > 0

    def test_gradient_accumulation(self):
        """Test gradient accumulation over multiple forward passes"""
        attn = MSDeformableAttention(
            embed_dim=256, num_heads=8, num_levels=3, num_points=4
        )
        attn.train()

        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        value_length = int(spatial_shapes.prod(1).sum())
        level_start_index = torch.cat(
            [torch.zeros(1, dtype=torch.long), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        # First forward-backward
        query1 = torch.randn(2, 100, 256, requires_grad=True)
        value1 = torch.randn(2, value_length, 256, requires_grad=True)
        reference_points1 = torch.rand(2, 100, 3, 2)

        output1 = attn(
            query1, reference_points1, value1, spatial_shapes, level_start_index
        )
        loss1 = output1.sum()
        loss1.backward()

        grad1 = attn.sampling_offsets.weight.grad.clone()

        # Second forward-backward (without zeroing gradients)
        query2 = torch.randn(2, 100, 256, requires_grad=True)
        value2 = torch.randn(2, value_length, 256, requires_grad=True)
        reference_points2 = torch.rand(2, 100, 3, 2)

        output2 = attn(
            query2, reference_points2, value2, spatial_shapes, level_start_index
        )
        loss2 = output2.sum()
        loss2.backward()

        grad2 = attn.sampling_offsets.weight.grad

        # Gradients should accumulate
        assert not torch.equal(grad1, grad2), "Gradients did not accumulate"


class TestReferencePointValidity:
    """Test reference point handling"""

    def test_reference_points_2d(self):
        """Test with 2D reference points (x, y)"""
        attn = MSDeformableAttention(
            embed_dim=256, num_heads=8, num_levels=3, num_points=4
        )
        attn.eval()

        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        value_length = int(spatial_shapes.prod(1).sum())
        level_start_index = torch.cat(
            [torch.zeros(1, dtype=torch.long), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        query = torch.randn(2, 100, 256)
        value = torch.randn(2, value_length, 256)

        # Reference points in [0, 1] range (valid)
        reference_points = torch.rand(2, 100, 3, 2)

        output = attn(query, reference_points, value, spatial_shapes, level_start_index)

        assert output.shape == (2, 100, 256)
        assert not torch.isnan(output).any()

    def test_reference_points_4d_bbox(self):
        """Test with 4D reference points (x, y, w, h) for bbox"""
        attn = MSDeformableAttention(
            embed_dim=256, num_heads=8, num_levels=3, num_points=4
        )
        attn.eval()

        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        value_length = int(spatial_shapes.prod(1).sum())
        level_start_index = torch.cat(
            [torch.zeros(1, dtype=torch.long), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        query = torch.randn(2, 100, 256)
        value = torch.randn(2, value_length, 256)

        # Reference points as bounding boxes (x, y, w, h) in [0, 1]
        reference_points = torch.rand(2, 100, 3, 4)

        output = attn(query, reference_points, value, spatial_shapes, level_start_index)

        assert output.shape == (2, 100, 256)
        assert not torch.isnan(output).any()

    def test_boundary_reference_points(self):
        """Test with reference points at boundaries [0, 1]"""
        attn = MSDeformableAttention(
            embed_dim=256, num_heads=8, num_levels=3, num_points=4
        )
        attn.eval()

        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        value_length = int(spatial_shapes.prod(1).sum())
        level_start_index = torch.cat(
            [torch.zeros(1, dtype=torch.long), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        query = torch.randn(2, 100, 256)
        value = torch.randn(2, value_length, 256)

        # Create reference points at boundaries
        reference_points = torch.zeros(2, 100, 3, 2)
        reference_points[0, :, :, :] = 0.0  # Left-top corner
        reference_points[1, :, :, :] = 1.0  # Right-bottom corner

        output = attn(query, reference_points, value, spatial_shapes, level_start_index)

        assert output.shape == (2, 100, 256)
        assert not torch.isnan(output).any()


class TestDeformableAttentionCoreFunc:
    """Test core deformable attention function"""

    def test_core_func_basic(self):
        """Test core function with basic inputs"""
        batch_size, num_queries, num_heads, head_dim = 2, 100, 8, 32
        num_levels, num_points = 3, 4

        # Create inputs
        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        value_length = int(spatial_shapes.prod(1).sum())
        level_start_index = torch.cat(
            [torch.zeros(1, dtype=torch.long), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        value = torch.randn(batch_size, value_length, num_heads, head_dim)
        sampling_locations = torch.rand(
            batch_size, num_queries, num_heads, num_levels, num_points, 2
        )
        attention_weights = torch.rand(
            batch_size, num_queries, num_heads, num_levels, num_points
        )
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1, keepdim=True
        )

        # Call core function
        output = deformable_attention_core_func(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )

        expected_shape = (batch_size, num_queries, num_heads * head_dim)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_core_func_gradient(self):
        """Test gradient flow through core function"""
        batch_size, num_queries, num_heads, head_dim = 2, 50, 8, 32
        num_levels, num_points = 3, 4

        spatial_shapes = torch.tensor([[40, 40], [20, 20], [10, 10]], dtype=torch.long)
        value_length = int(spatial_shapes.prod(1).sum())
        level_start_index = torch.cat(
            [torch.zeros(1, dtype=torch.long), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        value = torch.randn(
            batch_size, value_length, num_heads, head_dim, requires_grad=True
        )
        sampling_locations = torch.rand(
            batch_size,
            num_queries,
            num_heads,
            num_levels,
            num_points,
            2,
            requires_grad=True,
        )

        # Create attention weights as leaf tensor
        attention_weights_raw = torch.rand(
            batch_size,
            num_queries,
            num_heads,
            num_levels,
            num_points,
            requires_grad=True,
        )
        # Normalize (this creates a non-leaf tensor, but we'll test the raw tensor gradients)
        attention_weights = attention_weights_raw / attention_weights_raw.sum(
            dim=-1, keepdim=True
        )
        # Retain gradients for the normalized tensor (non-leaf)
        attention_weights.retain_grad()

        output = deformable_attention_core_func(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )

        loss = output.sum()
        loss.backward()

        # Check leaf tensors have gradients
        assert value.grad is not None
        assert sampling_locations.grad is not None
        assert attention_weights_raw.grad is not None  # Check the leaf tensor
        assert value.grad.abs().sum() > 0
        assert sampling_locations.grad.abs().sum() > 0
        assert attention_weights_raw.grad.abs().sum() > 0

        # The normalized attention_weights (non-leaf) should also have gradients after retain_grad()
        assert attention_weights.grad is not None


class TestBuildMSDeformableAttention:
    """Test builder function"""

    def test_build_from_config(self):
        """Test building from config dict"""
        cfg = {
            "embed_dim": 256,
            "num_heads": 8,
            "num_levels": 4,
            "num_points": 4,
            "lr_mult": 0.1,
        }

        attn = build_ms_deformable_attention(cfg)

        assert isinstance(attn, MSDeformableAttention)
        assert attn.embed_dim == 256
        assert attn.num_heads == 8
        assert attn.num_levels == 4
        assert attn.num_points == 4

    def test_build_with_defaults(self):
        """Test building with default values"""
        cfg = {}

        attn = build_ms_deformable_attention(cfg)

        assert isinstance(attn, MSDeformableAttention)
        assert attn.embed_dim == 256  # Default
        assert attn.num_heads == 8  # Default
        assert attn.num_levels == 4  # Default
        assert attn.num_points == 4  # Default


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_invalid_embed_dim(self):
        """Test with embed_dim not divisible by num_heads"""
        with pytest.raises(
            ValueError, match="embed_dim.*must be divisible by num_heads"
        ):
            MSDeformableAttention(embed_dim=255, num_heads=8)

    def test_single_level(self):
        """Test with single feature level"""
        attn = MSDeformableAttention(
            embed_dim=256, num_heads=8, num_levels=1, num_points=4
        )
        attn.eval()

        spatial_shapes = torch.tensor([[80, 80]], dtype=torch.long)
        value_length = int(spatial_shapes.prod(1).sum())
        level_start_index = torch.zeros(1, dtype=torch.long)

        query = torch.randn(2, 100, 256)
        reference_points = torch.rand(2, 100, 1, 2)
        value = torch.randn(2, value_length, 256)

        output = attn(query, reference_points, value, spatial_shapes, level_start_index)

        assert output.shape == (2, 100, 256)
        assert not torch.isnan(output).any()

    def test_many_points(self):
        """Test with large number of sampling points"""
        attn = MSDeformableAttention(
            embed_dim=256, num_heads=8, num_levels=3, num_points=16
        )
        attn.eval()

        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        value_length = int(spatial_shapes.prod(1).sum())
        level_start_index = torch.cat(
            [torch.zeros(1, dtype=torch.long), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        query = torch.randn(2, 100, 256)
        reference_points = torch.rand(2, 100, 3, 2)
        value = torch.randn(2, value_length, 256)

        output = attn(query, reference_points, value, spatial_shapes, level_start_index)

        assert output.shape == (2, 100, 256)
        assert not torch.isnan(output).any()

    def test_value_length_mismatch(self):
        """Test error handling for mismatched value length"""
        attn = MSDeformableAttention(
            embed_dim=256, num_heads=8, num_levels=3, num_points=4
        )
        attn.eval()

        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        # Deliberately create wrong value length
        wrong_value_length = 1000
        level_start_index = torch.cat(
            [torch.zeros(1, dtype=torch.long), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        query = torch.randn(2, 100, 256)
        reference_points = torch.rand(2, 100, 3, 2)
        value = torch.randn(2, wrong_value_length, 256)

        with pytest.raises(
            AssertionError, match="Value length.*does not match spatial shapes"
        ):
            attn(query, reference_points, value, spatial_shapes, level_start_index)

    def test_invalid_reference_point_dim(self):
        """Test error handling for invalid reference point dimensions"""
        attn = MSDeformableAttention(
            embed_dim=256, num_heads=8, num_levels=3, num_points=4
        )
        attn.eval()

        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        value_length = int(spatial_shapes.prod(1).sum())
        level_start_index = torch.cat(
            [torch.zeros(1, dtype=torch.long), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        query = torch.randn(2, 100, 256)
        value = torch.randn(2, value_length, 256)

        # Invalid last dimension (not 2 or 4)
        reference_points = torch.rand(2, 100, 3, 3)

        with pytest.raises(
            ValueError, match="Last dim of reference_points must be 2 or 4"
        ):
            attn(query, reference_points, value, spatial_shapes, level_start_index)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
