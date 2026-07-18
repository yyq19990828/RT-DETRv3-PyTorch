"""
Unit tests for Transformer Decoder

Tests cover:
- TransformerDecoderLayer forward pass
- TransformerDecoder forward pass
- MultiHeadAttention mechanism
- Gradient flow through decoder
- Iterative refinement

Following PaddlePaddle implementation for numerical equivalence.
"""

import pytest
import torch
import torch.nn as nn
from ppdet_pytorch.modeling.transformers.rtdetr_transformerv3 import (
    TransformerDecoderLayer,
    TransformerDecoder,
    MultiHeadAttention
)
from ppdet_pytorch.modeling.transformers.utils import MLP


class TestMultiHeadAttention:
    """Test MultiHeadAttention module"""

    def test_forward_shape(self):
        """Test forward pass output shape"""
        attn = MultiHeadAttention(embed_dim=256, num_heads=8, dropout=0.1)
        attn.eval()

        query = torch.randn(2, 100, 256)
        key = torch.randn(2, 300, 256)
        value = torch.randn(2, 300, 256)

        output = attn(query, key, value)

        assert output.shape == (2, 100, 256)

    def test_self_attention(self):
        """Test self-attention (query = key = value)"""
        attn = MultiHeadAttention(embed_dim=256, num_heads=8)
        attn.eval()

        x = torch.randn(2, 100, 256)
        output = attn(x, x, x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_with_attention_mask(self):
        """Test attention with mask"""
        attn = MultiHeadAttention(embed_dim=256, num_heads=8)
        attn.eval()

        query = torch.randn(2, 100, 256)
        key = torch.randn(2, 300, 256)
        value = torch.randn(2, 300, 256)

        # Create additive mask (0 for valid, -inf for invalid)
        attn_mask = torch.zeros(2, 100, 300)
        attn_mask[:, :, 200:] = float('-inf')  # Mask out last 100 positions

        output = attn(query, key, value, attn_mask)

        assert output.shape == (2, 100, 256)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test gradient flow through attention"""
        attn = MultiHeadAttention(embed_dim=256, num_heads=8)
        attn.train()

        query = torch.randn(2, 100, 256, requires_grad=True)
        key = torch.randn(2, 300, 256, requires_grad=True)
        value = torch.randn(2, 300, 256, requires_grad=True)

        output = attn(query, key, value)
        loss = output.sum()
        loss.backward()

        assert query.grad is not None and query.grad.abs().sum() > 0
        assert key.grad is not None and key.grad.abs().sum() > 0
        assert value.grad is not None and value.grad.abs().sum() > 0


class TestTransformerDecoderLayer:
    """Test TransformerDecoderLayer"""

    def test_forward_shape(self):
        """Test forward pass output shape"""
        layer = TransformerDecoderLayer(
            d_model=256,
            n_head=8,
            dim_feedforward=1024,
            n_levels=3,
            n_points=4
        )
        layer.eval()

        # Prepare inputs
        tgt = torch.randn(2, 100, 256)
        ref_points = torch.rand(2, 100, 3, 2)
        memory = torch.randn(2, 8400, 256)
        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        level_start_index = torch.tensor([0, 6400, 8000], dtype=torch.long)

        output = layer(tgt, ref_points, memory, spatial_shapes, level_start_index)

        assert output.shape == tgt.shape
        assert not torch.isnan(output).any()

    def test_with_query_pos_embed(self):
        """Test forward pass with query positional embedding"""
        layer = TransformerDecoderLayer(d_model=256, n_head=8, n_levels=3, n_points=4)
        layer.eval()

        tgt = torch.randn(2, 100, 256)
        ref_points = torch.rand(2, 100, 3, 2)
        memory = torch.randn(2, 8400, 256)
        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        level_start_index = torch.tensor([0, 6400, 8000], dtype=torch.long)
        query_pos_embed = torch.randn(2, 100, 256)

        output = layer(
            tgt, ref_points, memory, spatial_shapes, level_start_index,
            query_pos_embed=query_pos_embed
        )

        assert output.shape == tgt.shape

    def test_with_attention_mask(self):
        """Test forward pass with self-attention mask"""
        layer = TransformerDecoderLayer(d_model=256, n_head=8, n_levels=3, n_points=4)
        layer.eval()

        tgt = torch.randn(2, 100, 256)
        ref_points = torch.rand(2, 100, 3, 2)
        memory = torch.randn(2, 8400, 256)
        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        level_start_index = torch.tensor([0, 6400, 8000], dtype=torch.long)

        # Create attention mask
        attn_mask = torch.zeros(2, 100, 100)
        attn_mask[:, :50, 50:] = 1.0  # Mask half of the queries

        output = layer(
            tgt, ref_points, memory, spatial_shapes, level_start_index,
            attn_mask=attn_mask
        )

        assert output.shape == tgt.shape

    def test_gradient_flow(self):
        """Test gradient flow through decoder layer"""
        layer = TransformerDecoderLayer(d_model=256, n_head=8, n_levels=3, n_points=4)
        layer.train()

        tgt = torch.randn(2, 100, 256, requires_grad=True)
        ref_points = torch.rand(2, 100, 3, 2)
        memory = torch.randn(2, 8400, 256, requires_grad=True)
        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        level_start_index = torch.tensor([0, 6400, 8000], dtype=torch.long)

        output = layer(tgt, ref_points, memory, spatial_shapes, level_start_index)
        loss = output.sum()
        loss.backward()

        # Check gradients (tgt goes through self-attn, memory goes through cross-attn)
        assert memory.grad is not None and memory.grad.abs().sum() > 0


class TestTransformerDecoder:
    """Test TransformerDecoder"""

    def test_forward_shape_training(self):
        """Test forward pass output shapes during training"""
        # Create decoder following PaddlePaddle style (direct instantiation)
        decoder_layer = TransformerDecoderLayer(
            d_model=256,
            n_head=8,
            dim_feedforward=1024,
            n_levels=3,
            n_points=4
        )
        decoder = TransformerDecoder(
            hidden_dim=256,
            decoder_layer=decoder_layer,
            num_layers=6,
            eval_idx=-1
        )
        decoder.train()

        # Prepare inputs
        tgt = torch.randn(2, 300, 256)
        ref_points_unact = torch.randn(2, 300, 4)  # Unactivated (will be sigmoid-ed)
        memory = torch.randn(2, 8400, 256)
        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        level_start_index = torch.tensor([0, 6400, 8000], dtype=torch.long)

        # Create bbox and score heads (one per layer)
        bbox_head = nn.ModuleList([MLP(256, 256, 4, num_layers=3) for _ in range(6)])
        score_head = nn.ModuleList([nn.Linear(256, 80) for _ in range(6)])
        query_pos_head = MLP(4, 512, 256, num_layers=2)

        # Forward pass
        dec_out_bboxes, dec_out_logits = decoder(
            tgt, ref_points_unact, memory, spatial_shapes, level_start_index,
            bbox_head, score_head, query_pos_head
        )

        # Check shapes: (num_layers, batch, num_queries, ...)
        assert dec_out_bboxes.shape == (6, 2, 300, 4)
        assert dec_out_logits.shape == (6, 2, 300, 80)
        assert not torch.isnan(dec_out_bboxes).any()
        assert not torch.isnan(dec_out_logits).any()

    def test_forward_shape_eval(self):
        """Test forward pass output shapes during eval"""
        # Create decoder following PaddlePaddle style
        decoder_layer = TransformerDecoderLayer(
            d_model=256,
            n_head=8,
            dim_feedforward=1024,
            n_levels=3,
            n_points=4
        )
        decoder = TransformerDecoder(
            hidden_dim=256,
            decoder_layer=decoder_layer,
            num_layers=6,
            eval_idx=-1  # Use last layer
        )
        decoder.eval()

        tgt = torch.randn(2, 300, 256)
        ref_points_unact = torch.randn(2, 300, 4)
        memory = torch.randn(2, 8400, 256)
        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        level_start_index = torch.tensor([0, 6400, 8000], dtype=torch.long)

        bbox_head = nn.ModuleList([MLP(256, 256, 4, num_layers=3) for _ in range(6)])
        score_head = nn.ModuleList([nn.Linear(256, 80) for _ in range(6)])
        query_pos_head = MLP(4, 512, 256, num_layers=2)

        dec_out_bboxes, dec_out_logits = decoder(
            tgt, ref_points_unact, memory, spatial_shapes, level_start_index,
            bbox_head, score_head, query_pos_head
        )

        # In eval mode, only one layer output is returned
        assert dec_out_bboxes.shape == (1, 2, 300, 4)
        assert dec_out_logits.shape == (1, 2, 300, 80)

    def test_iterative_refinement(self):
        """Test iterative bounding box refinement across layers"""
        decoder_layer = TransformerDecoderLayer(d_model=256, n_head=8, n_levels=3, n_points=4)
        decoder = TransformerDecoder(hidden_dim=256, decoder_layer=decoder_layer, num_layers=6, eval_idx=-1)
        decoder.train()

        tgt = torch.randn(2, 300, 256)
        ref_points_unact = torch.randn(2, 300, 4)
        memory = torch.randn(2, 8400, 256)
        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        level_start_index = torch.tensor([0, 6400, 8000], dtype=torch.long)

        bbox_head = nn.ModuleList([MLP(256, 256, 4, num_layers=3) for _ in range(6)])
        score_head = nn.ModuleList([nn.Linear(256, 80) for _ in range(6)])
        query_pos_head = MLP(4, 512, 256, num_layers=2)

        dec_out_bboxes, dec_out_logits = decoder(
            tgt, ref_points_unact, memory, spatial_shapes, level_start_index,
            bbox_head, score_head, query_pos_head
        )

        # Check that we have predictions for all 6 layers
        assert len(dec_out_bboxes) == 6
        assert len(dec_out_logits) == 6

        # All bboxes should be in [0, 1] range (sigmoid applied)
        assert (dec_out_bboxes >= 0).all() and (dec_out_bboxes <= 1).all()

    def test_gradient_flow(self):
        """Test gradient flow through entire decoder"""
        decoder_layer = TransformerDecoderLayer(d_model=256, n_head=8, n_levels=3, n_points=4)
        decoder = TransformerDecoder(hidden_dim=256, decoder_layer=decoder_layer, num_layers=3, eval_idx=-1)
        decoder.train()

        tgt = torch.randn(2, 100, 256, requires_grad=True)
        ref_points_unact = torch.randn(2, 100, 4, requires_grad=True)
        memory = torch.randn(2, 8400, 256, requires_grad=True)
        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        level_start_index = torch.tensor([0, 6400, 8000], dtype=torch.long)

        bbox_head = nn.ModuleList([MLP(256, 256, 4, num_layers=3) for _ in range(3)])
        score_head = nn.ModuleList([nn.Linear(256, 80) for _ in range(3)])
        query_pos_head = MLP(4, 512, 256, num_layers=2)

        dec_out_bboxes, dec_out_logits = decoder(
            tgt, ref_points_unact, memory, spatial_shapes, level_start_index,
            bbox_head, score_head, query_pos_head
        )

        loss = dec_out_bboxes.sum() + dec_out_logits.sum()
        loss.backward()

        # Check gradients exist
        assert memory.grad is not None and memory.grad.abs().sum() > 0
        # Note: ref_points_unact may be detached during training, so we don't check its gradient


class TestDirectInstantiation:
    """Test direct decoder instantiation (PaddlePaddle style)"""

    def test_instantiate_with_custom_config(self):
        """Test instantiating decoder with custom config"""
        decoder_layer = TransformerDecoderLayer(
            d_model=256,
            n_head=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            n_levels=4,
            n_points=4
        )
        decoder = TransformerDecoder(
            hidden_dim=256,
            decoder_layer=decoder_layer,
            num_layers=6,
            eval_idx=-1
        )

        assert isinstance(decoder, TransformerDecoder)
        assert decoder.num_layers == 6
        assert decoder.hidden_dim == 256
        assert decoder.eval_idx == 5  # -1 means last layer (index 5)

    def test_instantiate_with_defaults(self):
        """Test instantiating decoder with default values"""
        decoder_layer = TransformerDecoderLayer(d_model=256, n_head=8, n_levels=3, n_points=4)
        decoder = TransformerDecoder(hidden_dim=256, decoder_layer=decoder_layer, num_layers=6, eval_idx=-1)

        assert isinstance(decoder, TransformerDecoder)
        assert decoder.num_layers == 6
        assert decoder.hidden_dim == 256


class TestEdgeCases:
    """Test edge cases"""

    def test_single_query(self):
        """Test with single query"""
        layer = TransformerDecoderLayer(d_model=256, n_head=8, n_levels=3, n_points=4)
        layer.eval()

        tgt = torch.randn(1, 1, 256)
        ref_points = torch.rand(1, 1, 3, 2)  # 3 levels
        # Memory with 3 levels: 10*10 + 5*5 + 5*5 = 100 + 25 + 25 = 150
        memory = torch.randn(1, 150, 256)
        spatial_shapes = torch.tensor([[10, 10], [5, 5], [5, 5]], dtype=torch.long)
        level_start_index = torch.tensor([0, 100, 125], dtype=torch.long)

        output = layer(tgt, ref_points, memory, spatial_shapes, level_start_index)

        assert output.shape == (1, 1, 256)

    def test_large_num_queries(self):
        """Test with large number of queries"""
        layer = TransformerDecoderLayer(d_model=256, n_head=8, n_levels=3, n_points=4)
        layer.eval()

        tgt = torch.randn(2, 1000, 256)
        ref_points = torch.rand(2, 1000, 3, 2)
        memory = torch.randn(2, 8400, 256)
        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        level_start_index = torch.tensor([0, 6400, 8000], dtype=torch.long)

        output = layer(tgt, ref_points, memory, spatial_shapes, level_start_index)

        assert output.shape == (2, 1000, 256)

    def test_different_activations(self):
        """Test with different activation functions"""
        for activation in ['relu', 'gelu']:
            layer = TransformerDecoderLayer(
                d_model=256, n_head=8, activation=activation, n_levels=3, n_points=4
            )
            layer.eval()

            tgt = torch.randn(2, 100, 256)
            ref_points = torch.rand(2, 100, 3, 2)
            memory = torch.randn(2, 8400, 256)
            spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
            level_start_index = torch.tensor([0, 6400, 8000], dtype=torch.long)

            output = layer(tgt, ref_points, memory, spatial_shapes, level_start_index)
            assert output.shape == (2, 100, 256)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
