"""
HybridEncoder for RT-DETRv3 PyTorch implementation
A hybrid encoder that combines CNN and Transformer features
"""

import copy
import math
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register

__all__ = ['HybridEncoder']


def get_activation(act: str, inplace: bool = True) -> nn.Module:
    """Get activation function by name"""
    if act is None:
        return nn.Identity()
    
    if isinstance(act, nn.Module):
        return act
    
    act = act.lower()
    
    if act == 'silu' or act == 'swish':
        return nn.SiLU(inplace=inplace)
    elif act == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act == 'leaky_relu':
        return nn.LeakyReLU(inplace=inplace)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'hardsigmoid':
        return nn.Hardsigmoid(inplace=inplace)
    elif act == 'hardswish':
        return nn.Hardswish(inplace=inplace)
    elif act == 'mish':
        return nn.Mish(inplace=inplace)
    else:
        raise ValueError(f"Unsupported activation: {act}")


class ConvNormLayer(nn.Module):
    """Convolution + Normalization + Activation layer"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: Optional[int] = None,
                 bias: bool = False,
                 activation: Optional[str] = None,
                 norm_type: str = 'bn'):
        super().__init__()
        
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias
        )
        
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'sync_bn':
            self.norm = nn.SyncBatchNorm(out_channels)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, out_channels)
        elif norm_type == 'ln':
            self.norm = nn.LayerNorm(out_channels)
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")
        
        self.activation = nn.Identity() if activation is None else get_activation(activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class RepVGGBlock(nn.Module):
    """RepVGG Block with structural re-parameterization"""
    
    def __init__(self, in_channels: int, out_channels: int, activation: str = 'relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv3x3 = ConvNormLayer(in_channels, out_channels, 3, 1, activation=None)
        self.conv1x1 = ConvNormLayer(in_channels, out_channels, 1, 1, activation=None)
        
        self.activation = nn.Identity() if activation is None else get_activation(activation)
        
        # For deploy mode
        self.deploy = False
        self.conv_deploy = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy and self.conv_deploy is not None:
            return self.activation(self.conv_deploy(x))
        else:
            return self.activation(self.conv3x3(x) + self.conv1x1(x))
    
    def convert_to_deploy(self):
        """Convert to deploy mode by fusing conv and bn"""
        if self.deploy:
            return
        
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_deploy = nn.Conv2d(
            self.in_channels, self.out_channels, 3, 1, 1, bias=True
        )
        self.conv_deploy.weight.data = kernel
        self.conv_deploy.bias.data = bias
        self.deploy = True
        
        # Remove original convs to save memory
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1')
    
    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get equivalent kernel and bias for deploy mode"""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv3x3)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv1x1)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1: torch.Tensor) -> torch.Tensor:
        """Pad 1x1 kernel to 3x3"""
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, branch: ConvNormLayer) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse conv and bn parameters"""
        if branch is None:
            return 0, 0
        
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    """CSP RepVGG Layer"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 3,
                 expansion: float = 1.0,
                 bias: bool = False,
                 activation: str = 'silu'):
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, activation=activation)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, activation=activation)
        
        self.bottlenecks = nn.Sequential(*[
            RepVGGBlock(hidden_channels, hidden_channels, activation=activation)
            for _ in range(num_blocks)
        ])
        
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, activation=activation)
        else:
            self.conv3 = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x1 = self.bottlenecks(x1)
        x2 = self.conv2(x)
        return self.conv3(x1 + x2)


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer"""
    
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 normalize_before: bool = False):
        super().__init__()
        
        self.normalize_before = normalize_before
        
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = get_activation(activation)
    
    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        """Add positional embedding to tensor"""
        return tensor if pos is None else tensor + pos
    
    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        # Self-attention
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        
        q = k = self.with_pos_embed(src, pos_embed)
        src_attn, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = residual + self.dropout1(src_attn)
        
        if not self.normalize_before:
            src = self.norm1(src)
        
        # Feed forward network
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        
        src_ffn = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src_ffn)
        
        if not self.normalize_before:
            src = self.norm2(src)
        
        return src


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""
    
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        output = src
        
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output


@register()
class HybridEncoder(nn.Module):
    """
    Hybrid Encoder for RT-DETRv3
    Combines CNN feature extraction with Transformer self-attention
    """
    __share__ = ['eval_spatial_size']
    
    def __init__(self,
                 in_channels: List[int] = [512, 1024, 2048],
                 feat_strides: List[int] = [8, 16, 32],
                 hidden_dim: int = 256,
                 nhead: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.0,
                 enc_act: str = 'gelu',
                 use_encoder_idx: List[int] = [2],
                 num_encoder_layers: int = 1,
                 pe_temperature: float = 10000,
                 expansion: float = 1.0,
                 depth_mult: float = 1.0,
                 activation: str = 'silu',
                 eval_spatial_size: Optional[List[int]] = None,
                 norm_type: str = 'bn'):
        super().__init__()
        
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # Channel projection layers
        self.input_proj = nn.ModuleList()
        for in_ch in in_channels:
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm2d(hidden_dim) if norm_type == 'bn' else nn.SyncBatchNorm(hidden_dim))
            ]))
            self.input_proj.append(proj)
        
        # Transformer encoders
        if num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=enc_act
            )
            
            self.encoder = nn.ModuleList([
                TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
                for _ in range(len(use_encoder_idx))
            ])
        else:
            self.encoder = nn.ModuleList()
        
        # Top-down FPN
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 1, 1, activation=activation)
            )
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    expansion=expansion,
                    activation=activation
                )
            )
        
        # Bottom-up PAN
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, activation=activation)
            )
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    expansion=expansion,
                    activation=activation
                )
            )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize positional embeddings for evaluation"""
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature
                )
                self.register_buffer(f'pos_embed{idx}', pos_embed)
    
    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0
    ) -> torch.Tensor:
        """Build 2D sincos position embedding"""
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4'
        
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)
        
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        
        return torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]
    
    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass"""
        assert len(feats) == len(self.in_channels)
        
        # Project features to hidden dimension
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # Apply transformer encoders
        if self.num_encoder_layers > 0:
            for i, enc_idx in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_idx].shape[2:]
                # Flatten [B, C, H, W] to [B, H*W, C]
                src_flatten = proj_feats[enc_idx].flatten(2).permute(0, 2, 1)
                
                # Get positional embedding
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_idx}', None)
                    if pos_embed is not None:
                        pos_embed = pos_embed.to(src_flatten.device)
                
                # Apply transformer
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_idx] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
        
        # Top-down FPN
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            
            upsample_feat = F.interpolate(feat_high, scale_factor=2.0, mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)
        
        # Bottom-up PAN
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_high], dim=1))
            outs.append(out)
        
        return outs
    
    @property
    def out_shape(self):
        """Output shape specification"""
        return [{'channels': ch, 'stride': stride} 
                for ch, stride in zip(self.out_channels, self.out_strides)]