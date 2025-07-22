"""
多尺度可变形注意力PyTorch实现

这个模块提供了多尺度可变形注意力机制的高性能CUDA实现，
支持PyTorch的自动求导系统。

示例用法:
    import torch
    from src.ops.ms_deform_attn import ms_deform_attn
    
    # 准备输入张量
    value = torch.randn(2, 100, 8, 32, device='cuda')
    spatial_shapes = torch.tensor([[10, 10]], dtype=torch.int64, device='cuda')
    level_start_index = torch.tensor([0], dtype=torch.int64, device='cuda')
    sampling_locations = torch.rand(2, 50, 8, 1, 4, 2, device='cuda')
    attention_weights = torch.rand(2, 50, 8, 1, 4, device='cuda')
    
    # 执行多尺度可变形注意力
    output = ms_deform_attn(value, spatial_shapes, level_start_index,
                           sampling_locations, attention_weights)
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Tuple, Optional
import warnings

# 尝试导入编译好的C++扩展
try:
    import ms_deform_attn_ext
    _C_EXT_AVAILABLE = True
except ImportError as e:
    _C_EXT_AVAILABLE = False
    warnings.warn(
        f"MS Deformable Attention C++扩展未找到: {e}\n"
        "请确保已编译扩展: cd src/ops && python setup.py build_ext --inplace",
        ImportWarning
    )


class MSDeformAttnFunction(Function):
    """
    多尺度可变形注意力的PyTorch自动求导函数
    
    这个类实现了多尺度可变形注意力的前向和反向传播，
    支持PyTorch的自动求导系统。
    """
    
    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index, 
                sampling_locations, attention_weights):
        """
        前向传播
        
        Args:
            ctx: PyTorch上下文对象，用于保存反向传播所需的张量
            value: [B, L, H, C] 特征值张量
            spatial_shapes: [Levels, 2] 每层的空间形状 [height, width]
            level_start_index: [Levels] 每层在value中的起始索引
            sampling_locations: [B, Q, H, Levels, Points, 2] 归一化采样位置 [0,1]
            attention_weights: [B, Q, H, Levels, Points] 注意力权重
            
        Returns:
            output: [B, Q, H*C] 输出特征张量
        """
        # 检查C++扩展是否可用
        if not _C_EXT_AVAILABLE:
            raise RuntimeError(
                "MS Deformable Attention C++扩展未编译。"
                "请运行: cd src/ops && python setup.py build_ext --inplace"
            )
        
        # 保存张量用于反向传播
        ctx.save_for_backward(value, spatial_shapes, level_start_index, 
                              sampling_locations, attention_weights)
        
        # 调用C++前向传播实现
        output = ms_deform_attn_ext.forward(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights
        )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        
        Args:
            ctx: PyTorch上下文对象
            grad_output: [B, Q, H*C] 输出梯度
            
        Returns:
            tuple: 各输入张量的梯度
                - grad_value: [B, L, H, C] 
                - grad_spatial_shapes: None (不需要梯度)
                - grad_level_start_index: None (不需要梯度)
                - grad_sampling_locations: [B, Q, H, Levels, Points, 2]
                - grad_attention_weights: [B, Q, H, Levels, Points]
        """
        # 获取保存的张量
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        
        # 确保grad_output是连续的
        grad_output = grad_output.contiguous()
        
        # 调用C++反向传播实现
        grads = ms_deform_attn_ext.backward(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, grad_output
        )
        
        # 返回各输入的梯度
        # spatial_shapes和level_start_index通常不需要梯度
        grad_value, grad_spatial_shapes, grad_level_start_index, grad_sampling_locations, grad_attention_weights = grads
        
        return grad_value, None, None, grad_sampling_locations, grad_attention_weights


def ms_deform_attn(value: torch.Tensor,
                   spatial_shapes: torch.Tensor,
                   level_start_index: torch.Tensor,
                   sampling_locations: torch.Tensor,
                   attention_weights: torch.Tensor) -> torch.Tensor:
    """
    多尺度可变形注意力机制
    
    这是用户友好的接口函数，提供了输入验证和类型检查。
    
    Args:
        value: [B, L, H, C] 特征值张量
            - B: batch size
            - L: 所有层级特征点的总数
            - H: attention heads数量  
            - C: 每个head的channel数
        spatial_shapes: [Levels, 2] 每层的空间形状
            - Levels: 特征层级数量
            - 2: [height, width] 
        level_start_index: [Levels] 每层在value中的起始索引
        sampling_locations: [B, Q, H, Levels, Points, 2] 归一化采样位置
            - Q: query长度
            - Points: 每个query的采样点数量
            - 2: [x, y] 坐标，范围[0,1]
        attention_weights: [B, Q, H, Levels, Points] 注意力权重
    
    Returns:
        output: [B, Q, H*C] 聚合后的特征张量
        
    Raises:
        RuntimeError: 如果C++扩展未编译
        ValueError: 如果输入张量形状不匹配
        TypeError: 如果输入张量类型不正确
        
    Example:
        >>> # 创建测试输入
        >>> B, H, C = 2, 8, 32
        >>> value = torch.randn(B, 100, H, C, device='cuda')
        >>> spatial_shapes = torch.tensor([[10, 10]], dtype=torch.int64, device='cuda')
        >>> level_start_index = torch.tensor([0], dtype=torch.int64, device='cuda')
        >>> sampling_locations = torch.rand(B, 50, H, 1, 4, 2, device='cuda')
        >>> attention_weights = torch.rand(B, 50, H, 1, 4, device='cuda')
        >>> 
        >>> # 执行注意力计算
        >>> output = ms_deform_attn(value, spatial_shapes, level_start_index,
        ...                        sampling_locations, attention_weights)
        >>> print(output.shape)  # torch.Size([2, 50, 256])
    """
    
    # 输入类型检查
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"value must be torch.Tensor, got {type(value)}")
    if not isinstance(spatial_shapes, torch.Tensor):
        raise TypeError(f"spatial_shapes must be torch.Tensor, got {type(spatial_shapes)}")
    if not isinstance(level_start_index, torch.Tensor):
        raise TypeError(f"level_start_index must be torch.Tensor, got {type(level_start_index)}")
    if not isinstance(sampling_locations, torch.Tensor):
        raise TypeError(f"sampling_locations must be torch.Tensor, got {type(sampling_locations)}")
    if not isinstance(attention_weights, torch.Tensor):
        raise TypeError(f"attention_weights must be torch.Tensor, got {type(attention_weights)}")
    
    # 设备检查
    if not value.is_cuda:
        raise ValueError("All tensors must be on CUDA device")
    
    # 数据类型检查
    if spatial_shapes.dtype != torch.int64:
        spatial_shapes = spatial_shapes.to(torch.int64)
    if level_start_index.dtype != torch.int64:
        level_start_index = level_start_index.to(torch.int64)
    
    # 基本形状检查
    if value.dim() != 4:
        raise ValueError(f"value must be 4D tensor [B,L,H,C], got {value.shape}")
    if spatial_shapes.dim() != 2 or spatial_shapes.size(1) != 2:
        raise ValueError(f"spatial_shapes must be [Levels,2] tensor, got {spatial_shapes.shape}")
    if sampling_locations.dim() != 6 or sampling_locations.size(-1) != 2:
        raise ValueError(f"sampling_locations must be [B,Q,H,Levels,Points,2] tensor, got {sampling_locations.shape}")
    if attention_weights.dim() != 5:
        raise ValueError(f"attention_weights must be [B,Q,H,Levels,Points] tensor, got {attention_weights.shape}")
    
    # 维度一致性检查
    B, L, H, C = value.shape
    Q = sampling_locations.size(1)
    Levels = spatial_shapes.size(0)
    Points = sampling_locations.size(4)
    
    if sampling_locations.shape != (B, Q, H, Levels, Points, 2):
        raise ValueError(f"sampling_locations shape mismatch: expected ({B},{Q},{H},{Levels},{Points},2), got {sampling_locations.shape}")
    if attention_weights.shape != (B, Q, H, Levels, Points):
        raise ValueError(f"attention_weights shape mismatch: expected ({B},{Q},{H},{Levels},{Points}), got {attention_weights.shape}")
    if level_start_index.size(0) != Levels:
        raise ValueError(f"level_start_index size mismatch: expected {Levels}, got {level_start_index.size(0)}")
    
    # 调用自动求导函数
    return MSDeformAttnFunction.apply(
        value, spatial_shapes, level_start_index,
        sampling_locations, attention_weights
    )


class MSDeformAttn(nn.Module):
    """
    多尺度可变形注意力nn.Module包装器
    
    这个类提供了一个标准的PyTorch模块接口，方便集成到更大的模型中。
    """
    
    def __init__(self):
        """初始化模块"""
        super().__init__()
    
    def forward(self, value: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数和返回值与ms_deform_attn函数相同
        """
        return ms_deform_attn(value, spatial_shapes, level_start_index,
                             sampling_locations, attention_weights)
    
    def extra_repr(self) -> str:
        """返回模块的额外描述信息"""
        return "MS Deformable Attention (CUDA)"


def is_available() -> bool:
    """
    检查MS Deformable Attention算子是否可用
    
    Returns:
        bool: 如果C++扩展已编译并可用返回True，否则返回False
    """
    return _C_EXT_AVAILABLE


__all__ = [
    'ms_deform_attn',
    'MSDeformAttn', 
    'MSDeformAttnFunction',
    'is_available'
]