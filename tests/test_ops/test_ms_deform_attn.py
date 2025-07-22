"""
MS Deformable Attention单元测试

这个测试模块包含了MS Deformable Attention算子的全面测试，
包括正确性验证、梯度检查、边界条件测试等。

测试覆盖范围：
1. 基础功能测试 - 验证前向传播、不同参数设置
2. 正确性测试 - 对比CUDA实现与原生PyTorch实现
3. MSDeformableAttention模块测试 - 完整模块的功能测试
4. 梯度测试 - 验证反向传播和数值梯度
5. 边界条件和错误处理测试
6. 性能测试

运行测试:
    pytest tests/test_ops/test_ms_deform_attn.py -v
    python -m pytest tests/test_ops/test_ms_deform_attn.py::test_forward_basic -v
    python3 tests/test_ops/test_ms_deform_attn.py  # 直接运行

更新历史:
- 2025-07-22: 根据最新的MSDeformableAttention实现更新测试逻辑
- 优化了正确性验证函数，确保与layers.py中的实现完全一致
- 新增了完整的MSDeformableAttention模块测试
"""

import pytest
import torch
import numpy as np
import random
from typing import Tuple, List
import warnings
import torch.nn.functional as F

# 尝试导入我们的算子实现
try:
    import sys
    import os
    # 添加项目路径 - 采用与run_tests.py相同的策略
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    src_path = os.path.join(project_root, 'src')
    ops_path = os.path.join(src_path, 'ops')
    sys.path.insert(0, project_root)  # 添加项目根目录
    sys.path.insert(0, src_path)
    sys.path.insert(0, ops_path)
    
    # 临时切换到ops目录以便找到C++扩展
    original_cwd = os.getcwd()
    os.chdir(ops_path)
    
    from src.ops.ms_deform_attn import ms_deform_attn, MSDeformAttn, is_available
    from src.nn.transformer.layers import MSDeformableAttention
    MS_DEFORM_ATTN_AVAILABLE = is_available()
    
    # 恢复原始工作目录
    os.chdir(original_cwd)
except ImportError as e:
    MS_DEFORM_ATTN_AVAILABLE = False
    warnings.warn(f"MS Deformable Attention算子不可用: {e}")

# 跳过所有测试如果算子不可用
pytestmark = pytest.mark.skipif(
    not MS_DEFORM_ATTN_AVAILABLE or not torch.cuda.is_available(),
    reason="MS Deformable Attention算子未编译或CUDA不可用"
)


def set_random_seed(seed: int = 42):
    """设置随机种子以确保测试的可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 确保确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TestMSDeformAttnBasic:
    """基础功能测试"""
    
    def setup_method(self):
        """测试前的设置"""
        set_random_seed(42)
        self.device = 'cuda'
        self.dtype = torch.float32
        
        # 基本测试参数
        self.batch_size = 2
        self.num_heads = 8
        self.channels = 8
        self.query_length = 2
        self.num_levels = 2
        self.num_points = 2
        
    def create_test_tensors(self, channels: int = None) -> Tuple[torch.Tensor, ...]:
        """创建测试张量"""
        if channels is None:
            channels = self.channels
            
        # 空间形状定义
        spatial_shapes = torch.tensor([[6, 4], [3, 2]], dtype=torch.int64, device=self.device)
        level_start_index = torch.tensor([0, 24], dtype=torch.int64, device=self.device)
        value_length = sum(h * w for h, w in spatial_shapes.tolist())
        
        # 创建输入张量
        value = torch.randn(
            self.batch_size, value_length, self.num_heads, channels,
            dtype=self.dtype, device=self.device
        ) * 0.01
        
        sampling_locations = torch.rand(
            self.batch_size, self.query_length, self.num_heads, 
            self.num_levels, self.num_points, 2,
            dtype=self.dtype, device=self.device
        )
        
        attention_weights = torch.rand(
            self.batch_size, self.query_length, self.num_heads,
            self.num_levels, self.num_points,
            dtype=self.dtype, device=self.device
        ) + 1e-5
        
        # 归一化注意力权重
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        
        return (value, spatial_shapes, level_start_index, 
                sampling_locations, attention_weights)
    
    def test_forward_basic(self):
        """测试基本前向传播功能"""
        tensors = self.create_test_tensors()
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = tensors
        
        # 执行前向传播
        output = ms_deform_attn(value, spatial_shapes, level_start_index,
                               sampling_locations, attention_weights)
        
        # 验证输出形状
        expected_shape = (self.batch_size, self.query_length, 
                         self.num_heads * self.channels)
        assert output.shape == expected_shape, f"输出形状错误: 期望 {expected_shape}, 得到 {output.shape}"
        
        # 验证输出不包含NaN或Inf
        assert torch.isfinite(output).all(), "输出包含NaN或Inf值"
        
        # 验证输出设备和数据类型
        assert output.device.type == 'cuda', "输出设备错误"
        assert output.dtype == self.dtype, "输出数据类型错误"
    
    def test_module_interface(self):
        """测试nn.Module接口"""
        module = MSDeformAttn()
        module.to(self.device)
        
        tensors = self.create_test_tensors()
        output = module(*tensors)
        
        expected_shape = (self.batch_size, self.query_length, 
                         self.num_heads * self.channels)
        assert output.shape == expected_shape
        
    def test_different_channel_sizes(self):
        """测试不同的通道大小"""
        channel_sizes = [1, 4, 16, 32, 64, 128]
        
        for channels in channel_sizes:
            tensors = self.create_test_tensors(channels)
            output = ms_deform_attn(*tensors)
            
            expected_shape = (self.batch_size, self.query_length, 
                             self.num_heads * channels)
            assert output.shape == expected_shape, f"通道数 {channels} 测试失败"
    
    def test_different_batch_sizes(self):
        """测试不同的批次大小"""
        original_batch_size = self.batch_size
        
        batch_sizes = [1, 3, 5]
        for batch_size in batch_sizes:
            self.batch_size = batch_size
            tensors = self.create_test_tensors()
            output = ms_deform_attn(*tensors)
            
            expected_shape = (batch_size, self.query_length, 
                             self.num_heads * self.channels)
            assert output.shape == expected_shape, f"批次大小 {batch_size} 测试失败"
        
        self.batch_size = original_batch_size


class TestMSDeformAttnGradients:
    """梯度测试"""
    
    def setup_method(self):
        """测试前的设置"""
        set_random_seed(42)
        self.device = 'cuda'
        self.dtype = torch.float64  # 使用double精度进行数值稳定性
        
        # 小规模测试参数，便于数值梯度检查
        self.batch_size = 1
        self.num_heads = 2
        self.channels = 4
        self.query_length = 2
        self.num_levels = 1
        self.num_points = 2
    
    def create_test_tensors_for_gradcheck(self) -> Tuple[torch.Tensor, ...]:
        """为梯度检查创建小规模测试张量"""
        spatial_shapes = torch.tensor([[3, 3]], dtype=torch.int64, device=self.device)
        level_start_index = torch.tensor([0], dtype=torch.int64, device=self.device)
        value_length = 9  # 3*3
        
        value = torch.randn(
            self.batch_size, value_length, self.num_heads, self.channels,
            dtype=self.dtype, device=self.device, requires_grad=True
        ) * 0.01
        
        sampling_locations = torch.rand(
            self.batch_size, self.query_length, self.num_heads,
            self.num_levels, self.num_points, 2,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        
        attention_weights = torch.rand(
            self.batch_size, self.query_length, self.num_heads,
            self.num_levels, self.num_points,
            dtype=self.dtype, device=self.device, requires_grad=True
        ) + 1e-5
        
        # 归一化注意力权重
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        
        return (value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights)
    
    def test_gradient_existence(self):
        """测试梯度是否正确计算"""
        tensors = self.create_test_tensors_for_gradcheck()
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = tensors
        
        # 确保张量需要梯度并且是叶子张量
        value = value.detach().requires_grad_(True)
        sampling_locations = sampling_locations.detach().requires_grad_(True) 
        attention_weights = attention_weights.detach().requires_grad_(True)
        
        output = ms_deform_attn(value, spatial_shapes, level_start_index,
                               sampling_locations, attention_weights)
        
        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()
        
        # 检查梯度是否存在且有效
        assert value.grad is not None, "value张量梯度未计算"
        assert sampling_locations.grad is not None, "sampling_locations张量梯度未计算"
        assert attention_weights.grad is not None, "attention_weights张量梯度未计算"
        
        # 检查梯度不包含NaN
        assert torch.isfinite(value.grad).all(), "value梯度包含NaN"
        assert torch.isfinite(sampling_locations.grad).all(), "sampling_locations梯度包含NaN"
        assert torch.isfinite(attention_weights.grad).all(), "attention_weights梯度包含NaN"
    
    @pytest.mark.slow
    def test_numerical_gradient_check(self):
        """数值梯度检查 - 较慢的测试"""
        from torch.autograd import gradcheck
        
        tensors = self.create_test_tensors_for_gradcheck()
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = tensors
        
        # PyTorch gradcheck
        inputs = (value, spatial_shapes, level_start_index, 
                 sampling_locations, attention_weights)
        
        def func(*args):
            return ms_deform_attn(*args)
        
        # 使用较宽松的容差进行数值梯度检查
        check_result = gradcheck(
            func, inputs, eps=1e-6, atol=1e-4, rtol=1e-3, 
            raise_exception=False, nondet_tol=1e-5
        )
        
        assert check_result, "数值梯度检查失败"


class TestMSDeformAttnEdgeCases:
    """边界条件和错误处理测试"""
    
    def setup_method(self):
        """测试前的设置"""
        set_random_seed(42)
        self.device = 'cuda'
    
    def test_input_validation(self):
        """测试输入验证"""
        # 测试错误的张量维度
        with pytest.raises(ValueError, match="must be 4D tensor"):
            value = torch.randn(2, 10, 8)  # 3D而不是4D
            spatial_shapes = torch.tensor([[6, 4]], dtype=torch.int64)
            level_start_index = torch.tensor([0], dtype=torch.int64)
            sampling_locations = torch.rand(2, 2, 8, 1, 4, 2)
            attention_weights = torch.rand(2, 2, 8, 1, 4)
            
            ms_deform_attn(value, spatial_shapes, level_start_index,
                          sampling_locations, attention_weights)
    
    def test_device_mismatch(self):
        """测试设备不匹配的情况"""
        # CPU张量应该引发错误
        value = torch.randn(2, 10, 8, 8)  # CPU张量
        spatial_shapes = torch.tensor([[6, 4]], dtype=torch.int64)
        level_start_index = torch.tensor([0], dtype=torch.int64)
        sampling_locations = torch.rand(2, 2, 8, 1, 4, 2)
        attention_weights = torch.rand(2, 2, 8, 1, 4)
        
        with pytest.raises(ValueError, match="must be on CUDA device"):
            ms_deform_attn(value, spatial_shapes, level_start_index,
                          sampling_locations, attention_weights)
    
    def test_dtype_handling(self):
        """测试数据类型处理"""
        # 测试自动类型转换
        value = torch.randn(2, 10, 8, 8, device='cuda')
        spatial_shapes = torch.tensor([[6, 4]], dtype=torch.int32, device='cuda')  # int32
        level_start_index = torch.tensor([0], dtype=torch.int32, device='cuda')  # int32
        sampling_locations = torch.rand(2, 2, 8, 1, 4, 2, device='cuda')
        attention_weights = torch.rand(2, 2, 8, 1, 4, device='cuda')
        
        # 应该自动转换为int64
        output = ms_deform_attn(value, spatial_shapes, level_start_index,
                               sampling_locations, attention_weights)
        assert output.shape == (2, 2, 64)


class TestMSDeformAttnPerformance:
    """性能测试"""
    
    def setup_method(self):
        """测试前的设置"""
        set_random_seed(42)
        self.device = 'cuda'
        
    @pytest.mark.slow
    def test_large_scale_performance(self):
        """大规模数据的性能测试"""
        # 大规模参数
        batch_size = 4
        value_length = 1000
        num_heads = 8
        channels = 64
        query_length = 100
        num_levels = 4
        num_points = 4
        
        # 创建大规模张量
        value = torch.randn(batch_size, value_length, num_heads, channels, 
                           device=self.device) * 0.01
        spatial_shapes = torch.tensor([[32, 32], [16, 16], [8, 8], [4, 4]], 
                                    dtype=torch.int64, device=self.device)
        level_start_index = torch.tensor([0, 256, 512, 576], 
                                       dtype=torch.int64, device=self.device)
        sampling_locations = torch.rand(batch_size, query_length, num_heads,
                                      num_levels, num_points, 2, device=self.device)
        attention_weights = torch.rand(batch_size, query_length, num_heads,
                                     num_levels, num_points, device=self.device)
        
        # 归一化注意力权重
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        
        # 性能测试
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        output = ms_deform_attn(value, spatial_shapes, level_start_index,
                               sampling_locations, attention_weights)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"大规模前向传播时间: {elapsed_time:.2f}ms")
        
        # 验证输出
        expected_shape = (batch_size, query_length, num_heads * channels)
        assert output.shape == expected_shape
        assert torch.isfinite(output).all()

    @pytest.mark.slow  
    def test_performance_comparison(self):
        """性能对比测试：CUDA vs 原生PyTorch"""
        # 中等规模参数用于对比
        batch_size = 2
        num_heads = 8
        channels = 32
        query_length = 50
        num_levels = 3
        num_points = 4
        
        # 创建测试张量
        spatial_shapes = torch.tensor([[16, 16], [8, 8], [4, 4]], 
                                    dtype=torch.int64, device=self.device)
        level_start_index = torch.cat([
            torch.tensor([0], device=self.device),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ]).to(torch.int64)
        value_length = spatial_shapes.prod(1).sum().item()
        
        value = torch.randn(batch_size, value_length, num_heads, channels, 
                           device=self.device) * 0.01
        sampling_locations = torch.rand(batch_size, query_length, num_heads,
                                      num_levels, num_points, 2, device=self.device)
        attention_weights = torch.rand(batch_size, query_length, num_heads,
                                     num_levels, num_points, device=self.device)
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        
        tensors = (value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
        
        # 预热
        for _ in range(5):
            _ = ms_deform_attn(*tensors)
            _ = ms_deform_attn_naive(*tensors)
        
        torch.cuda.synchronize()
        
        # CUDA实现性能测试
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(10):
            output_cuda = ms_deform_attn(*tensors)
        end_time.record()
        
        torch.cuda.synchronize()
        cuda_time = start_time.elapsed_time(end_time) / 10
        
        # 原生PyTorch实现性能测试
        start_time.record()
        for _ in range(10):
            output_naive = ms_deform_attn_naive(*tensors)
        end_time.record()
        
        torch.cuda.synchronize()
        naive_time = start_time.elapsed_time(end_time) / 10
        
        print(f"\n性能对比结果:")
        print(f"CUDA实现平均时间: {cuda_time:.3f}ms")
        print(f"原生PyTorch实现平均时间: {naive_time:.3f}ms")
        print(f"加速比: {naive_time/cuda_time:.2f}x")
        
        # 验证正确性
        max_abs_err = (output_cuda - output_naive).abs().max().item()
        print(f"最大绝对误差: {max_abs_err:.2e}")
        
        assert torch.allclose(output_cuda, output_naive, rtol=1e-3, atol=1e-4), \
            "性能测试中CUDA和原生实现结果不一致"


def ms_deform_attn_naive(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor
) -> torch.Tensor:
    """
    Pytorch Naive Implementation of Multi-Scale Deformable Attention.
    This function replicates the exact logic from MSDeformableAttention._deformable_attention_core_func
    for correctness verification.
    """
    bs, _, n_heads, head_dim = value.shape
    _, len_q, _, n_levels, n_points, _ = sampling_locations.shape
    
    # 按层级分割value：[bs, len_v, n_heads, head_dim] -> List[[bs, H*W, n_heads, head_dim]]
    value_list = value.split([H * W for H, W in value_spatial_shapes], dim=1)
    
    # 准备输出累加器
    output_list = []
    
    # 逐层级处理
    for level, (H, W) in enumerate(value_spatial_shapes):
        H, W = int(H), int(W)
        
        # 获取当前层级的value: [bs, H*W, n_heads, head_dim]
        value_l = value_list[level]
        
        # 获取当前层级的采样位置: [bs, len_q, n_heads, n_points, 2]
        sampling_locations_l = sampling_locations[:, :, :, level, :, :]
        
        # 获取当前层级的注意力权重: [bs, len_q, n_heads, n_points]
        attention_weights_l = attention_weights[:, :, :, level, :]
        
        # 重塑value为适合grid_sample的格式
        # [bs, H*W, n_heads, head_dim] -> [bs*n_heads, head_dim, H, W]
        value_reshaped = value_l.permute(0, 2, 3, 1).reshape(bs * n_heads, head_dim, H, W)
        
        # 转换采样位置到grid_sample格式
        # [bs, len_q, n_heads, n_points, 2] -> [bs*n_heads, len_q*n_points, 2]
        sampling_grid = sampling_locations_l.permute(0, 2, 1, 3, 4)  # [bs, n_heads, len_q, n_points, 2]
        sampling_grid = sampling_grid.reshape(bs * n_heads, len_q * n_points, 2)
        sampling_grid = 2.0 * sampling_grid - 1.0  # 坐标变换：[0,1] -> [-1,1]
        
        # Grid sampling
        # 添加高度维度用于grid_sample: [bs*n_heads, len_q*n_points, 1, 2]
        sampled_values = F.grid_sample(
            value_reshaped, 
            sampling_grid.unsqueeze(-2),
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        ).squeeze(-1)  # 移除高度维度: [bs*n_heads, head_dim, len_q*n_points]
        
        # 重塑回原始格式: [bs, n_heads, head_dim, len_q, n_points]
        sampled_values = sampled_values.view(bs, n_heads, head_dim, len_q, n_points)
        
        # 应用注意力权重
        # attention_weights_l: [bs, len_q, n_heads, n_points] -> [bs, n_heads, len_q, n_points]
        attn_weights_reshaped = attention_weights_l.permute(0, 2, 1, 3)
        
        # 广播相乘并在points维度求和
        # [bs, n_heads, head_dim, len_q, n_points] * [bs, n_heads, 1, len_q, n_points]
        weighted_values = sampled_values * attn_weights_reshaped.unsqueeze(2)
        level_output = weighted_values.sum(dim=-1)  # [bs, n_heads, head_dim, len_q]
        
        output_list.append(level_output)
    
    # 合并所有层级的输出（在层级维度求和）
    total_output = torch.stack(output_list, dim=0).sum(dim=0)  # [bs, n_heads, head_dim, len_q]
    
    # 重排维度到期望格式: [bs, len_q, n_heads * head_dim]
    final_output = total_output.permute(0, 3, 1, 2).reshape(bs, len_q, n_heads * head_dim)
    
    return final_output


class TestMSDeformAttnCorrectness:
    """
    正确性测试，对比CUDA实现和原生PyTorch实现
    """

    def setup_method(self):
        """测试前的设置"""
        set_random_seed(42)
        self.device = 'cuda'
        self.dtype = torch.float32

        # 测试参数
        self.batch_size = 2
        self.num_heads = 4
        self.channels = 8
        self.query_length = 10
        self.num_levels = 2
        self.num_points = 2

    def create_test_tensors(self, channels: int = None) -> Tuple[torch.Tensor, ...]:
        """创建测试张量"""
        if channels is None:
            channels = self.channels

        spatial_shapes = torch.tensor([[12, 10], [6, 5]], dtype=torch.int64, device=self.device)
        level_start_index = torch.cat([
            torch.tensor([0], device=self.device),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ]).to(torch.int64)
        value_length = sum(h * w for h, w in spatial_shapes.tolist())

        value = torch.randn(
            self.batch_size, value_length, self.num_heads, channels,
            dtype=self.dtype, device=self.device
        )

        sampling_locations = torch.rand(
            self.batch_size, self.query_length, self.num_heads,
            self.num_levels, self.num_points, 2,
            dtype=self.dtype, device=self.device
        )

        attention_weights = torch.rand(
            self.batch_size, self.query_length, self.num_heads,
            self.num_levels, self.num_points,
            dtype=self.dtype, device=self.device
        ) + 1e-5

        attention_weights = attention_weights / attention_weights.sum(
            dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

        return (value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights)

    def test_forward_correctness(self):
        """测试前向传播的正确性"""
        tensors = self.create_test_tensors()
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = tensors

        # CUDA实现
        output_cuda = ms_deform_attn(*tensors)

        # 原生PyTorch实现
        output_naive = ms_deform_attn_naive(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights
        )

        # 比较结果
        max_abs_err = (output_cuda - output_naive).abs().max().item()
        max_rel_err = ((output_cuda - output_naive).abs() / output_naive.abs().clamp(1e-6)).max().item()

        print(f"\nCUDA vs Naive PyTorch Forward: max_abs_err={max_abs_err:.2e}, max_rel_err={max_rel_err:.2e}")

        assert torch.allclose(output_cuda, output_naive, rtol=1e-3, atol=1e-4), \
            f"CUDA实现与原生PyTorch实现的前向结果不一致 (max_abs_err={max_abs_err:.2e}, max_rel_err={max_rel_err:.2e})"


class TestMSDeformableAttentionModule:
    """
    测试完整的MSDeformableAttention模块
    """

    def setup_method(self):
        """测试前的设置"""
        set_random_seed(42)
        self.device = 'cuda'
        self.dtype = torch.float32

        # 测试参数
        self.batch_size = 2
        self.d_model = 256
        self.n_heads = 8
        self.n_levels = 4
        self.n_points = 4
        self.len_q = 100
        self.len_v = 1000

    def create_module_test_tensors(self) -> Tuple[torch.Tensor, ...]:
        """创建模块测试张量"""
        # 空间形状定义（4个层级）
        spatial_shapes = torch.tensor([[32, 32], [16, 16], [8, 8], [4, 4]], 
                                    dtype=torch.int64, device=self.device)
        level_start_index = torch.cat([
            torch.tensor([0], device=self.device),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ]).to(torch.int64)

        # 计算总的值长度
        value_length = spatial_shapes.prod(1).sum().item()

        # 创建输入张量
        query = torch.randn(self.batch_size, self.len_q, self.d_model, 
                           dtype=self.dtype, device=self.device)
        
        value = torch.randn(self.batch_size, value_length, self.d_model, 
                           dtype=self.dtype, device=self.device)
        
        # 参考点 (归一化坐标)
        reference_points = torch.rand(self.batch_size, self.len_q, self.n_levels, 2,
                                    dtype=self.dtype, device=self.device)

        return (query, reference_points, value, spatial_shapes, level_start_index)

    def test_module_forward(self):
        """测试模块前向传播"""
        module = MSDeformableAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_levels=self.n_levels,
            n_points=self.n_points
        ).to(self.device)

        tensors = self.create_module_test_tensors()
        query, reference_points, value, spatial_shapes, level_start_index = tensors

        # 前向传播
        output = module(query, reference_points, value, spatial_shapes, level_start_index)

        # 验证输出
        expected_shape = (self.batch_size, self.len_q, self.d_model)
        assert output.shape == expected_shape, f"输出形状错误: 期望 {expected_shape}, 得到 {output.shape}"
        assert torch.isfinite(output).all(), "输出包含NaN或Inf值"
        assert output.device.type == 'cuda', "输出设备错误"
        assert output.dtype == self.dtype, "输出数据类型错误"

    def test_module_gradient(self):
        """测试模块梯度"""
        module = MSDeformableAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_levels=self.n_levels,
            n_points=self.n_points
        ).to(self.device)

        tensors = self.create_module_test_tensors()
        query, reference_points, value, spatial_shapes, level_start_index = tensors
        
        # 设置需要梯度
        query.requires_grad_(True)
        value.requires_grad_(True)
        reference_points.requires_grad_(True)

        # 前向传播和反向传播
        output = module(query, reference_points, value, spatial_shapes, level_start_index)
        loss = output.sum()
        loss.backward()

        # 检查梯度
        assert query.grad is not None, "query梯度未计算"
        assert value.grad is not None, "value梯度未计算"
        assert reference_points.grad is not None, "reference_points梯度未计算"
        
        assert torch.isfinite(query.grad).all(), "query梯度包含NaN"
        assert torch.isfinite(value.grad).all(), "value梯度包含NaN"
        assert torch.isfinite(reference_points.grad).all(), "reference_points梯度包含NaN"

    def test_different_reference_point_formats(self):
        """测试不同格式的参考点"""
        module = MSDeformableAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_levels=self.n_levels,
            n_points=self.n_points
        ).to(self.device)

        query, _, value, spatial_shapes, level_start_index = self.create_module_test_tensors()

        # 测试2D参考点 [bs, len_q, n_levels, 2]
        reference_points_2d = torch.rand(self.batch_size, self.len_q, self.n_levels, 2,
                                        dtype=self.dtype, device=self.device)
        output_2d = module(query, reference_points_2d, value, spatial_shapes, level_start_index)
        assert output_2d.shape == (self.batch_size, self.len_q, self.d_model)

        # 测试4D参考点 [bs, len_q, n_levels, 4]
        reference_points_4d = torch.rand(self.batch_size, self.len_q, self.n_levels, 4,
                                        dtype=self.dtype, device=self.device)
        output_4d = module(query, reference_points_4d, value, spatial_shapes, level_start_index)
        assert output_4d.shape == (self.batch_size, self.len_q, self.d_model)

    def test_value_mask(self):
        """测试值掩码"""
        module = MSDeformableAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_levels=self.n_levels,
            n_points=self.n_points
        ).to(self.device)

        tensors = self.create_module_test_tensors()
        query, reference_points, value, spatial_shapes, level_start_index = tensors

        # 创建掩码
        value_mask = torch.rand(self.batch_size, value.shape[1], device=self.device) > 0.5

        # 带掩码的前向传播
        output_masked = module(query, reference_points, value, spatial_shapes, 
                             level_start_index, value_mask)
        
        # 不带掩码的前向传播
        output_unmasked = module(query, reference_points, value, spatial_shapes, 
                               level_start_index)

        # 验证掩码影响了输出
        assert not torch.allclose(output_masked, output_unmasked, rtol=1e-3), \
            "掩码应该影响输出结果"


# 测试运行器
if __name__ == "__main__":
    import sys
    
    # 检查环境
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过测试")
        exit(0)
    
    if not MS_DEFORM_ATTN_AVAILABLE:
        print("MS Deformable Attention算子不可用，请先编译扩展")
        exit(1)
    
    # 检查是否运行性能测试
    run_performance = "--performance" in sys.argv or "-p" in sys.argv
    
    print("="*60)
    print("MS Deformable Attention 测试套件")
    print("="*60)
    
    # 运行基础功能测试
    print("\n1. 运行基础功能测试...")
    test_basic = TestMSDeformAttnBasic()
    test_basic.setup_method()
    test_basic.test_forward_basic()
    test_basic.test_module_interface()
    test_basic.test_different_channel_sizes()
    test_basic.test_different_batch_sizes()
    print("✓ 基础功能测试通过")
    
    # 运行正确性测试
    print("\n2. 运行正确性测试...")
    test_correctness = TestMSDeformAttnCorrectness()
    test_correctness.setup_method()
    test_correctness.test_forward_correctness()
    print("✓ 正确性测试通过")

    # 运行MSDeformableAttention模块测试
    print("\n3. 运行MSDeformableAttention模块测试...")
    test_module = TestMSDeformableAttentionModule()
    test_module.setup_method()
    test_module.test_module_forward()
    test_module.test_module_gradient()
    test_module.test_different_reference_point_formats()
    test_module.test_value_mask()
    print("✓ MSDeformableAttention模块测试通过")

    # 运行梯度测试
    print("\n4. 运行梯度测试...")
    test_grad = TestMSDeformAttnGradients()
    test_grad.setup_method()
    test_grad.test_gradient_existence()
    print("✓ 梯度测试通过")
    
    # 运行边界条件测试
    print("\n5. 运行边界条件测试...")
    test_edge = TestMSDeformAttnEdgeCases()
    test_edge.setup_method()
    test_edge.test_dtype_handling()
    print("✓ 边界条件测试通过")
    
    # 性能测试（可选）
    if run_performance:
        print("\n6. 运行性能测试...")
        test_perf = TestMSDeformAttnPerformance()
        test_perf.setup_method()
        test_perf.test_large_scale_performance()
        test_perf.test_performance_comparison()
        print("✓ 性能测试通过")
    else:
        print("\n6. 跳过性能测试（使用 --performance 或 -p 参数启用）")
    
    print("\n" + "="*60)
    print("所有测试通过! ✓")
    print("="*60)
    
    if not run_performance:
        print("\n提示: 使用 'python3 tests/test_ops/test_ms_deform_attn.py --performance' 运行性能测试")