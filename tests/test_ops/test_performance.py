"""
MS Deformable Attention性能基准测试

这个模块专门用于测试MS Deformable Attention算子的性能，
包括执行时间、内存使用和与参考实现的对比。

运行测试:
    python tests/test_ops/test_performance.py
    pytest tests/test_ops/test_performance.py -v -s
"""

import time
import gc
import json
from typing import Dict, List, Tuple, Optional
import warnings

import torch
import torch.nn.functional as F
import numpy as np

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


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self, device: str = 'cuda', warmup_iterations: int = 10, 
                 benchmark_iterations: int = 100):
        """
        初始化基准测试
        
        Args:
            device: 测试设备
            warmup_iterations: 预热迭代次数
            benchmark_iterations: 基准测试迭代次数
        """
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results = []
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，无法进行性能测试")
        
        if not MS_DEFORM_ATTN_AVAILABLE:
            raise RuntimeError("MS Deformable Attention算子不可用")
    
    def create_test_data(self, config: Dict) -> Tuple[torch.Tensor, ...]:
        """
        创建测试数据
        
        Args:
            config: 测试配置字典
            
        Returns:
            测试张量元组
        """
        batch_size = config['batch_size']
        num_heads = config['num_heads']
        channels = config['channels']
        query_length = config['query_length']
        num_levels = config['num_levels']
        num_points = config['num_points']
        spatial_shapes_list = config['spatial_shapes']
        
        # 创建空间形状张量
        spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.int64, device=self.device)
        
        # 计算level start index
        level_start_index = [0]
        for h, w in spatial_shapes_list[:-1]:
            level_start_index.append(level_start_index[-1] + h * w)
        level_start_index = torch.tensor(level_start_index, dtype=torch.int64, device=self.device)
        
        # 计算总的value length
        value_length = sum(h * w for h, w in spatial_shapes_list)
        
        # 创建输入张量
        value = torch.randn(
            batch_size, value_length, num_heads, channels,
            dtype=torch.float32, device=self.device
        ) * 0.01
        
        sampling_locations = torch.rand(
            batch_size, query_length, num_heads, num_levels, num_points, 2,
            dtype=torch.float32, device=self.device
        )
        
        attention_weights = torch.rand(
            batch_size, query_length, num_heads, num_levels, num_points,
            dtype=torch.float32, device=self.device
        ) + 1e-5
        
        # 归一化注意力权重
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        
        return (value, spatial_shapes, level_start_index, 
                sampling_locations, attention_weights)
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """测量GPU内存使用情况"""
        torch.cuda.synchronize()
        memory_stats = torch.cuda.memory_stats(self.device)
        
        return {
            'allocated_mb': torch.cuda.memory_allocated(self.device) / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved(self.device) / 1024**2,
            'max_allocated_mb': memory_stats.get('allocated_bytes.all.peak', 0) / 1024**2,
            'max_reserved_mb': memory_stats.get('reserved_bytes.all.peak', 0) / 1024**2
        }
    
    def benchmark_forward(self, config: Dict) -> Dict:
        """
        前向传播性能基准测试
        
        Args:
            config: 测试配置
            
        Returns:
            性能测试结果
        """
        print(f"测试配置: {config}")
        
        # 创建测试数据
        test_data = self.create_test_data(config)
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        
        # 记录初始内存
        initial_memory = self.measure_memory_usage()
        
        # 预热
        print("预热中...")
        for _ in range(self.warmup_iterations):
            output = ms_deform_attn(*test_data)
            del output
        
        torch.cuda.synchronize()
        
        # 基准测试
        print("开始基准测试...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        
        for i in range(self.benchmark_iterations):
            start_event.record()
            output = ms_deform_attn(*test_data)
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event)
            times.append(elapsed_time)
            
            if i % 20 == 0:
                print(f"完成 {i+1}/{self.benchmark_iterations} 次迭代")
            
            del output
        
        # 记录峰值内存
        peak_memory = self.measure_memory_usage()
        
        # 计算统计信息
        times = np.array(times)
        result = {
            'config': config,
            'times_ms': {
                'mean': float(np.mean(times)),
                'std': float(np.std(times)),
                'min': float(np.min(times)),
                'max': float(np.max(times)),
                'median': float(np.median(times)),
                'p95': float(np.percentile(times, 95)),
                'p99': float(np.percentile(times, 99))
            },
            'memory_mb': {
                'initial': initial_memory,
                'peak': peak_memory,
                'peak_allocated': peak_memory['max_allocated_mb'] - initial_memory['allocated_mb']
            },
            'throughput': {
                'items_per_second': config['batch_size'] * 1000.0 / np.mean(times),
                'queries_per_second': config['batch_size'] * config['query_length'] * 1000.0 / np.mean(times)
            }
        }
        
        return result
    
    def benchmark_backward(self, config: Dict) -> Dict:
        """
        反向传播性能基准测试
        
        Args:
            config: 测试配置
            
        Returns:
            性能测试结果
        """
        print(f"反向传播测试配置: {config}")
        
        # 创建测试数据（需要梯度）
        test_data = self.create_test_data(config)
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = test_data
        
        # 设置需要梯度
        value.requires_grad_(True)
        sampling_locations.requires_grad_(True)
        attention_weights.requires_grad_(True)
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        
        # 预热
        for _ in range(self.warmup_iterations):
            output = ms_deform_attn(value, spatial_shapes, level_start_index,
                                   sampling_locations, attention_weights)
            loss = output.sum()
            loss.backward()
            
            # 清零梯度
            value.grad = None
            sampling_locations.grad = None  
            attention_weights.grad = None
        
        torch.cuda.synchronize()
        
        # 基准测试
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        
        for i in range(self.benchmark_iterations):
            # 清零梯度
            value.grad = None
            sampling_locations.grad = None
            attention_weights.grad = None
            
            # 前向传播
            output = ms_deform_attn(value, spatial_shapes, level_start_index,
                                   sampling_locations, attention_weights)
            loss = output.sum()
            
            # 测量反向传播时间
            start_event.record()
            loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event)
            times.append(elapsed_time)
            
            if i % 20 == 0:
                print(f"完成反向传播 {i+1}/{self.benchmark_iterations} 次迭代")
        
        # 计算统计信息
        times = np.array(times)
        result = {
            'config': config,
            'backward_times_ms': {
                'mean': float(np.mean(times)),
                'std': float(np.std(times)),
                'min': float(np.min(times)),
                'max': float(np.max(times)),
                'median': float(np.median(times))
            }
        }
        
        return result
    
    def run_comprehensive_benchmark(self) -> Dict:
        """运行全面的性能基准测试"""
        print("=" * 60)
        print("MS Deformable Attention 性能基准测试")
        print("=" * 60)
        
        # 不同规模的测试配置
        test_configs = [
            # 小规模测试
            {
                'name': 'small_scale',
                'batch_size': 1,
                'num_heads': 8,
                'channels': 32,
                'query_length': 100,
                'num_levels': 4,
                'num_points': 4,
                'spatial_shapes': [[64, 64], [32, 32], [16, 16], [8, 8]]
            },
            # 中等规模测试
            {
                'name': 'medium_scale', 
                'batch_size': 2,
                'num_heads': 8,
                'channels': 64,
                'query_length': 300,
                'num_levels': 4,
                'num_points': 4,
                'spatial_shapes': [[100, 100], [50, 50], [25, 25], [13, 13]]
            },
            # 大规模测试
            {
                'name': 'large_scale',
                'batch_size': 4,
                'num_heads': 8,
                'channels': 64,
                'query_length': 900,
                'num_levels': 4,
                'num_points': 4,
                'spatial_shapes': [[120, 120], [60, 60], [30, 30], [15, 15]]
            }
        ]
        
        all_results = {}
        
        for config in test_configs:
            print(f"\n测试 {config['name']}...")
            
            try:
                # 前向传播测试
                forward_result = self.benchmark_forward(config)
                
                # 反向传播测试
                backward_result = self.benchmark_backward(config)
                
                # 合并结果
                result = {**forward_result, **backward_result}
                all_results[config['name']] = result
                
                # 打印结果摘要
                self.print_result_summary(config['name'], result)
                
            except Exception as e:
                print(f"测试 {config['name']} 失败: {e}")
                all_results[config['name']] = {'error': str(e)}
        
        return all_results
    
    def print_result_summary(self, name: str, result: Dict):
        """打印测试结果摘要"""
        if 'error' in result:
            print(f"{name}: 错误 - {result['error']}")
            return
        
        times = result['times_ms']
        memory = result['memory_mb']
        throughput = result['throughput']
        
        print(f"\n{name} 结果:")
        print(f"  前向传播时间: {times['mean']:.2f}±{times['std']:.2f}ms "
              f"(中位数: {times['median']:.2f}ms)")
        print(f"  内存使用: 峰值分配 {memory['peak_allocated']:.1f}MB")
        print(f"  吞吐量: {throughput['items_per_second']:.1f} items/sec, "
              f"{throughput['queries_per_second']:.1f} queries/sec")
        
        if 'backward_times_ms' in result:
            backward_times = result['backward_times_ms']
            print(f"  反向传播时间: {backward_times['mean']:.2f}±{backward_times['std']:.2f}ms")
    
    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """保存测试结果到文件"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"结果已保存到 {filename}")


def test_accuracy_vs_reference():
    """测试与参考实现的数值精度 (Paddle CUDA, PyTorch CUDA, PyTorch Naive)"""
    print("\n数值精度测试: Paddle CUDA vs PyTorch CUDA vs PyTorch Naive...")

    # 1. 加载Paddle测试数据
    paddle_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'ext_op', 'paddle_test_data')

    if not os.path.exists(paddle_data_dir):
        pytest.skip(f"Paddle测试数据目录不存在: {paddle_data_dir}\n"
                    "请先运行 'SAVE_TENSORS_FOR_TEST=1 python examples/ext_op/test_ms_deformable_attn_op.py' 生成测试数据。")

    device = 'cuda'

    try:
        value = torch.from_numpy(np.load(os.path.join(paddle_data_dir, 'value.npy'))).to(device)
        spatial_shapes = torch.from_numpy(np.load(os.path.join(paddle_data_dir, 'spatial_shapes.npy'))).to(device)
        level_start_index = torch.from_numpy(np.load(os.path.join(paddle_data_dir, 'level_start_index.npy'))).to(device)
        sampling_locations = torch.from_numpy(np.load(os.path.join(paddle_data_dir, 'sampling_locations.npy'))).to(device)
        attention_weights = torch.from_numpy(np.load(os.path.join(paddle_data_dir, 'attention_weights.npy'))).to(device)
        output_paddle_cuda = torch.from_numpy(np.load(os.path.join(paddle_data_dir, 'output_cuda.npy'))).to(device)
    except FileNotFoundError as e:
        pytest.fail(f"加载Paddle测试数据失败: {e}. 请确保已生成测试数据。")

    # 2. 运行PyTorch CUDA OP
    output_pytorch_cuda = ms_deform_attn(
        value, spatial_shapes, level_start_index,
        sampling_locations, attention_weights
    )

    # 3. 运行PyTorch Naive OP
    bs, _, n_heads, c = value.shape
    naive_attn = MSDeformableAttention(
        d_model=n_heads * c,
        n_heads=n_heads,
        n_levels=sampling_locations.shape[3],
        n_points=sampling_locations.shape[4]
    ).to(device)
    output_pytorch_naive = naive_attn._deformable_attention_core_func(
        value, spatial_shapes, level_start_index,
        sampling_locations, attention_weights
    )

    # 4. 比较结果
    # PyTorch CUDA vs Paddle CUDA
    pt_cuda_vs_pd_cuda_abs_err = (output_pytorch_cuda - output_paddle_cuda).abs().max().item()
    pt_cuda_vs_pd_cuda_rel_err = ((output_pytorch_cuda - output_paddle_cuda).abs() / output_paddle_cuda.abs().clamp(1e-6)).max().item()
    print(f"PyTorch CUDA vs Paddle CUDA: max_abs_err={pt_cuda_vs_pd_cuda_abs_err:.2e}, max_rel_err={pt_cuda_vs_pd_cuda_rel_err:.2e}")
    assert torch.allclose(output_pytorch_cuda, output_paddle_cuda, rtol=1e-4, atol=1e-5), \
        "PyTorch CUDA OP与Paddle CUDA OP的输出不一致"

    # PyTorch CUDA vs PyTorch Naive
    pt_cuda_vs_pt_naive_abs_err = (output_pytorch_cuda - output_pytorch_naive).abs().max().item()
    pt_cuda_vs_pt_naive_rel_err = ((output_pytorch_cuda - output_pytorch_naive).abs() / output_pytorch_naive.abs().clamp(1e-6)).max().item()
    print(f"PyTorch CUDA vs PyTorch Naive: max_abs_err={pt_cuda_vs_pt_naive_abs_err:.2e}, max_rel_err={pt_cuda_vs_pt_naive_rel_err:.2e}")
    assert torch.allclose(output_pytorch_cuda, output_pytorch_naive, rtol=1e-4, atol=1e-5), \
        "PyTorch CUDA OP与PyTorch Naive OP的输出不一致"

    print("✓ 数值精度对比测试通过")


def main():
    """主测试函数"""
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过性能测试")
        return
    
    if not MS_DEFORM_ATTN_AVAILABLE:
        print("MS Deformable Attention算子不可用，请先编译扩展")
        return
    
    try:
        # 运行数值精度测试
        test_accuracy_vs_reference()
        
        # 运行性能基准测试
        benchmark = PerformanceBenchmark()
        results = benchmark.run_comprehensive_benchmark()
        
        # 保存结果
        benchmark.save_results(results)
        
        print("\n" + "=" * 60)
        print("所有性能测试完成! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"性能测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()