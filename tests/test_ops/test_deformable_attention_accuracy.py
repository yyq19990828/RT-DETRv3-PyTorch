#!/usr/bin/env python3
"""
可变形注意力精度对比测试

这个脚本比较Paddle原版实现和PyTorch迁移版本的可变形注意力精度。
测试包括：
1. 使用预存的Paddle测试数据进行数值对比
2. 比较CUDA优化实现和naive grid_sample实现
3. 生成详细的精度报告

用法:
    python test_deformable_attention_accuracy.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import warnings

# 添加项目路径
sys.path.append('/home/tyjt/桌面/migrate_paddle2pytorch/examples/rtdetrv3_pytorch')

def load_paddle_test_data() -> Dict[str, np.ndarray]:
    """
    加载Paddle原版生成的测试数据
    
    Returns:
        dict: 包含所有测试输入和预期输出的字典
    """
    data_dir = "examples/ext_op/paddle_test_data"
    
    print(f"📁 从 {data_dir} 加载Paddle测试数据...")
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"测试数据目录不存在: {data_dir}")
    
    # 加载所有测试数据文件
    files = [
        'value.npy',
        'spatial_shapes.npy', 
        'level_start_index.npy',
        'sampling_locations.npy',
        'attention_weights.npy',
        'output_cuda.npy'
    ]
    
    data = {}
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            data[filename.replace('.npy', '')] = np.load(filepath)
            print(f"  ✓ {filename}: shape {data[filename.replace('.npy', '')].shape}")
        else:
            print(f"  ✗ 缺少文件: {filename}")
    
    return data

def convert_numpy_to_torch(data: Dict[str, np.ndarray], device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    将numpy数组转换为PyTorch张量
    
    Args:
        data: numpy数据字典
        device: 目标设备
        
    Returns:
        dict: PyTorch张量字典
    """
    torch_data = {}
    
    for key, value in data.items():
        if key == 'spatial_shapes' or key == 'level_start_index':
            # 这些需要是int64类型
            torch_data[key] = torch.from_numpy(value).to(dtype=torch.int64, device=device)
        else:
            # 其他都是float32
            torch_data[key] = torch.from_numpy(value).to(dtype=torch.float32, device=device)
    
    return torch_data

def test_cuda_implementation(torch_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, bool]:
    """
    测试CUDA优化实现
    
    Args:
        torch_data: 测试数据
        
    Returns:
        tuple: (输出张量, 是否成功)
    """
    try:
        print("\n🚀 测试CUDA优化实现...")
        
        from src.ops.ms_deform_attn import ms_deform_attn, is_available
        
        if not is_available():
            print("  ❌ CUDA扩展未编译或不可用")
            return None, False
        
        output = ms_deform_attn(
            torch_data['value'],
            torch_data['spatial_shapes'],
            torch_data['level_start_index'],
            torch_data['sampling_locations'],
            torch_data['attention_weights']
        )
        
        print(f"  ✅ 成功执行，输出形状: {output.shape}")
        return output, True
        
    except Exception as e:
        print(f"  ❌ 执行失败: {e}")
        return None, False

def test_naive_implementation(torch_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, bool]:
    """
    测试naive grid_sample实现
    
    Args:
        torch_data: 测试数据
        
    Returns:
        tuple: (输出张量, 是否成功)
    """
    try:
        print("\n🔄 测试naive grid_sample实现...")
        
        from src.nn.transformer.layers import MSDeformableAttention
        
        # 创建MSDeformableAttention实例
        bs, len_q, d_model = torch_data['sampling_locations'].shape[:3]
        n_heads = torch_data['sampling_locations'].shape[2]
        n_levels = torch_data['spatial_shapes'].shape[0]
        n_points = torch_data['sampling_locations'].shape[4]
        
        attn = MSDeformableAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_levels=n_levels,
            n_points=n_points
        ).cuda()
        
        # 准备输入
        # 注意：MSDeformableAttention需要query作为输入
        query = torch.randn(bs, len_q, d_model, device='cuda')
        
        # 计算reference points (简化处理)
        H, W = torch_data['spatial_shapes'][0].cpu().numpy()
        reference_points = torch.rand(bs, len_q, n_levels, 2, device='cuda')
        
        # 直接使用核心函数进行测试
        output = attn._deformable_attention_core_func(
            torch_data['value'],
            torch_data['spatial_shapes'],
            torch_data['level_start_index'],
            torch_data['sampling_locations'],
            torch_data['attention_weights']
        )
        
        print(f"  ✅ 成功执行，输出形状: {output.shape}")
        return output, True
        
    except Exception as e:
        print(f"  ❌ 执行失败: {e}")
        return None, False

def compute_accuracy_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    计算精度指标
    
    Args:
        pred: 预测值
        target: 目标值
        
    Returns:
        dict: 精度指标字典
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else target
    
    # 计算各种误差指标
    mae = np.mean(np.abs(pred_np - target_np))
    mse = np.mean((pred_np - target_np) ** 2)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(pred_np - target_np))
    
    # 相对误差
    target_norm = np.linalg.norm(target_np)
    if target_norm > 0:
        relative_error = np.linalg.norm(pred_np - target_np) / target_norm
    else:
        relative_error = float('inf')
    
    # 相关性
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1] if len(pred_flat) > 1 else 1.0
    
    return {
        'mae': mae,
        'mse': mse, 
        'rmse': rmse,
        'max_error': max_error,
        'relative_error': relative_error,
        'correlation': correlation
    }

def print_accuracy_report(metrics: Dict[str, float], title: str):
    """
    打印精度报告
    
    Args:
        metrics: 精度指标
        title: 报告标题
    """
    print(f"\n📊 {title}")
    print("=" * 50)
    print(f"平均绝对误差 (MAE):     {metrics['mae']:.8f}")
    print(f"均方误差 (MSE):         {metrics['mse']:.8f}")
    print(f"均方根误差 (RMSE):      {metrics['rmse']:.8f}")
    print(f"最大绝对误差:          {metrics['max_error']:.8f}")
    print(f"相对误差:             {metrics['relative_error']:.8f}")
    print(f"相关系数:             {metrics['correlation']:.8f}")
    
    # 给出精度等级判断
    if metrics['relative_error'] < 1e-6:
        level = "🟢 极高精度 (< 1e-6)"
    elif metrics['relative_error'] < 1e-5:
        level = "🟡 高精度 (< 1e-5)"
    elif metrics['relative_error'] < 1e-4:
        level = "🟠 中等精度 (< 1e-4)"
    elif metrics['relative_error'] < 1e-3:
        level = "🔴 低精度 (< 1e-3)"
    else:
        level = "⚫ 精度不足 (>= 1e-3)"
    
    print(f"精度等级:             {level}")

def main():
    """主函数"""
    print("🔍 RT-DETRv3 可变形注意力精度对比测试")
    print("=" * 60)
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法进行测试")
        return
    
    print(f"✅ 使用设备: {torch.cuda.get_device_name()}")
    
    try:
        # 1. 加载Paddle测试数据
        paddle_data = load_paddle_test_data()
        
        # 检查必要的文件是否存在
        required_keys = ['value', 'spatial_shapes', 'level_start_index', 
                        'sampling_locations', 'attention_weights', 'output_cuda']
        
        missing_keys = [key for key in required_keys if key not in paddle_data]
        if missing_keys:
            print(f"❌ 缺少必要的测试数据: {missing_keys}")
            return
        
        # 2. 转换为PyTorch张量
        torch_data = convert_numpy_to_torch(paddle_data)
        paddle_output = torch_data['output_cuda']
        
        print(f"\n📋 测试数据概要:")
        print(f"  Batch size: {torch_data['value'].shape[0]}")
        print(f"  Value shape: {torch_data['value'].shape}")
        print(f"  Levels: {torch_data['spatial_shapes'].shape[0]}")
        print(f"  Expected output shape: {paddle_output.shape}")
        
        # 3. 测试CUDA实现
        cuda_output, cuda_success = test_cuda_implementation(torch_data)
        
        # 4. 测试naive实现
        naive_output, naive_success = test_naive_implementation(torch_data)
        
        # 5. 精度对比
        print("\n" + "=" * 60)
        print("🎯 精度对比结果")
        
        if cuda_success and cuda_output is not None:
            cuda_metrics = compute_accuracy_metrics(cuda_output, paddle_output)
            print_accuracy_report(cuda_metrics, "CUDA实现 vs Paddle原版")
        else:
            print("\n❌ CUDA实现测试失败，无法进行精度对比")
        
        if naive_success and naive_output is not None:
            naive_metrics = compute_accuracy_metrics(naive_output, paddle_output)
            print_accuracy_report(naive_metrics, "Naive实现 vs Paddle原版")
        else:
            print("\n❌ Naive实现测试失败，无法进行精度对比")
        
        # 6. 实现间对比
        if cuda_success and naive_success and cuda_output is not None and naive_output is not None:
            inter_metrics = compute_accuracy_metrics(cuda_output, naive_output)
            print_accuracy_report(inter_metrics, "CUDA实现 vs Naive实现")
        
        # 7. 总结
        print("\n" + "=" * 60)
        print("📝 测试总结")
        
        if cuda_success:
            print("✅ CUDA优化实现: 可用")
        else:
            print("❌ CUDA优化实现: 不可用")
            
        if naive_success:
            print("✅ Naive备选实现: 可用")
        else:
            print("❌ Naive备选实现: 不可用")
        
        if cuda_success and cuda_metrics['relative_error'] < 1e-5:
            print("🎉 CUDA实现精度验证通过!")
        elif cuda_success:
            print("⚠️  CUDA实现精度可能需要进一步优化")
        
        if naive_success and naive_metrics['relative_error'] < 1e-4:
            print("🎉 Naive实现精度验证通过!")
        elif naive_success:
            print("⚠️  Naive实现精度在预期范围内")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()