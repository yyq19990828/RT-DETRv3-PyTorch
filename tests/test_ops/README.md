# MS Deformable Attention 测试文档

本目录包含了MS Deformable Attention算子的完整测试套件。

## 测试文件

- `test_ms_deform_attn.py` - 主要测试文件，包含所有测试用例

## 测试覆盖范围

### 1. 基础功能测试 (`TestMSDeformAttnBasic`)
- 前向传播基本功能
- 不同通道大小适配性
- 不同批次大小适配性
- nn.Module接口测试

### 2. 正确性验证 (`TestMSDeformAttnCorrectness`)
- CUDA实现 vs 原生PyTorch实现对比
- 数值精度验证
- 误差分析

### 3. MSDeformableAttention模块测试 (`TestMSDeformableAttentionModule`)
- 完整模块功能测试
- 不同参考点格式支持 (2D/4D)
- 值掩码功能
- 模块梯度计算

### 4. 梯度测试 (`TestMSDeformAttnGradients`)
- 梯度存在性验证
- 梯度数值稳定性
- 数值梯度检查（可选慢速测试）

### 5. 边界条件测试 (`TestMSDeformAttnEdgeCases`)
- 输入验证
- 设备匹配检查
- 数据类型处理

### 6. 性能测试 (`TestMSDeformAttnPerformance`)
- 大规模数据性能
- CUDA vs 原生实现性能对比
- 加速比分析

## 运行测试

### 基本测试
```bash
# 运行标准测试套件
python3 tests/test_ops/test_ms_deform_attn.py

# 使用pytest
pytest tests/test_ops/test_ms_deform_attn.py -v
```

### 包含性能测试
```bash
# 运行完整测试（包括性能测试）
python3 tests/test_ops/test_ms_deform_attn.py --performance

# 或使用短参数
python3 tests/test_ops/test_ms_deform_attn.py -p
```

### 特定测试用例
```bash
# 运行特定测试函数
python -m pytest tests/test_ops/test_ms_deform_attn.py::TestMSDeformAttnBasic::test_forward_basic -v

# 运行特定测试类
python -m pytest tests/test_ops/test_ms_deform_attn.py::TestMSDeformAttnCorrectness -v
```

## 测试结果解读

### 正确性测试
- `max_abs_err`: 最大绝对误差，应该 < 1e-5
- `max_rel_err`: 最大相对误差，应该 < 1e-3

### 性能测试
- `CUDA实现平均时间`: CUDA扩展的执行时间
- `原生PyTorch实现平均时间`: 纯PyTorch实现的执行时间
- `加速比`: CUDA实现相对于原生实现的性能提升倍数

## 环境要求

- CUDA支持的GPU
- 已编译的MS Deformable Attention CUDA扩展
- PyTorch >= 1.7.0
- Python >= 3.6

## 故障排除

### 测试失败：算子不可用
```
MS Deformable Attention算子不可用，请先编译扩展
```
**解决方案**: 
1. 确保CUDA环境正确配置
2. 编译CUDA扩展：`cd src/ops && python setup.py build_ext --inplace`

### 测试失败：CUDA不可用
```
CUDA不可用，跳过测试
```
**解决方案**: 
1. 检查GPU驱动
2. 确认PyTorch安装了CUDA支持版本

### 正确性测试失败
如果CUDA实现与原生实现结果不一致，可能的原因：
1. CUDA扩展编译问题
2. 输入数据格式不匹配
3. 数值精度设置

## 更新历史

- **2025-07-22**: 根据最新的MSDeformableAttention实现更新测试
  - 修正了`ms_deform_attn_naive`函数，确保与layers.py实现一致
  - 新增MSDeformableAttention完整模块测试
  - 优化了测试组织结构和输出格式
  - 添加了性能基准测试
