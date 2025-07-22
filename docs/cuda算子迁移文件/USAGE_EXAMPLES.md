# MS Deformable Attention 算子使用示例

本文档提供了MultiScale Deformable Attention算子在PyTorch中的详细使用示例。

## 快速开始

### 基本导入
```python
import torch
import torch.nn as nn
from src.ops import ms_deform_attn
```

### 简单使用示例
```python
# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 基本参数
batch_size = 2
num_queries = 100  # 查询点数量
num_heads = 8      # 注意力头数
embed_dims = 256   # 嵌入维度
num_levels = 4     # 特征层级数  
num_points = 4     # 每个查询的采样点数

# 创建输入数据
value = torch.randn(batch_size, num_queries, num_heads, embed_dims // num_heads).to(device)
spatial_shapes = torch.tensor([[100, 100], [50, 50], [25, 25], [13, 13]], dtype=torch.long).to(device)
level_start_index = torch.tensor([0, 10000, 12500, 13125], dtype=torch.long).to(device)
sampling_locations = torch.randn(batch_size, num_queries, num_heads, num_levels, num_points, 2).to(device)
attention_weights = torch.softmax(torch.randn(batch_size, num_queries, num_heads, num_levels, num_points), dim=-1).to(device)

# 前向传播
output = ms_deform_attn(
    value=value,
    spatial_shapes=spatial_shapes,
    level_start_index=level_start_index,
    sampling_locations=sampling_locations,
    attention_weights=attention_weights
)

print(f"输入形状: {value.shape}")
print(f"输出形状: {output.shape}")
```

## 详细使用指南

### 1. 输入参数说明

#### value (特征值张量)
- **形状**: `(batch_size, num_queries, num_heads, head_dim)`
- **数据类型**: `torch.float32`
- **描述**: 多尺度特征图的值表示

```python
# 示例: 从多尺度特征图生成value
feature_maps = [feat1, feat2, feat3, feat4]  # 不同尺度的特征图
value = torch.cat([f.flatten(2).transpose(1, 2) for f in feature_maps], dim=1)
value = value.view(batch_size, -1, num_heads, head_dim)
```

#### spatial_shapes (空间形状)
- **形状**: `(num_levels, 2)`
- **数据类型**: `torch.long`
- **描述**: 每个特征层级的空间尺寸 [高度, 宽度]

```python
# 示例: 从特征图推断spatial_shapes
spatial_shapes = torch.tensor([
    [feat.shape[-2], feat.shape[-1]] for feat in feature_maps
], dtype=torch.long)
```

#### level_start_index (层级起始索引)
- **形状**: `(num_levels,)`
- **数据类型**: `torch.long`  
- **描述**: 每个特征层级在flattened特征中的起始位置

```python
# 示例: 计算level_start_index
level_start_index = []
start_idx = 0
for h, w in spatial_shapes:
    level_start_index.append(start_idx)
    start_idx += h * w
level_start_index = torch.tensor(level_start_index, dtype=torch.long)
```

#### sampling_locations (采样位置)
- **形状**: `(batch_size, num_queries, num_heads, num_levels, num_points, 2)`
- **数据类型**: `torch.float32`
- **值范围**: [0, 1] 归一化坐标
- **描述**: 每个查询在每个特征层级的采样点坐标

```python
# 示例: 生成采样位置 
# 通常由网络学习生成，这里展示格式
sampling_locations = torch.rand(
    batch_size, num_queries, num_heads, num_levels, num_points, 2
)
# 确保坐标在[0,1]范围内
sampling_locations = torch.clamp(sampling_locations, 0, 1)
```

#### attention_weights (注意力权重)
- **形状**: `(batch_size, num_queries, num_heads, num_levels, num_points)`
- **数据类型**: `torch.float32`
- **约束**: 在最后一个维度上求和为1 (softmax归一化)
- **描述**: 每个采样点的注意力权重

```python
# 示例: 生成注意力权重
raw_weights = torch.randn(batch_size, num_queries, num_heads, num_levels, num_points)
attention_weights = torch.softmax(raw_weights, dim=-1)
```

### 2. 完整的神经网络模块示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.ops import ms_deform_attn

class MultiScaleDeformableAttention(nn.Module):
    """
    MultiScale Deformable Attention模块的完整实现
    """
    
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels  
        self.num_points = num_points
        self.head_dim = embed_dims // num_heads
        
        # 确保embed_dims能被num_heads整除
        assert embed_dims % num_heads == 0
        
        # 采样位置偏移预测网络
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        
        # 注意力权重预测网络  
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        
        # 值投影
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        
        # 输出投影
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        # 采样偏移的初始化很重要，通常初始化为很小的值
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        
        # 注意力权重初始化
        nn.init.constant_(self.attention_weights.weight, 0.0) 
        nn.init.constant_(self.attention_weights.bias, 0.0)
        
        # 值投影和输出投影正常初始化
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
    
    def forward(self, query, reference_points, value, spatial_shapes, level_start_index):
        """
        前向传播
        
        Args:
            query: 查询特征 (batch_size, num_queries, embed_dims)
            reference_points: 参考点坐标 (batch_size, num_queries, num_levels, 2)
            value: 多尺度特征值 (batch_size, num_value, embed_dims)  
            spatial_shapes: 空间形状 (num_levels, 2)
            level_start_index: 层级起始索引 (num_levels,)
        
        Returns:
            output: 注意力输出 (batch_size, num_queries, embed_dims)
        """
        
        batch_size, num_queries, _ = query.shape
        num_value = value.shape[1]
        
        # 值投影并重塑为多头格式
        value = self.value_proj(value)
        value = value.view(batch_size, num_value, self.num_heads, self.head_dim)
        
        # 预测采样位置偏移
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, num_queries, self.num_heads, 
            self.num_levels, self.num_points, 2
        )
        
        # 预测注意力权重并归一化
        attention_weights = self.attention_weights(query).view(
            batch_size, num_queries, self.num_heads, 
            self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            batch_size, num_queries, self.num_heads,
            self.num_levels, self.num_points
        )
        
        # 计算最终采样位置 = 参考点 + 偏移
        # reference_points: (batch_size, num_queries, num_levels, 2)
        # 扩展维度以匹配采样偏移
        reference_points = reference_points.unsqueeze(2).unsqueeze(4)  # 添加head和point维度
        sampling_locations = reference_points + sampling_offsets
        
        # 确保采样位置在有效范围内 [0, 1]
        sampling_locations = torch.clamp(sampling_locations, 0, 1)
        
        # 调用自定义CUDA算子
        output = ms_deform_attn(
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights
        )
        
        # 输出投影
        output = self.output_proj(output)
        
        return output

# 使用示例
if __name__ == "__main__":
    # 创建模块
    msda = MultiScaleDeformableAttention(
        embed_dims=256, 
        num_heads=8, 
        num_levels=4, 
        num_points=4
    ).cuda()
    
    # 准备输入数据
    batch_size = 2
    num_queries = 100
    
    # 查询特征 (通常来自decoder)
    query = torch.randn(batch_size, num_queries, 256).cuda()
    
    # 参考点 (通常来自位置编码或学习的位置)
    reference_points = torch.rand(batch_size, num_queries, 4, 2).cuda()
    
    # 多尺度特征值 (来自encoder的不同尺度特征)
    num_value = 13125  # 100*100 + 50*50 + 25*25 + 13*13
    value = torch.randn(batch_size, num_value, 256).cuda()
    
    # 空间形状和起始索引
    spatial_shapes = torch.tensor([[100, 100], [50, 50], [25, 25], [13, 13]], 
                                 dtype=torch.long).cuda()
    level_start_index = torch.tensor([0, 10000, 12500, 13125], dtype=torch.long).cuda()
    
    # 前向传播
    output = msda(query, reference_points, value, spatial_shapes, level_start_index)
    
    print(f"查询形状: {query.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数数量: {sum(p.numel() for p in msda.parameters())}")
```

### 3. 在Transformer中的使用

```python
class DeformableTransformerDecoderLayer(nn.Module):
    """
    带有Deformable Attention的Transformer Decoder层
    """
    
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        
        # Self-attention (通常使用标准多头注意力)
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        
        # Cross-attention (使用Deformable Attention)  
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dims, num_heads, num_levels, num_points
        )
        
        # Feed-forward网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dims * 4, embed_dims),
            nn.Dropout(0.1)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims) 
        self.norm3 = nn.LayerNorm(embed_dims)
    
    def forward(self, query, reference_points, value, spatial_shapes, level_start_index):
        # Self-attention
        query2, _ = self.self_attn(query, query, query)
        query = query + query2
        query = self.norm1(query)
        
        # Cross-attention with deformable attention
        query2 = self.cross_attn(query, reference_points, value, spatial_shapes, level_start_index)
        query = query + query2
        query = self.norm2(query)
        
        # Feed-forward
        query2 = self.ffn(query)
        query = query + query2
        query = self.norm3(query)
        
        return query
```

### 4. 性能优化技巧

#### 内存优化
```python
# 使用torch.cuda.amp混合精度训练
with torch.cuda.amp.autocast():
    output = ms_deform_attn(value, spatial_shapes, level_start_index, 
                           sampling_locations, attention_weights)
```

#### 批处理优化
```python
# 确保输入张量是连续的以获得最佳性能
value = value.contiguous()
sampling_locations = sampling_locations.contiguous()
attention_weights = attention_weights.contiguous()

output = ms_deform_attn(value, spatial_shapes, level_start_index,
                       sampling_locations, attention_weights)
```

### 5. 梯度检查和调试

```python
def test_gradient():
    """测试算子的梯度计算是否正确"""
    
    # 创建需要梯度的输入
    value = torch.randn(2, 100, 8, 32, requires_grad=True).cuda()
    sampling_locations = torch.rand(2, 100, 8, 4, 4, 2, requires_grad=True).cuda()
    attention_weights = torch.softmax(
        torch.randn(2, 100, 8, 4, 4, requires_grad=True), dim=-1
    ).cuda()
    
    # 不需要梯度的输入
    spatial_shapes = torch.tensor([[100, 100], [50, 50], [25, 25], [13, 13]], 
                                 dtype=torch.long).cuda()
    level_start_index = torch.tensor([0, 10000, 12500, 13125], dtype=torch.long).cuda()
    
    # 前向传播
    output = ms_deform_attn(value, spatial_shapes, level_start_index,
                           sampling_locations, attention_weights)
    
    # 计算损失并反向传播
    loss = output.sum()
    loss.backward()
    
    # 检查梯度
    print("value梯度形状:", value.grad.shape if value.grad is not None else "None")
    print("采样位置梯度形状:", sampling_locations.grad.shape if sampling_locations.grad is not None else "None")
    print("注意力权重梯度形状:", attention_weights.grad.shape if attention_weights.grad is not None else "None")
    
    # 数值梯度检查 (计算量大，仅用于调试)
    if False:  # 设置为True来启用数值梯度检查
        torch.autograd.gradcheck(
            lambda v, sl, aw: ms_deform_attn(v, spatial_shapes, level_start_index, sl, aw),
            (value.double(), sampling_locations.double(), attention_weights.double()),
            eps=1e-6, atol=1e-4
        )
        print("梯度检查通过!")

# 运行梯度测试
test_gradient()
```

### 6. 常见错误和解决方案

#### 错误1: 张量形状不匹配
```python
# 错误示例
value = torch.randn(2, 100, 256)  # 缺少num_heads维度

# 正确做法
value = torch.randn(2, 100, 8, 32)  # (batch, num_value, num_heads, head_dim)
```

#### 错误2: 数据类型不正确  
```python
# 错误示例
spatial_shapes = torch.tensor([[100, 100]], dtype=torch.float32)

# 正确做法
spatial_shapes = torch.tensor([[100, 100]], dtype=torch.long)
```

#### 错误3: 设备不一致
```python
# 确保所有张量在同一设备上
device = torch.device('cuda')
value = value.to(device)
spatial_shapes = spatial_shapes.to(device)
level_start_index = level_start_index.to(device)
sampling_locations = sampling_locations.to(device)
attention_weights = attention_weights.to(device)
```

这份文档提供了从基本使用到高级应用的完整指南，帮助用户正确高效地使用MS Deformable Attention算子。