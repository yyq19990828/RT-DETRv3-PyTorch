/*
 * Copyright (c) 2024 RT-DETRv3 PyTorch Implementation
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <torch/extension.h>
#include <vector>

// 张量验证宏定义 - 改编自PyTorch C++扩展最佳实践
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 张量形状验证宏
#define CHECK_SHAPE(x, expected_dims) \
    TORCH_CHECK(x.dim() == expected_dims, #x " must have " #expected_dims " dimensions")

#define CHECK_SAME_DEVICE(x, y) \
    TORCH_CHECK(x.device() == y.device(), #x " and " #y " must be on the same device")

#define CHECK_SAME_TYPE(x, y) \
    TORCH_CHECK(x.dtype() == y.dtype(), #x " and " #y " must have the same dtype")

// 前向传播函数声明
at::Tensor ms_deform_attn_forward(
    const at::Tensor& value,                    // [B, L, H, C] 特征值张量
    const at::Tensor& spatial_shapes,           // [Levels, 2] 空间形状
    const at::Tensor& level_start_index,        // [Levels] 层级起始索引
    const at::Tensor& sampling_locations,       // [B, Q, H, Levels, Points, 2] 采样位置
    const at::Tensor& attention_weights         // [B, Q, H, Levels, Points] 注意力权重
);

// 反向传播函数声明
std::vector<at::Tensor> ms_deform_attn_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    const at::Tensor& grad_output              // [B, Q, H*C] 输出梯度
);

// CUDA前向传播实现声明 (在.cu文件中实现)
at::Tensor ms_deform_attn_forward_cuda(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights
);

// CUDA反向传播实现声明 (在.cu文件中实现)
std::vector<at::Tensor> ms_deform_attn_backward_cuda(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    const at::Tensor& grad_output
);

// 辅助函数声明
namespace ms_deform_attn_utils {

// 验证输入张量形状和设备
void validate_forward_inputs(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights
);

void validate_backward_inputs(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    const at::Tensor& grad_output
);

// 推断输出张量形状
std::vector<int64_t> infer_output_shape(
    const at::Tensor& value,
    const at::Tensor& sampling_locations
);

} // namespace ms_deform_attn_utils