/*
 * Copyright (c) 2024 RT-DETRv3 PyTorch Implementation
 * Adapted from PaddlePaddle implementation
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

#include "ms_deform_attn.h"
#include <vector>

// 辅助函数实现
namespace ms_deform_attn_utils {

void validate_forward_inputs(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights) {
    
    // 基本输入验证
    CHECK_INPUT(value);
    CHECK_INPUT(spatial_shapes);
    CHECK_INPUT(level_start_index);
    CHECK_INPUT(sampling_locations);
    CHECK_INPUT(attention_weights);
    
    // 张量维度验证
    CHECK_SHAPE(value, 4);  // [B, L, H, C]
    CHECK_SHAPE(spatial_shapes, 2);  // [Levels, 2]
    CHECK_SHAPE(level_start_index, 1);  // [Levels]
    CHECK_SHAPE(sampling_locations, 6);  // [B, Q, H, Levels, Points, 2]
    CHECK_SHAPE(attention_weights, 5);  // [B, Q, H, Levels, Points]
    
    // 设备一致性检查
    CHECK_SAME_DEVICE(value, spatial_shapes);
    CHECK_SAME_DEVICE(value, level_start_index);
    CHECK_SAME_DEVICE(value, sampling_locations);
    CHECK_SAME_DEVICE(value, attention_weights);
    
    // 数据类型检查
    TORCH_CHECK(spatial_shapes.dtype() == at::kLong, 
                "spatial_shapes must be int64 tensor");
    TORCH_CHECK(level_start_index.dtype() == at::kLong, 
                "level_start_index must be int64 tensor");
    CHECK_SAME_TYPE(value, sampling_locations);
    CHECK_SAME_TYPE(value, attention_weights);
    
    // 形状一致性检查
    const int batch_size = value.size(0);
    const int num_heads = value.size(2);
    const int num_levels = spatial_shapes.size(0);
    const int query_length = sampling_locations.size(1);
    const int num_points = sampling_locations.size(4);
    
    TORCH_CHECK(sampling_locations.size(0) == batch_size, 
                "sampling_locations batch dimension mismatch");
    TORCH_CHECK(sampling_locations.size(2) == num_heads,
                "sampling_locations heads dimension mismatch");
    TORCH_CHECK(sampling_locations.size(3) == num_levels,
                "sampling_locations levels dimension mismatch");
    TORCH_CHECK(sampling_locations.size(5) == 2,
                "sampling_locations last dimension must be 2");
    
    TORCH_CHECK(attention_weights.size(0) == batch_size,
                "attention_weights batch dimension mismatch");
    TORCH_CHECK(attention_weights.size(1) == query_length,
                "attention_weights query dimension mismatch");
    TORCH_CHECK(attention_weights.size(2) == num_heads,
                "attention_weights heads dimension mismatch");
    TORCH_CHECK(attention_weights.size(3) == num_levels,
                "attention_weights levels dimension mismatch");
    TORCH_CHECK(attention_weights.size(4) == num_points,
                "attention_weights points dimension mismatch");
    
    TORCH_CHECK(level_start_index.size(0) == num_levels,
                "level_start_index size mismatch");
}

void validate_backward_inputs(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    const at::Tensor& grad_output) {
    
    // 首先验证前向传播输入
    validate_forward_inputs(value, spatial_shapes, level_start_index,
                          sampling_locations, attention_weights);
    
    // 验证梯度输出
    CHECK_INPUT(grad_output);
    CHECK_SHAPE(grad_output, 3);  // [B, Q, H*C]
    CHECK_SAME_DEVICE(value, grad_output);
    CHECK_SAME_TYPE(value, grad_output);
    
    const int batch_size = value.size(0);
    const int num_heads = value.size(2);
    const int channels = value.size(3);
    const int query_length = sampling_locations.size(1);
    
    TORCH_CHECK(grad_output.size(0) == batch_size,
                "grad_output batch dimension mismatch");
    TORCH_CHECK(grad_output.size(1) == query_length,
                "grad_output query dimension mismatch");
    TORCH_CHECK(grad_output.size(2) == num_heads * channels,
                "grad_output channel dimension mismatch");
}

std::vector<int64_t> infer_output_shape(
    const at::Tensor& value,
    const at::Tensor& sampling_locations) {
    const int batch_size = value.size(0);
    const int num_heads = value.size(2);
    const int channels = value.size(3);
    const int query_length = sampling_locations.size(1);
    
    return {batch_size, query_length, num_heads * channels};
}

} // namespace ms_deform_attn_utils

// 前向传播CPU版本 (如果需要的话)
at::Tensor ms_deform_attn_forward_cpu(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights) {
    
    AT_ERROR("MS Deformable Attention forward is not implemented on CPU");
}

// 反向传播CPU版本 (如果需要的话)
std::vector<at::Tensor> ms_deform_attn_backward_cpu(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    const at::Tensor& grad_output) {
    
    AT_ERROR("MS Deformable Attention backward is not implemented on CPU");
}

// 前向传播主接口 - PyTorch dispatch wrapper
at::Tensor ms_deform_attn_forward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights) {
    
    // 输入验证
    ms_deform_attn_utils::validate_forward_inputs(
        value, spatial_shapes, level_start_index, 
        sampling_locations, attention_weights);
    
    // 根据张量设备类型分发到不同实现
    if (value.device().is_cuda()) {
        return ms_deform_attn_forward_cuda(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights);
    } else {
        return ms_deform_attn_forward_cpu(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights);
    }
}

// 反向传播主接口 - PyTorch dispatch wrapper
std::vector<at::Tensor> ms_deform_attn_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    const at::Tensor& grad_output) {
    
    // 输入验证
    ms_deform_attn_utils::validate_backward_inputs(
        value, spatial_shapes, level_start_index,
        sampling_locations, attention_weights, grad_output);
    
    // 根据张量设备类型分发到不同实现
    if (value.device().is_cuda()) {
        return ms_deform_attn_backward_cuda(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, grad_output);
    } else {
        return ms_deform_attn_backward_cpu(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, grad_output);
    }
}

// Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MS Deformable Attention PyTorch Extension";
    
    m.def("forward", &ms_deform_attn_forward, 
          "MS Deformable Attention forward pass",
          pybind11::arg("value"),
          pybind11::arg("spatial_shapes"), 
          pybind11::arg("level_start_index"),
          pybind11::arg("sampling_locations"),
          pybind11::arg("attention_weights"));
    
    m.def("backward", &ms_deform_attn_backward,
          "MS Deformable Attention backward pass", 
          pybind11::arg("value"),
          pybind11::arg("spatial_shapes"),
          pybind11::arg("level_start_index"), 
          pybind11::arg("sampling_locations"),
          pybind11::arg("attention_weights"),
          pybind11::arg("grad_output"));
    
    // 版本信息
    m.attr("__version__") = "1.0.0";
    
    // 实用函数导出
    m.def("validate_forward_inputs", &ms_deform_attn_utils::validate_forward_inputs,
          "Validate forward pass input tensors");
    m.def("validate_backward_inputs", &ms_deform_attn_utils::validate_backward_inputs,
          "Validate backward pass input tensors");
    m.def("infer_output_shape", &ms_deform_attn_utils::infer_output_shape,
          "Infer output tensor shape");
}