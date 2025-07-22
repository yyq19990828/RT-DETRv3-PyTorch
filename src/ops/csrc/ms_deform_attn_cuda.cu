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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ static inline double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ >= 700
__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val)
{
    atomicAdd(reinterpret_cast<__half*>(address), static_cast<__half>(val));
}
#else
__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val)
{
    unsigned int* address_as_uint = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_uint;
    unsigned int assumed;

    do {
        assumed = old;
        __half2 old_h2 = *(__half2*)&assumed;
        if (((size_t)address & 2) == 0) {
            old_h2.x = __hadd(old_h2.x, (__half)val);
        } else {
            old_h2.y = __hadd(old_h2.y, (__half)val);
        }
        old = atomicCAS(address_as_uint, assumed, *(unsigned int*)&old_h2);
    } while (assumed != old);
}
#endif
#endif

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

// 前向传播双线性插值函数
template <typename data_t>
__device__ data_t deformable_attn_bilinear_forward(
    const data_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const data_t &h, const data_t &w,
    const int &m, const int &c) {
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const data_t lh = h - h_low;
  const data_t lw = w - w_low;
  const data_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  data_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  data_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  data_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  data_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const data_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const data_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

// 前向传播CUDA核函数
template <typename data_t>
__global__ void deformable_attn_cuda_kernel_forward(
    const int n, const data_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const data_t *data_sampling_loc,
    const data_t *data_attn_weight, const int batch_size,
    const int value_length, const int num_heads, const int channels,
    const int num_levels, const int query_length, const int num_points,
    data_t *output_data_ptr) {
  CUDA_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    // const int q_col = _temp % query_length;
    _temp /= query_length;
    const int b_col = _temp;

    data_t *data_ptr = output_data_ptr + index;
    int data_weight_ptr = sampling_index * num_levels * num_points;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * value_length * qid_stride;
    data_t col = 0;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const data_t *data_value_ptr = data_value + (data_value_ptr_init_offset +
                                                   level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_points; ++p_col) {
        const data_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const data_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const data_t weight = data_attn_weight[data_weight_ptr];

        // Convert from [0,1] normalized coordinates to pixel coordinates
        // Following PaddlePaddle's implementation: sampling_grids = 2 * sampling_locations - 1
        // Then convert from [-1,1] grid coordinates to pixel coordinates using PyTorch's align_corners=False formula
        const data_t grid_h = 2.0 * loc_h - 1.0;
        const data_t grid_w = 2.0 * loc_w - 1.0;
        const data_t h_im = (grid_h + 1.0) * static_cast<data_t>(spatial_h) * 0.5 - 0.5;
        const data_t w_im = (grid_w + 1.0) * static_cast<data_t>(spatial_w) * 0.5 - 0.5;

        // Always call bilinear interpolation - it handles boundaries internally
        col += deformable_attn_bilinear_forward(
                   data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                   h_im, w_im, m_col, c_col) *
               weight;

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_ptr = col;
  }
}

// 反向传播双线性插值函数
template <typename data_t>
__device__ void deformable_attn_bilinear_backward(
    const data_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const data_t &h, const data_t &w,
    const int &m, const int &c, const data_t &top_grad,
    const data_t &attn_weight, data_t *&grad_value, data_t *grad_sampling_loc,
    data_t *grad_attn_weight) {
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const data_t lh = h - h_low;
  const data_t lw = w - w_low;
  const data_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const data_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const data_t top_grad_value = top_grad * attn_weight;
  data_t grad_h_weight = 0, grad_w_weight = 0;

  data_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  data_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  data_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  data_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const data_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_attn_weight = top_grad * val;
  *grad_sampling_loc = width * grad_w_weight * top_grad_value;
  *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
}

// 全局内存版本的反向传播双线性插值函数
template <typename data_t>
__device__ void deformable_attn_bilinear_backward_gm(
    const data_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const data_t &h, const data_t &w,
    const int &m, const int &c, const data_t &top_grad,
    const data_t &attn_weight, data_t *&grad_value, data_t *grad_sampling_loc,
    data_t *grad_attn_weight) {
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const data_t lh = h - h_low;
  const data_t lw = w - w_low;
  const data_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const data_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const data_t top_grad_value = top_grad * attn_weight;
  data_t grad_h_weight = 0, grad_w_weight = 0;

  data_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  data_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  data_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  data_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const data_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_attn_weight, top_grad * val);
  atomicAdd(grad_sampling_loc, width * grad_w_weight * top_grad_value);
  atomicAdd(grad_sampling_loc + 1, height * grad_h_weight * top_grad_value);
}

// 反向传播CUDA核函数 (简化版本，适用于大多数情况)
template <typename data_t>
__global__ void deformable_attn_cuda_kernel_backward_gm(
    const int n, const data_t *grad_col, const data_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const data_t *data_sampling_loc, const data_t *data_attn_weight,
    const int batch_size, const int value_length, const int num_heads,
    const int channels, const int num_levels, const int query_length,
    const int num_points, data_t *grad_value, data_t *grad_sampling_loc,
    data_t *grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    // const int q_col = _temp % query_length;
    _temp /= query_length;
    const int b_col = _temp;

    const data_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_points;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * value_length * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const data_t *data_value_ptr = data_value + value_ptr_offset;
      data_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_points; ++p_col) {
        const data_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const data_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const data_t weight = data_attn_weight[data_weight_ptr];

        // Convert from [0,1] normalized coordinates to pixel coordinates
        // Following PaddlePaddle's implementation: sampling_grids = 2 * sampling_locations - 1
        // Then convert from [-1,1] grid coordinates to pixel coordinates using PyTorch's align_corners=False formula
        const data_t grid_h = 2.0 * loc_h - 1.0;
        const data_t grid_w = 2.0 * loc_w - 1.0;
        const data_t h_im = (grid_h + 1.0) * static_cast<data_t>(spatial_h) * 0.5 - 0.5;
        const data_t w_im = (grid_w + 1.0) * static_cast<data_t>(spatial_w) * 0.5 - 0.5;
        // Always call bilinear backward - it handles boundaries internally
        deformable_attn_bilinear_backward_gm(
            data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
            w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
            grad_sampling_loc, grad_attn_weight);
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

// PyTorch前向传播接口
at::Tensor ms_deform_attn_forward_cuda(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights) {

  // 获取张量维度信息
  const int batch_size = value.size(0);
  const int value_length = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);

  const int num_levels = spatial_shapes.size(0);
  const int query_length = sampling_locations.size(1);
  const int num_points = sampling_locations.size(4);

  // 创建输出张量
  auto output = at::zeros({batch_size, query_length, num_heads * channels},
                          value.options());

  const int num_kernels = batch_size * query_length * num_heads * channels;

  // 获取CUDA设备和流
  const auto& device = value.device();
  c10::cuda::CUDAGuard device_guard(device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  // 启动CUDA核函数
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.scalar_type(), "ms_deform_attn_forward_cuda", [&] {
    deformable_attn_cuda_kernel_forward<scalar_t>
        <<<GET_BLOCKS(num_kernels, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels, 
            value.data_ptr<scalar_t>(),
            spatial_shapes.data_ptr<int64_t>(),
            level_start_index.data_ptr<int64_t>(),
            sampling_locations.data_ptr<scalar_t>(),
            attention_weights.data_ptr<scalar_t>(), 
            batch_size,
            value_length, 
            num_heads, 
            channels, 
            num_levels,
            query_length, 
            num_points, 
            output.data_ptr<scalar_t>());
  });

  // 检查CUDA错误
  AT_CUDA_CHECK(cudaGetLastError());

  return output;
}

// PyTorch反向传播接口
std::vector<at::Tensor> ms_deform_attn_backward_cuda(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    const at::Tensor& grad_output) {

  // 获取张量维度信息
  const int batch_size = value.size(0);
  const int value_length = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);

  const int num_levels = spatial_shapes.size(0);
  const int query_length = sampling_locations.size(1);
  const int num_points = sampling_locations.size(4);

  // 创建梯度张量
  auto grad_value = at::zeros_like(value);
  auto grad_spatial_shapes = at::zeros_like(spatial_shapes);  // 通常为零
  auto grad_level_start_index = at::zeros_like(level_start_index);  // 通常为零
  auto grad_sampling_locations = at::zeros_like(sampling_locations);
  auto grad_attention_weights = at::zeros_like(attention_weights);

  const int num_kernels = batch_size * query_length * num_heads * channels;

  // 获取CUDA设备和流
  const auto& device = value.device();
  c10::cuda::CUDAGuard device_guard(device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  // 启动CUDA核函数
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.scalar_type(), "ms_deform_attn_backward_cuda", [&] {
    deformable_attn_cuda_kernel_backward_gm<scalar_t>
        <<<GET_BLOCKS(num_kernels, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels, 
            grad_output.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            spatial_shapes.data_ptr<int64_t>(),
            level_start_index.data_ptr<int64_t>(),
            sampling_locations.data_ptr<scalar_t>(),
            attention_weights.data_ptr<scalar_t>(), 
            batch_size, 
            value_length,
            num_heads, 
            channels, 
            num_levels, 
            query_length, 
            num_points,
            grad_value.data_ptr<scalar_t>(),
            grad_sampling_locations.data_ptr<scalar_t>(),
            grad_attention_weights.data_ptr<scalar_t>());
  });

  // 检查CUDA错误
  AT_CUDA_CHECK(cudaGetLastError());

  return {grad_value, grad_spatial_shapes, grad_level_start_index,
          grad_sampling_locations, grad_attention_weights};
}