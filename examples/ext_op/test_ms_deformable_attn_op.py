# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import random
import numpy as np
import paddle

try:
    gpu_index = int(sys.argv[1])
except:
    gpu_index = 0
print(f'Use gpu {gpu_index} to test...')
paddle.set_device(f'gpu:{gpu_index}')

try:
    from deformable_detr_ops import ms_deformable_attn
except Exception as e:
    print('import deformable_detr_ops error', e)
    sys.exit(-1)

paddle.seed(1)
random.seed(1)
np.random.seed(1)

bs, n_heads, c = 2, 8, 8
query_length, n_levels, n_points = 2, 2, 2
spatial_shapes = paddle.to_tensor([(6, 4), (3, 2)], dtype=paddle.int64)
level_start_index = paddle.concat((paddle.to_tensor(
    [0], dtype=paddle.int64), spatial_shapes.prod(1).cumsum(0)[:-1]))
value_length = sum([(H * W).item() for H, W in spatial_shapes])


def get_test_tensors(channels):
    value = paddle.rand(
        [bs, value_length, n_heads, channels], dtype=paddle.float32) * 0.01
    sampling_locations = paddle.rand(
        [bs, query_length, n_heads, n_levels, n_points, 2],
        dtype=paddle.float32)
    attention_weights = paddle.rand(
        [bs, query_length, n_heads, n_levels, n_points],
        dtype=paddle.float32) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
        -2, keepdim=True)

    return [value, sampling_locations, attention_weights]


def save_tensors_for_pytorch_test(channels=32):
    """
    生成并保存用于PyTorch对比测试的张量。
    这会保存输入和Paddle CUDA OP的输出。
    """
    print("\nGenerating and saving tensors for PyTorch comparison...")

    # 创建保存目录
    save_dir = os.path.join(os.path.dirname(__file__), 'paddle_test_data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    value, sampling_locations, attention_weights = get_test_tensors(channels)

    # 运行Paddle CUDA OP
    output_cuda = ms_deformable_attn(value, spatial_shapes, level_start_index,
                                     sampling_locations,
                                     attention_weights)

    # 保存张量
    tensors_to_save = {
        'value': value,
        'spatial_shapes': spatial_shapes,
        'level_start_index': level_start_index,
        'sampling_locations': sampling_locations,
        'attention_weights': attention_weights,
        'output_cuda': output_cuda
    }

    for name, tensor in tensors_to_save.items():
        filepath = os.path.join(save_dir, f'{name}.npy')
        np.save(filepath, tensor.numpy())
        print(f"Saved {name} to {filepath}")


if __name__ == '__main__':
    # The original tests are removed due to ppdet dependency.
    # This script is now used to generate test data for cross-framework comparison.
    print("Generating tensors for PyTorch comparison test...")
    save_tensors_for_pytorch_test(channels=32)
