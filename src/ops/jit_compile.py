"""
使用JIT编译方式构建MS Deformable Attention算子
这种方式可以避免setuptools的复杂性和模板实例化问题
"""

import os
import torch
from torch.utils.cpp_extension import load

def jit_compile_extension():
    """JIT编译扩展"""
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csrc_dir = os.path.join(current_dir, 'csrc')
    
    # 源文件列表
    sources = [
        os.path.join(csrc_dir, 'ms_deform_attn.cpp'),
        os.path.join(csrc_dir, 'ms_deform_attn_cuda.cu'),
    ]
    
    # 检查CUDA可用性
    is_cuda_available = torch.cuda.is_available()
    
    # 编译参数 - 避免包含导致模板错误的头文件
    extra_cflags = ['-O3', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=1']
    extra_cuda_cflags = ['-O3', '-std=c++17', '--expt-relaxed-constexpr']
    
    if is_cuda_available:
        # CUDA版本
        extra_ldflags = []
        define_macros = [('WITH_CUDA', None)]
        
        print(f"JIT编译CUDA版本算子...")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA版本: {torch.version.cuda}")
        
        extension = load(
            name='ms_deform_attn_ext',
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            verbose=True,
            with_cuda=True
        )
    else:
        # CPU版本
        cpu_sources = [s for s in sources if not s.endswith('.cu')]
        
        print(f"JIT编译CPU版本算子...")
        print(f"PyTorch版本: {torch.__version__}")
        
        extension = load(
            name='ms_deform_attn_ext',
            sources=cpu_sources,
            extra_cflags=extra_cflags,
            verbose=True,
            with_cuda=False
        )
    
    print(f"算子编译完成!")
    return extension

if __name__ == '__main__':
    ext = jit_compile_extension()
    print(f"算子可用函数: {dir(ext)}")