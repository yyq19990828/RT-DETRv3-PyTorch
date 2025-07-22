"""
MS Deformable Attention PyTorch扩展构建脚本

使用方法:
    python setup.py build_ext --inplace    # 本地编译
    python setup.py install                # 安装到site-packages
    python setup.py install_local          # 在当前目录本地构建, 不安装到 site-packages
    
环境要求:
    - PyTorch >= 1.8.0
    - CUDA Toolkit (如果启用CUDA)
    - ninja (可选，加速编译)
"""

import os
import sys
import glob
from pathlib import Path
from setuptools import setup, find_packages, Command
from pybind11.setup_helpers import Pybind11Extension, build_ext as _build_ext

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# 版本信息
__version__ = "1.0.0"

# 获取项目根路径
ROOT_DIR = Path(__file__).parent.absolute()
CSRC_DIR = ROOT_DIR / "csrc"

def get_version():
    """获取版本号"""
    return __version__

def get_cuda_version():
    """获取CUDA版本信息"""
    if torch.cuda.is_available():
        return torch.version.cuda
    return None

def check_dependencies():
    """检查依赖项"""
    print(f"PyTorch版本: {torch.__version__}")
    
    cuda_version = get_cuda_version()
    if cuda_version:
        print(f"CUDA版本: {cuda_version}")
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
    else:
        print("CUDA不可用")
    
    # 检查ninja
    try:
        import ninja
        print("Ninja已安装")
    except ImportError:
        print("Ninja未安装，将使用默认构建工具")

def get_compiler_flags():
    """获取编译器标志"""
    cxx_flags = []
    nvcc_flags = []
    
    # C++标准
    cxx_flags.extend(['-std=c++17'])
    
    # 减少模板实例化以避免编译错误
    cxx_flags.extend([
        '-DTORCH_EXTENSION_NAME=ms_deform_attn_ext',
        '-DTORCH_DISABLE_DEPRECATED_WARNINGS',  # 禁用弃用警告
        '-DWITH_PYTHON',                        # 明确启用Python支持
    ])
    
    # 优化标志
    if os.getenv('DEBUG', '0') == '1':
        cxx_flags.extend(['-O0', '-g'])
        nvcc_flags.extend(['-O0', '-g', '-G'])
        print("Debug模式编译")
    else:
        cxx_flags.extend(['-O3'])
        nvcc_flags.extend(['-O3'])
        print("Release模式编译")
    
    # CUDA架构
    # 自动检测可用的GPU架构
    if torch.cuda.is_available():
        # 获取当前设备的计算能力
        capability = torch.cuda.get_device_capability()
        arch = f"{capability[0]}.{capability[1]}"
        nvcc_flags.extend([f'-gencode=arch=compute_{capability[0]}{capability[1]},code=sm_{capability[0]}{capability[1]}'])
        print(f"为GPU架构 {arch} 编译")
        
        # 添加常见架构以获得更好的兼容性
        common_archs = [
            ('compute_60', 'sm_60'),  # Pascal
            ('compute_70', 'sm_70'),  # Volta  
            ('compute_75', 'sm_75'),  # Turing
            ('compute_80', 'sm_80'),  # Ampere
            ('compute_86', 'sm_86'),  # Ampere
        ]
        for compute_arch, sm_arch in common_archs:
            nvcc_flags.append(f'-gencode=arch={compute_arch},code={sm_arch}')
    
    # 添加其他有用的编译标志
    cxx_flags.extend([
        '-fPIC',
        '-Wno-unused-variable',
        '-Wno-unused-parameter'
    ])
    
    nvcc_flags.extend([
        '--use_fast_math',
        '--expt-relaxed-constexpr'
    ])
    
    return cxx_flags, nvcc_flags

def get_include_dirs():
    """获取包含目录"""
    include_dirs = []
    
    # PyTorch包含目录
    import torch.utils.cpp_extension
    include_dirs.extend(torch.utils.cpp_extension.include_paths())
    
    # 当前目录
    include_dirs.append(str(CSRC_DIR))
    
    return include_dirs

def get_source_files():
    """获取源文件列表"""
    cpp_files = list(CSRC_DIR.glob("*.cpp"))
    cu_files = list(CSRC_DIR.glob("*.cu"))
    
    print(f"C++源文件: {[f.name for f in cpp_files]}")
    print(f"CUDA源文件: {[f.name for f in cu_files]}")
    
    return [str(f) for f in cpp_files + cu_files]

def create_extension():
    """创建扩展模块"""
    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()
    
    # 检查是否强制CPU模式
    force_cpu = os.getenv('FORCE_CPU', '0') == '1'
    if force_cpu:
        print("强制使用CPU模式编译")
        cuda_available = False
    
    print(f"CUDA可用: {cuda_available}")
    
    # 获取源文件和编译标志
    sources = get_source_files()
    include_dirs = get_include_dirs()
    cxx_flags, nvcc_flags = get_compiler_flags()
    
    if cuda_available and any(f.endswith('.cu') for f in sources):
        # 创建CUDA扩展
        print("创建CUDA扩展...")
        extension = CUDAExtension(
            name='ms_deform_attn_ext',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': cxx_flags,
                'nvcc': nvcc_flags
            },
            define_macros=[
                ('WITH_CUDA', None),
                ('TORCH_EXTENSION_NAME', 'ms_deform_attn_ext')
            ]
        )
    else:
        # 创建CPU-only扩展
        print("创建CPU扩展...")
        # 只使用C++文件
        cpp_sources = [f for f in sources if f.endswith('.cpp')]
        extension = CppExtension(
            name='ms_deform_attn_ext',
            sources=cpp_sources,
            include_dirs=include_dirs,
            extra_compile_args=cxx_flags,
            define_macros=[
                ('TORCH_EXTENSION_NAME', 'ms_deform_attn_ext')
            ]
        )
    
    return extension

class InstallLocal(Command):
    """
    Custom command to build the extension in-place,
    creating the .so file in the same directory.
    """
    description = "build the C++/CUDA extension in-place"
    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""
        # We need to run the build_ext command with the --inplace option
        build_ext_cmd = self.get_finalized_command('build_ext')
        build_ext_cmd.inplace = True
        build_ext_cmd.run()


def main():
    """主函数"""
    print("=" * 60)
    print("MS Deformable Attention PyTorch扩展构建")
    print("=" * 60)
    
    # 检查依赖
    check_dependencies()
    
    # 创建扩展
    ext_module = create_extension()
    
    # 设置包信息
    setup(
        name='ms_deform_attn',
        version=get_version(),
        description='MS Deformable Attention PyTorch扩展',
        long_description='''
        高性能的多尺度可变形注意力CUDA实现，适用于RT-DETRv3等目标检测模型。
        提供与PyTorch自动求导系统完全兼容的接口。
        ''',
        author='RT-DETRv3 PyTorch Team',
        python_requires='>=3.6',
        ext_modules=[ext_module],
        cmdclass={
            'build_ext': BuildExtension.with_options(use_ninja=True),
            'install_local': InstallLocal,
        },
        zip_safe=False,
        install_requires=[
            'torch>=1.8.0',
            'numpy'
        ],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: C++',
            'Programming Language :: Python :: Implementation :: CPython',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )
    
    print("构建完成!")

if __name__ == '__main__':
    main()