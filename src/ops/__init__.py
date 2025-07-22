"""
PyTorch自定义算子模块
为RT-DETRv3提供高性能的CUDA算子实现
"""

import torch
import logging
import warnings
from typing import Optional

__version__ = "1.0.0"


def check_torch_version() -> bool:
    """
    检查PyTorch版本兼容性
    
    Returns:
        bool: 版本是否兼容
    """
    required_version = "1.8.0"
    current_version = torch.__version__.split('+')[0]  # 移除CUDA版本后缀
    
    try:
        from packaging import version
        return version.parse(current_version) >= version.parse(required_version)
    except ImportError:
        # 简单的版本比较作为后备方案
        return current_version >= required_version


def check_cuda_availability() -> bool:
    """
    检查CUDA是否可用
    
    Returns:
        bool: CUDA是否可用
    """
    return torch.cuda.is_available()


def setup_compilation_logging() -> None:
    """
    设置编译过程的日志记录
    """
    logger = logging.getLogger('torch.utils.cpp_extension')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


# 在模块导入时进行版本检查
if not check_torch_version():
    warnings.warn(
        f"PyTorch版本 {torch.__version__} 可能不兼容，建议使用 >= 1.8.0",
        UserWarning
    )

if not check_cuda_availability():
    warnings.warn(
        "CUDA不可用，自定义算子将无法使用GPU加速",
        UserWarning
    )

# 设置编译日志
setup_compilation_logging()

# 尝试导入编译好的扩展
_ms_deform_attn_ext: Optional[object] = None

try:
    from . import ms_deform_attn
    _ms_deform_attn_available = True
except ImportError as e:
    _ms_deform_attn_available = False
    import warnings
    warnings.warn(
        f"MS Deformable Attention扩展未编译或导入失败: {e}\n"
        "请运行以下命令编译扩展:\n"
        "  cd src/ops && python setup.py build_ext --inplace",
        ImportWarning
    )


def is_ms_deform_attn_available() -> bool:
    """
    检查MS Deformable Attention算子是否可用
    
    Returns:
        bool: 算子是否可用
    """
    return _ms_deform_attn_available


__all__ = [
    'check_torch_version',
    'check_cuda_availability', 
    'setup_compilation_logging',
    'is_ms_deform_attn_available',
    '__version__'
]

# 如果MS Deformable Attention可用，导出其接口
if _ms_deform_attn_available:
    from .ms_deform_attn import ms_deform_attn
    __all__.append('ms_deform_attn')