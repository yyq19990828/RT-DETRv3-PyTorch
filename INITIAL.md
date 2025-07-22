# INITIAL.md - C++ CUDA算子迁移项目初始化指南

## 项目特性 (FEATURE):

- 从Paddle框架迁移C++ CUDA算子到PyTorch框架
- MultiScale Deformable Attention算子的完整迁移实现
- PyTorch C++扩展的构建和集成系统
- 性能基准测试和验证工具
- 自动求导支持和Python绑定接口

## 示例参考 (EXAMPLES):

在 `examples/` 文件夹中，您可以找到以下参考实现来理解项目结构和最佳实践：

- `examples/ext_op/` - 当前已存在的CUDA算子实现，包含：
  - `ms_deformable_attn_op.cc` - C++主实现文件
  - `ms_deformable_attn_op.cu` - CUDA核函数实现
  - `setup_ms_deformable_attn_op.py` - 构建配置脚本
  - `test_ms_deformable_attn_op.py` - 测试验证脚本

- `examples/cpp_extension_example/` - PyTorch C++扩展示例：
  - `lltm.cpp` - 简单的C++扩展示例
  - `setup.py` - 标准PyTorch扩展构建脚本
  - `run.py` - 使用示例

不要直接复制这些示例，它们是为不同目的设计的。请将其作为最佳实践的参考和灵感来源。

## 技术文档 (DOCUMENTATION):

### 核心参考文档:
- **Paddle自定义算子文档**: https://www.paddlepaddle.org.cn/documentation/docs/zh/3.0-beta/guides/custom_op/index_cn.html
- **PyTorch C++扩展教程**: https://docs.pytorch.org/tutorials/advanced/cpp_extension.html
- **PyTorch C++ API文档**: https://pytorch.org/cppdocs/
- **CUDA编程指南**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

### 扩展阅读:
- PyTorch自定义算子最佳实践
- CUDA性能优化指南
- C++与Python互操作

## 其他考虑因素 (OTHER CONSIDERATIONS):

### 开发环境设置:
- 创建 `.env.example` 文件，包含必要的环境变量模板
- 在README中包含详细的安装和配置说明
- 虚拟环境已配置必要的依赖项
- 使用 `python_dotenv` 和 `load_dotenv()` 管理环境变量

### 项目结构要求:
- 在README中包含完整的项目目录结构说明
- 遵循模块化设计原则，每个文件不超过500行
- 实现完整的测试覆盖
- 提供性能基准和验证工具

### 技术要求:
- **CUDA兼容性**: 支持CUDA 10.2+
- **PyTorch版本**: 兼容PyTorch 1.8.0+
- **C++标准**: 使用C++14及以上
- **跨平台支持**: 优先支持Linux，考虑Windows兼容性

### 质量标准:
- 完整的单元测试和集成测试
- 性能不低于原Paddle实现
- 详细的API文档和使用示例
- 迁移过程的完整记录

### 构建和部署:
- 自动化构建脚本
- CI/CD集成考虑
- 性能回归测试
- 内存和计算效率优化

## 迁移路线图:

1. **第一阶段**: 环境准备和代码分析
2. **第二阶段**: 核心算子迁移实现
3. **第三阶段**: 测试验证和性能优化
4. **第四阶段**: 文档完善和集成部署

此初始化指南将帮助您建立一个结构化、高质量的C++ CUDA算子迁移项目。