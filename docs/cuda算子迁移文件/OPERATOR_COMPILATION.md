# 自定义算子编译指南

本文档提供了编译和安装 `ms_deform_attn` 自定义CUDA算子的详细说明。

## 1. 环境要求

在开始编译之前，请确保您的开发环境满足以下要求：

- **PyTorch**: 版本 >= 1.8.0
- **CUDA Toolkit**: 版本需与 PyTorch 所支持的CUDA版本匹配 (例如, CUDA 11.x 或 12.x)
- **C++ 编译器**: GCC 版本 >= 7.5
- **Ninja (可选)**: 一个更快的构建系统。如果已安装，`setuptools` 会自动使用它来加速编译过程。

您可以通过以下命令安装 Ninja:
```bash
pip install ninja
```

## 2. 编译选项

我们提供了三种编译和安装算子的方法，请根据您的需求选择合适的命令。所有命令都应在 `src/ops/` 目录下执行。

```bash
cd src/ops
```

### 选项 A: 本地构建 (推荐用于开发)

此命令会在当前目录 (`src/ops/`) 下生成动态链接库 (`.so` 文件)，而不会将其安装到 Python 的 `site-packages` 目录。这种方式非常适合开发和打包，因为它将所有依赖项都保留在项目本地。

**命令:**
```bash
python setup.py install_local
```

执行成功后，您会在 `src/ops/` 目录下看到一个类似于 `ms_deform_attn_ext.cpython-310-x86_64-linux-gnu.so` 的文件。

### 选项 B: 本地编译 (In-place Build)

此命令与 `install_local` 效果类似，它也会在当前目录生成 `.so` 文件。这是 `setuptools` 的标准命令。

**命令:**
```bash
python setup.py build_ext --inplace
```

### 选项 C: 安装到环境中

此命令会将编译后的算子安装到您当前 Python 环境的 `site-packages` 目录中，使其成为一个可以全局导入的库。

**命令:**
```bash
python setup.py install
```

## 3. 强制CPU模式编译

如果您的环境没有可用的CUDA设备，或者您希望强制使用CPU进行编译，可以设置 `FORCE_CPU` 环境变量。

```bash
export FORCE_CPU=1
python setup.py install_local
```

## 4. 调试模式编译

如果需要进行调试，可以设置 `DEBUG` 环境变量来编译一个包含调试符号的版本。

```bash
export DEBUG=1
python setup.py install_local
```

## 5. 常见问题

- **编译错误**:
  - 确保您的 C++ 编译器和 CUDA Toolkit 版本与 PyTorch 版本兼容。
  - 检查 `g++` 版本，过高或过低的版本可能导致与CUDA不兼容。
- **`ninja` 未找到**:
  - 这是一个可选依赖，如果未安装，编译过程会回退到默认的 `distutils` 构建工具，速度会稍慢。