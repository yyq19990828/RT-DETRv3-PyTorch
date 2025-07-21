# PyTorch Examples

This directory contains various examples demonstrating different features and use cases of PyTorch, ranging from basic neural networks to advanced concepts like C++ extensions and full object detection models.

## Table of Contents

1. [Basic PyTorch Scripts](#basic-pytorch-scripts)
   - [Simple CNN](#simple-cnn)
   - [Self-Attention Mechanism](#self-attention-mechanism)
2. [RT-DETRv2 PyTorch Implementation](#rt-detrv2-pytorch-implementation)
3. [PyTorch C++ Extension Example](#pytorch-c-extension-example)

---

### Basic PyTorch Scripts

These are standalone scripts demonstrating fundamental PyTorch concepts.

#### Simple CNN

- **File**: `pytorch_nn_example.py`
- **Description**: Implements a basic Convolutional Neural Network (CNN) using `torch.nn.Module`. It's a good starting point for understanding model definition, forward passes, and loss calculations in PyTorch.
- **To Run**:
  
  ```bash
  python pytorch_nn_example.py
  ```

#### Self-Attention Mechanism

- **File**: `pytorch_self_attention_example.py`
- **Description**: Provides a clear implementation of the scaled dot-product attention mechanism, a core component of Transformer models.
- **To Run**:
  
  ```bash
  python pytorch_self_attention_example.py
  ```

---

### RT-DETRv2 PyTorch Implementation

- **Directory**: `rtdetrv2_pytorch/`
- **Description**: A complete and structured implementation of the RT-DETRv2 object detection model in PyTorch. This example is much more complex and includes configuration files, training/evaluation scripts, and model definitions.
- **Details**: For comprehensive instructions on setup, training, and usage, please refer to the dedicated README within the directory:
  - [RT-DETRv2 PyTorch README](./rtdetrv2_pytorch/README.md)

---

### PyTorch C++ Extension Example

- **Directory**: `cpp_extension_example/`
- **Description**: This example demonstrates how to build a custom neural network operation in C++ and integrate it into PyTorch. This is useful for performance-critical code or for integrating existing C++ libraries.
- **Details**: The directory contains the C++ source (`lltm.cpp`), a `setup.py` for compilation, and a `run.py` script for execution. For build and run instructions, see the README inside the directory:
  - [C++ Extension README](./cpp_extension_example/README.md)
