# PyTorch C++ Extension Example

This directory contains a simple example of a PyTorch C++ extension, based on the official PyTorch tutorials.

## Files

- `lltm.cpp`: The C++ source code for the custom "Long-term Memory" (LLTM) module. This file defines the forward and backward passes of the operation.
- `setup.py`: The build script used to compile the C++ code into a Python module that can be imported in PyTorch.
- `run.py`: An example Python script that imports the compiled module and runs a forward pass to demonstrate its usage.

## How to Run

1.  **Build the C++ Extension**:
    Navigate to this directory and run the following command. This will compile `lltm.cpp` into a Python-importable module.

    ```bash
    cd examples/cpp_extension_example
    python setup.py install
    ```

2.  **Run the Example**:
    After the build is complete, you can run the example script:

    ```bash
    python run.py
    ```

    If successful, you will see a confirmation message with the output tensor shapes.

    **Note**: This example requires a CUDA-enabled GPU to run, as specified by the checks within the C++ code.
