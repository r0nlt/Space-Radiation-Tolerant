# Testing LibTorch Standalone

This guide shows you how to test LibTorch functionality independently of the rad_ml framework to ensure your PyTorch installation is working correctly.

## Overview

Before using the rad_ml framework's PyTorch integration, it's important to verify that LibTorch is properly installed and functioning. This guide provides both C++ and Python test methods.

## Prerequisites

- CMake 3.16 or higher
- C++17 compatible compiler
- Python 3.7+ (for Python tests)
- LibTorch installation

## Quick Test

### Option 1: Automated C++ Test (Recommended)

Run the automated test script:

```bash
./test_libtorch.sh
```

This script will:
- Detect your LibTorch installation
- Build a test executable
- Run comprehensive tests
- Report results

### Option 2: Python Test

Run the Python test script:

```bash
python3 test_libtorch_python.py
```

## What the Tests Cover

### C++ Tests (`test_libtorch_standalone.cpp`)

1. **Basic Tensor Operations**
   - Tensor creation (`torch::randn`, `torch::ones`)
   - Arithmetic operations (addition, multiplication)
   - Shape manipulation

2. **Mathematical Operations**
   - Trigonometric functions (`torch::sin`)
   - Statistical functions (`torch::mean`)
   - Linear algebra operations

3. **Neural Network**
   - Linear layers (`torch::nn::Linear`)
   - Forward pass
   - Parameter management

4. **Optimization**
   - SGD optimizer
   - Parameter updates

5. **CUDA Support**
   - CUDA availability check
   - GPU tensor operations
   - Device management

6. **Complex Operations**
   - Eigenvalue computation
   - Matrix operations

7. **Serialization**
   - Model saving/loading
   - File I/O

8. **Version Information**
   - LibTorch version display

### Python Tests (`test_libtorch_python.py`)

1. **Basic Tensor Operations**
   - Tensor creation and manipulation
   - Mathematical operations

2. **Neural Network**
   - Sequential models
   - Custom layers
   - Forward/backward passes

3. **Optimization**
   - SGD optimizer
   - Training steps
   - Loss computation

4. **CUDA Support**
   - GPU availability
   - Device management

5. **Complex Operations**
   - Linear algebra (SVD, eigenvalues)
   - Matrix operations

6. **Serialization**
   - Model state dict saving/loading

7. **Version Information**
   - PyTorch version
   - CUDA version
   - CUDNN version

## Manual Testing

### C++ Manual Test

If you prefer to test manually:

```bash
# Create build directory
mkdir build_test
cd build_test

# Copy test files
cp ../test_libtorch_standalone.cpp .
cp ../CMakeLists_test_libtorch.txt CMakeLists.txt

# Configure and build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch .
make

# Run test
./test_libtorch_standalone
```

### Python Manual Test

```python
import torch
import torch.nn as nn

# Test basic functionality
tensor = torch.randn(3, 4)
print(f"Tensor created: {tensor.shape}")

# Test neural network
model = nn.Linear(4, 2)
output = model(tensor)
print(f"Model output: {output.shape}")

print("LibTorch is working!")
```

## Expected Output

### Successful C++ Test

```
=== LibTorch Standalone Test ===

1. Testing basic tensor operations...
Created tensor:
 0.0996 -0.3437 -0.7140  0.0307
 0.0011  1.9249  0.2291 -0.0785
-2.4912  0.9140  2.2838  0.5835
[ CPUFloatType{3,4} ]

2. Testing mathematical operations...
Sin of tensor:
 0.0994 -0.3369 -0.6548  0.0306
 0.0011  0.9379  0.2271 -0.0784
-0.6055  0.7919  0.7564  0.5510
[ CPUFloatType{3,4} ]
Mean of tensor: 0.203285

3. Testing neural network...
Created linear layer
Input shape: [2, 4]
Output shape: [2, 2]
Output:
 0.4284  0.7334
-0.4212  0.7643
[ CPUFloatType{2,2} ]

4. Testing optimizer...
Created SGD optimizer

5. Testing CUDA availability...
CUDA is not available, using CPU only

6. Testing complex operations...
Eigenvalues of 5x5 matrix:
-1.4676
-1.4676
-0.6036
-0.6036
 0.3540
[ CPUComplexFloatType{5} ]

7. Testing serialization...
Saved model to test_model.pt
Loaded model from test_model.pt

8. LibTorch version info...
LibTorch version: 2.1.0

✅ All LibTorch tests passed successfully!
```

### Successful Python Test

```
=== LibTorch Python Standalone Test ===
1. Testing basic tensor operations...
Created tensor:
tensor([[ 0.4758,  0.4421,  1.4236,  0.0024],
        [-1.4197, -1.6453,  1.5260,  0.0868],
        [-2.2158, -0.0207, -0.1684,  0.7685]])

2. Testing neural network...
Created model: Sequential(
  (0): Linear(in_features=4, out_features=2, bias=True)
  (1): ReLU()
  (2): Linear(in_features=2, out_features=1, bias=True)
)

3. Testing optimizer...
Created SGD optimizer with lr=0.01
Training step completed, loss: 0.850029

4. Testing CUDA availability...
CUDA is not available, using CPU only

5. Testing complex operations...
Eigenvalues of 5x5 matrix:
tensor([ 1.4847+1.3502j,  1.4847-1.3502j, -2.0307+0.0000j, -0.8808+0.5497j,
        -0.8808-0.5497j])

6. Testing serialization...
Saved model to test_model_python.pt
Loaded model from test_model_python.pt
Serialization test passed - models produce same output

7. PyTorch version info...
PyTorch version: 2.2.2
CUDA version: None
CUDNN version: None

✅ All LibTorch Python tests passed successfully!
```

## Troubleshooting

### Common Issues

1. **Library Not Found**
   ```
   dyld: Library not loaded: @rpath/libc10.dylib
   ```
   **Solution**: Set `DYLD_LIBRARY_PATH` to include LibTorch lib directory

2. **CMake Configuration Failed**
   ```
   CMake Error: Could not find Torch
   ```
   **Solution**: Set `CMAKE_PREFIX_PATH` to LibTorch installation directory

3. **Python Import Error**
   ```
   ModuleNotFoundError: No module named 'torch'
   ```
   **Solution**: Install PyTorch via pip or conda

### Environment Variables

For macOS:
```bash
export DYLD_LIBRARY_PATH=/path/to/libtorch/lib:$DYLD_LIBRARY_PATH
```

For Linux:
```bash
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

### CMake Variables

```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DTorch_DIR=/path/to/libtorch/share/cmake/Torch \
      .
```

## Next Steps

Once LibTorch is working correctly, you can:

1. **Use the rad_ml PyTorch integration**:
   ```bash
   ./tools/build_with_pytorch.sh
   ```

2. **Build the full framework**:
   ```bash
   mkdir build
   cd build
   cmake .. -DENABLE_PYTORCH=ON
   make
   ```

3. **Run integration tests**:
   ```bash
   ctest -R pytorch_integration_test
   ```

## Files Created

- `test_libtorch_standalone.cpp` - C++ test program
- `test_libtorch_python.py` - Python test program
- `CMakeLists_test_libtorch.txt` - CMake configuration
- `test_libtorch.sh` - Automated test script
- `build_libtorch_test/` - Build directory (created automatically)

## Cleanup

To clean up test files:

```bash
rm -rf build_libtorch_test
rm -f test_model.pt test_model_python.pt
```

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your LibTorch installation
3. Ensure all dependencies are installed
4. Check system-specific requirements for your platform
