# Manual Compilation Guide

This guide provides commands for manually compiling individual files in the rad_ml framework without using the CMake build system.

## Prerequisites

Make sure you have the following libraries installed:
- **Eigen3**: `brew install eigen`
- **PyTorch**: `brew install pytorch`

## Basic Compilation (Without PyTorch)

### Standard Test Compilation
```bash
g++ -std=c++17 \
    -Iinclude \
    -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3 \
    your_test_file.cpp \
    -o your_executable
```

### Example: Tensor Protection Test
```bash
g++ -std=c++17 \
    -Iinclude \
    -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3 \
    test/test_tensor_protection.cpp \
    -o test_tensor_protection
```

## PyTorch-Enabled Compilation

### With PyTorch Support
```bash
g++ -std=c++17 \
    -Iinclude \
    -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3 \
    -I/usr/local/Cellar/pytorch/2.5.1_4/libexec/lib/python3.13/site-packages/torch/include/torch/csrc/api/include \
    -DRAD_ML_PYTORCH_ENABLED \
    your_test_file.cpp \
    -o your_executable
```

### Example: PyTorch Tensor Protection Test
```bash
g++ -std=c++17 \
    -Iinclude \
    -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3 \
    -I/usr/local/Cellar/pytorch/2.5.1_4/libexec/lib/python3.13/site-packages/torch/include/torch/csrc/api/include \
    -DRAD_ML_PYTORCH_ENABLED \
    test/test_tensor_protection.cpp \
    -o test_tensor_protection_pytorch
```

## Common Include Paths

### Eigen3
```bash
-I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3
```

### PyTorch
```bash
-I/usr/local/Cellar/pytorch/2.5.1_4/libexec/lib/python3.13/site-packages/torch/include/torch/csrc/api/include
```

### Framework Headers
```bash
-Iinclude
```

## Compilation Flags

### Standard C++17
```bash
-std=c++17
```

### PyTorch Support
```bash
-DRAD_ML_PYTORCH_ENABLED
```

### Debug Information
```bash
-g -O0
```

### Optimization
```bash
-O2 -DNDEBUG
```

## Example Compilations

### 1. Basic TMR Test
```bash
g++ -std=c++17 \
    -Iinclude \
    -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3 \
    test/test_apply_protection.cpp \
    -o test_apply_protection
```

### 2. Comprehensive Protection Test
```bash
g++ -std=c++17 \
    -Iinclude \
    -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3 \
    test/test_apply_protection_comprehensive.cpp \
    -o test_apply_protection_comprehensive
```

### 3. Monte Carlo Validation
```bash
g++ -std=c++17 \
    -Iinclude \
    -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3 \
    test/verification/monte_carlo_validation.cpp \
    -o monte_carlo_validation
```

## Troubleshooting

### Eigen Not Found
If you get `fatal error: 'Eigen/Dense' file not found`:
1. Check if Eigen is installed: `brew list eigen`
2. Find Eigen path: `find /usr -name "Eigen" -type d 2>/dev/null`
3. Add the correct include path

### PyTorch Not Found
If you get `fatal error: 'torch/torch.h' file not found`:
1. Check if PyTorch is installed: `brew list pytorch`
2. Find PyTorch include path: `find /usr/local -name "torch.h" 2>/dev/null`
3. Add the correct include path

### Namespace Errors
If you get namespace errors like `use of undeclared identifier 'neural'`:
1. Make sure to include the correct headers
2. Use fully qualified names: `rad_ml::neural::ProtectionLevel`
3. Check that all required headers are included

## Quick Reference

### Minimal Compilation
```bash
g++ -std=c++17 -Iinclude -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3 file.cpp -o executable
```

### Full PyTorch Compilation
```bash
g++ -std=c++17 \
    -Iinclude \
    -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3 \
    -I/usr/local/Cellar/pytorch/2.5.1_4/libexec/lib/python3.13/site-packages/torch/include/torch/csrc/api/include \
    -DRAD_ML_PYTORCH_ENABLED \
    file.cpp \
    -o executable
```

## Notes

- Manual compilation is useful for quick testing and debugging
- For production builds, use the CMake system: `make` or `cmake --build build/`
- The CMake system handles all dependencies automatically
- Manual compilation requires you to specify all include paths manually
