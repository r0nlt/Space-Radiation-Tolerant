# PyTorch Integration Guide

This document describes how to use the PyTorch integration with the rad_ml framework for radiation-tolerant machine learning, including detailed CMake build system integration.

## Overview

The PyTorch integration provides seamless integration between PyTorch models and the rad_ml framework's radiation hardening and TMR (Triple Modular Redundancy) protection features. This allows you to:

- Protect PyTorch tensors with radiation hardening
- Apply TMR protection to neural network models
- Validate model integrity during training
- Detect and correct radiation-induced errors

## Integration Architecture

### 1. Adaptive Protection Integration

The PyTorch integration seamlessly works with the comprehensive adaptive protection system:

```cpp
// PyTorch tensors can use all protection levels
neural::AdaptiveProtection<float> protection;
protection.set_protection_level(neural::ProtectionLevel::HIGH); // Reed-Solomon

// Apply to PyTorch tensors
auto tensor = torch::randn({10, 10});
auto protected_tensor = protection.protect_tensor(tensor);

// Radiation effects simulation
auto irradiated_tensor = protection.apply_radiation_effects(protected_tensor, 0.01);

// Error correction
auto [recovered_tensor, was_corrected] = protection.recover_tensor(irradiated_tensor);
```

### 2. Thread Safety Integration

The PyTorch integration inherits the thread-safe design from the adaptive protection system:

```cpp
// Thread-safe PyTorch operations
std::vector<std::thread> threads;
for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&]() {
        auto tensor = torch::randn({100, 100});
        auto protected_tensor = integration.create_protected_tensor(tensor);
        // Thread-local RNG ensures no race conditions
        protected_tensor.apply_radiation_hardening();
    });
}
```

### 3. Multi-Bit Protection Support

PyTorch tensors support all multi-bit upset types:

```cpp
// Configure multi-bit protection
protection.set_error_model(neural::MultibitUpsetType::ADJACENT_BITS);

// Apply to PyTorch model weights
for (auto& param : model.parameters()) {
    auto protected_param = protection.protect_value(param);
    param = protected_param;
}
```

## Build System Files

### File Structure

```
rad_ml/
├── cmake/Modules/
│   └── FindPyTorch.cmake              # PyTorch finder module
├── include/rad_ml/pytorch/
│   └── pytorch_integration.hpp        # Main integration header
├── src/rad_ml/pytorch/
│   ├── CMakeLists.txt                 # PyTorch module build
│   ├── pytorch_integration.cpp        # Core integration
│   ├── pytorch_model_protection.cpp   # Model protection
│   ├── pytorch_radiation_hardening.cpp # Radiation hardening
│   └── pytorch_tensor_wrapper.cpp     # Tensor wrapper
├── test/
│   └── pytorch_integration_test.cpp   # Integration tests
├── tools/
│   └── build_with_pytorch.sh          # Automated build script
└── CMakeLists.txt                     # Main build (updated)
```

### CMake Integration Points

1. **FindPyTorch Module**: Custom PyTorch detection
2. **Main CMakeLists.txt**: Optional PyTorch integration
3. **PyTorch Module**: Dedicated build configuration
4. **Test Integration**: PyTorch-aware test system
5. **Dependency Management**: Proper linking and include paths

### Build Configuration Examples

```bash
# Development build with PyTorch
mkdir build_dev
cd build_dev
cmake .. -DENABLE_PYTORCH=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Production build without PyTorch
mkdir build_prod
cd build_prod
cmake .. -DENABLE_PYTORCH=OFF -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Custom PyTorch installation
mkdir build_custom
cd build_custom
cmake .. -DENABLE_PYTORCH=ON -DPYTORCH_ROOT=/opt/custom/libtorch
make -j$(nproc)
```

## Testing Integration

### PyTorch Integration Tests

The build system includes comprehensive PyTorch integration tests:

```cpp
// test/pytorch_integration_test.cpp
#include <rad_ml/pytorch/pytorch_integration.hpp>
#include <gtest/gtest.h>

TEST(PyTorchIntegration, TensorProtection) {
    auto tensor = torch::randn({5, 5});
    auto protected_tensor = integration.create_protected_tensor(tensor);

    // Test radiation hardening
    protected_tensor.apply_radiation_hardening();

    // Test integrity validation
    EXPECT_TRUE(protected_tensor.validate_integrity());
}

TEST(PyTorchIntegration, ModelProtection) {
    auto model = std::make_unique<MyRadiationHardenedModel>();

    // Test weight protection
    model->apply_weight_protection();

    // Test forward pass protection
    auto input = torch::randn({1, 10});
    auto output = model->forward_protected(input);

    EXPECT_FALSE(torch::isnan(output).any().item<bool>());
}
```

### Test Execution

```bash
# Run PyTorch integration tests
cd build_pytorch
ctest -R pytorch_integration_test

# Run all tests including PyTorch
ctest --output-on-failure
```

## Troubleshooting

### Common Build Issues

1. **PyTorch Not Found**
   ```bash
   # Solution: Set PYTORCH_ROOT environment variable
   export PYTORCH_ROOT=/path/to/libtorch
   cmake .. -DENABLE_PYTORCH=ON
   ```

2. **Linker Errors**
   ```bash
   # Solution: Ensure proper library linking
   cmake .. -DENABLE_PYTORCH=ON -DCMAKE_VERBOSE_MAKEFILE=ON
   ```

3. **Include Path Issues**
   ```bash
   # Solution: Check include directories
   cmake .. -DENABLE_PYTORCH=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
   ```

### Debug Configuration

```bash
# Debug build with verbose output
mkdir build_debug
cd build_debug
cmake .. -DENABLE_PYTORCH=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=ON
make VERBOSE=1
```

## Performance Considerations

### Memory Overhead

- **Basic Protection**: ~12.5% overhead (parity)
- **Moderate Protection**: ~75% overhead (Hamming)
- **High Protection**: ~200% overhead (Reed-Solomon)
- **Very High Protection**: ~400% overhead (Reed-Solomon 16)

### Computational Overhead

- **Tensor Protection**: Minimal for basic operations
- **Model Protection**: Moderate for complex models
- **Radiation Hardening**: Configurable based on protection level

### Optimization Strategies

1. **Selective Protection**: Protect only critical tensors
2. **Adaptive Levels**: Use different protection levels for different components
3. **Batch Processing**: Apply protection to batches of tensors
4. **Lazy Evaluation**: Defer protection until needed

## Future Enhancements

### Planned Features

1. **GPU Acceleration**: CUDA-enabled protection mechanisms
2. **Dynamic Protection**: Runtime protection level adjustment
3. **Memory Pooling**: Optimized memory management for protected tensors
4. **Distributed Protection**: Multi-GPU protection coordination

### Integration Roadmap

1. **Phase 1**: Basic PyTorch integration ✅
2. **Phase 2**: Advanced protection features 🔧
3. **Phase 3**: Performance optimization 📈
4. **Phase 4**: Production deployment 🚀
