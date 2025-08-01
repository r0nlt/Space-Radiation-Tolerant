# LibTorch Setup Guide for macOS

This guide helps you set up and test LibTorch (PyTorch) on macOS systems, optimized for CPU-only operations.

## Prerequisites

- macOS 10.14 or later
- Python 3.7 or later
- pip3 package manager

## Installation

### 1. Install PyTorch for macOS

For CPU-only installation (recommended for macOS):

```bash
# Install PyTorch CPU version
pip3 install torch torchvision torchaudio

# Verify installation
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

### 2. Install Additional Dependencies

```bash
# Install common ML libraries
pip3 install numpy scipy matplotlib

# Install testing dependencies
pip3 install pytest psutil
```

## Testing Your Installation

### Quick Test

Run the macOS-specific test suite:

```bash
cd test
./run_macos_libtorch_tests.sh
```

### Manual Testing

You can also test manually:

```bash
# Test basic PyTorch functionality
python3 -c "
import torch
import torch.nn as nn
import torch.optim as optim

# Create tensors
tensor1 = torch.randn(10, 10)
tensor2 = torch.ones(10, 10)
result = tensor1 + tensor2
print('✅ Basic tensor operations work')

# Create a simple neural network
model = nn.Linear(10, 5)
input_tensor = torch.randn(5, 10)
output = model(input_tensor)
print('✅ Neural network operations work')

# Test training
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
loss.backward()
optimizer.step()
print('✅ Training operations work')

print('🎉 PyTorch is working correctly on macOS!')
"
```

## macOS-Specific Optimizations

### 1. CPU Threading

PyTorch automatically uses all available CPU cores on macOS. You can control this:

```python
import torch
import multiprocessing as mp

# Set number of threads
num_threads = mp.cpu_count()
torch.set_num_threads(num_threads)
print(f"Using {num_threads} CPU threads")
```

### 2. Memory Management

For better memory management on macOS:

```python
import torch
import gc

# After large operations, clean up memory
def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Example usage
large_tensor = torch.randn(1000, 1000)
result = torch.matmul(large_tensor, large_tensor.t())
del large_tensor
cleanup_memory()
```

### 3. Batch Size Optimization

Use smaller batch sizes for better memory efficiency:

```python
# Recommended batch sizes for macOS
BATCH_SIZE = 16  # Smaller than typical GPU batch sizes
LEARNING_RATE = 0.001  # Standard learning rate
```

## Performance Tips

### 1. Use Appropriate Tensor Sizes

```python
# Good for macOS
tensor_size = (100, 100)  # Reasonable size
large_tensor = torch.randn(*tensor_size)

# Avoid very large tensors
# tensor_size = (10000, 10000)  # May cause memory issues
```

### 2. Efficient Operations

```python
# Use in-place operations when possible
tensor1 = torch.randn(10, 10)
tensor2 = torch.randn(10, 10)

# Efficient
tensor1.add_(tensor2)  # In-place addition

# Less efficient
tensor1 = tensor1 + tensor2  # Creates new tensor
```

### 3. Threading for Parallel Processing

```python
import threading
import torch

def parallel_operation(worker_id, results):
    tensor = torch.randn(50, 50)
    result = torch.matmul(tensor, tensor.t())
    results[worker_id] = torch.mean(result).item()

# Run parallel operations
threads = []
results = [0] * 4

for i in range(4):
    thread = threading.Thread(target=parallel_operation, args=(i, results))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print(f"Results: {results}")
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Check available memory
   sysctl -n hw.memsize | awk '{print $0/1024/1024/1024 " GB"}'
   
   # Reduce tensor sizes if needed
   ```

2. **Performance Issues**
   ```python
   # Check CPU usage
   import psutil
   print(f"CPU usage: {psutil.cpu_percent()}%")
   
   # Monitor memory usage
   print(f"Memory usage: {psutil.virtual_memory().percent}%")
   ```

3. **Import Errors**
   ```bash
   # Reinstall PyTorch if needed
   pip3 uninstall torch torchvision torchaudio
   pip3 install torch torchvision torchaudio
   ```

### System Requirements

- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB+ RAM, 4+ CPU cores
- **Optimal**: 16GB+ RAM, 8+ CPU cores

## Advanced Configuration

### Environment Variables

```bash
# Set PyTorch to use specific number of threads
export OMP_NUM_THREADS=4

# Disable CUDA (force CPU usage)
export CUDA_VISIBLE_DEVICES=""
```

### PyTorch Configuration

```python
import torch

# Force CPU usage
torch.set_default_tensor_type(torch.FloatTensor)

# Set number of threads
torch.set_num_threads(4)

# Disable CUDA
torch.cuda.is_available = lambda: False
```

## Testing Suite

The macOS testing suite includes:

1. **Basic Functionality Tests**
   - Tensor operations
   - Neural network operations
   - Training operations

2. **Performance Tests**
   - Memory management
   - CPU utilization
   - Threading performance

3. **Compatibility Tests**
   - Serialization
   - Error handling
   - macOS-specific features

4. **Integration Tests**
   - Rad-ML framework integration
   - Radiation hardening features
   - Adaptive protection mechanisms

## Next Steps

After successful installation and testing:

1. **Explore the Framework**: Check out the rad_ml framework integration
2. **Run Examples**: Try the provided example scripts
3. **Custom Development**: Start building your own models
4. **Performance Tuning**: Optimize for your specific use case

## Support

For issues specific to macOS:

1. Check the troubleshooting section above
2. Review PyTorch macOS documentation
3. Consider system resource limitations
4. Test with smaller models first

---

**Note**: This setup is optimized for CPU-only operations on macOS. While CUDA is not available, the CPU implementation provides excellent performance for most machine learning tasks. 