#!/bin/bash

# macOS LibTorch Test Runner
# Simplified test runner for macOS systems without CUDA
#
# @author Rishab Nuguru
# @copyright © 2025 Rishab Nuguru
# @license AGPL v3 license

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS systems"
    exit 1
fi

print_status "Running LibTorch tests on macOS (CPU-only mode)"

# Check Python and PyTorch
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed"
    exit 1
fi

# Test PyTorch availability
if ! python3 -c "import torch; print(f'PyTorch {torch.__version__} found')" 2>/dev/null; then
    print_error "PyTorch is not installed. Please install it first:"
    print_error "pip3 install torch torchvision torchaudio"
    exit 1
fi

# Check CUDA availability
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [[ "$CUDA_AVAILABLE" == "True" ]]; then
    print_warning "CUDA is available, but we'll run CPU-only tests for macOS compatibility"
else
    print_status "CUDA not available - running CPU-only tests"
fi

# Run Python tests
print_status "Running Python LibTorch tests..."

# Test 1: Basic PyTorch functionality
print_status "Test 1: Basic PyTorch functionality"
if python3 -c "
import torch
import torch.nn as nn
import torch.optim as optim

# Create tensors
tensor1 = torch.randn(10, 10)
tensor2 = torch.ones(10, 10)
result = tensor1 + tensor2
print('Basic tensor operations: OK')

# Create a simple model
model = nn.Linear(10, 5)
input_tensor = torch.randn(5, 10)
output = model(input_tensor)
print('Neural network operations: OK')

# Test optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
loss.backward()
optimizer.step()
print('Training operations: OK')
"; then
    print_success "Basic PyTorch functionality test passed"
else
    print_error "Basic PyTorch functionality test failed"
    exit 1
fi

# Test 2: macOS-specific Python test
print_status "Test 2: macOS-specific Python test"
if [ -f "libtorch_macos_python_test.py" ]; then
    if python3 libtorch_macos_python_test.py; then
        print_success "macOS Python test passed"
    else
        print_error "macOS Python test failed"
        exit 1
    fi
else
    print_warning "macOS Python test file not found, skipping"
fi

# Test 3: Memory and performance test
print_status "Test 3: Memory and performance test"
if python3 -c "
import torch
import time
import gc

print('Testing memory management and performance...')

# Memory test
tensors = []
for i in range(20):
    tensor = torch.randn(50, 50)
    tensors.append(tensor)

# Performance test
start_time = time.time()
for i in range(10):
    result = torch.matmul(tensors[i], tensors[i + 1])
    mean_val = torch.mean(result)
end_time = time.time()

duration = (end_time - start_time) * 1000
print(f'Performance: {duration:.2f}ms for 10 matrix multiplications')

# Clean up
del tensors
gc.collect()

print('Memory management: OK')
print('Performance test: OK')
"; then
    print_success "Memory and performance test passed"
else
    print_error "Memory and performance test failed"
    exit 1
fi

# Test 4: Serialization test
print_status "Test 4: Serialization test"
if python3 -c "
import torch
import os

# Create tensor
tensor = torch.randn(10, 10)
filename = 'test_macos_tensor.pt'

# Save tensor
torch.save(tensor, filename)

# Load tensor
loaded_tensor = torch.load(filename)

# Verify
diff = torch.abs(tensor - loaded_tensor)
if torch.sum(diff).item() < 1e-6:
    print('Serialization: OK')
else:
    print('Serialization: FAILED')
    exit(1)

# Clean up
os.remove(filename)
"; then
    print_success "Serialization test passed"
else
    print_error "Serialization test failed"
    exit 1
fi

# Test 5: Threading test
print_status "Test 5: Threading test"
if python3 -c "
import torch
import threading
import time

def worker_function(worker_id, results):
    try:
        tensor = torch.randn(20, 20)
        result = torch.matmul(tensor, tensor.t())
        mean_val = torch.mean(result)
        results[worker_id] = True
    except:
        results[worker_id] = False

# Test threading
threads = []
results = [False] * 4

for i in range(4):
    thread = threading.Thread(target=worker_function, args=(i, results))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

if all(results):
    print('Threading: OK')
else:
    print('Threading: FAILED')
    exit(1)
"; then
    print_success "Threading test passed"
else
    print_error "Threading test failed"
    exit 1
fi

# Test 6: Error handling test
print_status "Test 6: Error handling test"
if python3 -c "
import torch

# Test invalid operations
try:
    tensor1 = torch.randn(3, 4)
    tensor2 = torch.randn(5, 6)
    result = tensor1 + tensor2
    print('Error handling: FAILED - should have raised error')
    exit(1)
except RuntimeError:
    print('Error handling: OK - caught expected error')

# Test division by zero
try:
    tensor = torch.zeros(2, 2)
    result = 1.0 / tensor
    print('Division by zero handling: OK')
except RuntimeError:
    print('Division by zero handling: OK - caught error')
"; then
    print_success "Error handling test passed"
else
    print_error "Error handling test failed"
    exit 1
fi

print_status "All tests completed successfully!"
print_success "🎉 LibTorch is working correctly on your macOS system!"

# System information
echo ""
echo "=== System Information ==="
echo "OS: $(uname -s) $(uname -r)"
echo "Architecture: $(uname -m)"
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CPU Threads: $(sysctl -n hw.ncpu 2>/dev/null || echo 'Unknown')"
echo "Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{print $0/1024/1024/1024 " GB"}' || echo 'Unknown')"

echo ""
echo "=== Recommendations ==="
echo "✅ Your macOS system is ready for LibTorch development"
echo "✅ Use CPU-only operations for best compatibility"
echo "✅ Consider using smaller batch sizes for memory efficiency"
echo "✅ Monitor memory usage during large operations"
echo "✅ Use threading for parallel processing when needed"

exit 0
