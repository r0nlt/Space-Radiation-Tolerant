#include <iostream>
#include <torch/torch.h>

int main() {
    std::cout << "=== LibTorch Standalone Test ===" << std::endl;
    
    try {
        // Test 1: Basic tensor operations
        std::cout << "\n1. Testing basic tensor operations..." << std::endl;
        
        auto tensor = torch::randn({3, 4});
        std::cout << "Created tensor:\n" << tensor << std::endl;
        
        auto tensor2 = torch::ones({3, 4});
        std::cout << "Created ones tensor:\n" << tensor2 << std::endl;
        
        auto result = tensor + tensor2;
        std::cout << "Addition result:\n" << result << std::endl;
        
        // Test 2: Mathematical operations
        std::cout << "\n2. Testing mathematical operations..." << std::endl;
        
        auto sin_result = torch::sin(tensor);
        std::cout << "Sin of tensor:\n" << sin_result << std::endl;
        
        auto mean_val = torch::mean(tensor);
        std::cout << "Mean of tensor: " << mean_val.item<float>() << std::endl;
        
        // Test 3: Neural network
        std::cout << "\n3. Testing neural network..." << std::endl;
        
        torch::nn::Linear linear(torch::nn::LinearOptions(4, 2));
        std::cout << "Created linear layer" << std::endl;
        
        auto input = torch::randn({2, 4});
        auto output = linear->forward(input);
        std::cout << "Input shape: " << input.sizes() << std::endl;
        std::cout << "Output shape: " << output.sizes() << std::endl;
        std::cout << "Output:\n" << output << std::endl;
        
        // Test 4: Optimizer
        std::cout << "\n4. Testing optimizer..." << std::endl;
        
        torch::optim::SGD optimizer(linear->parameters(), torch::optim::SGDOptions(0.01));
        std::cout << "Created SGD optimizer" << std::endl;
        
        // Test 5: CUDA availability
        std::cout << "\n5. Testing CUDA availability..." << std::endl;
        
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available!" << std::endl;
            std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
            
            // Test moving tensor to GPU
            auto gpu_tensor = tensor.to(torch::kCUDA);
            std::cout << "Moved tensor to GPU successfully" << std::endl;
            std::cout << "GPU tensor device: " << gpu_tensor.device() << std::endl;
        } else {
            std::cout << "CUDA is not available, using CPU only" << std::endl;
        }
        
        // Test 6: Complex operations
        std::cout << "\n6. Testing complex operations..." << std::endl;
        
        auto matrix = torch::randn({5, 5});
        auto eigenvals = torch::linalg::eigvals(matrix);
        std::cout << "Eigenvalues of 5x5 matrix:\n" << eigenvals << std::endl;
        
        // Test 7: Serialization
        std::cout << "\n7. Testing serialization..." << std::endl;
        
        torch::save(linear, "test_model.pt");
        std::cout << "Saved model to test_model.pt" << std::endl;
        
        torch::nn::Linear loaded_linear(torch::nn::LinearOptions(4, 2));
        torch::load(loaded_linear, "test_model.pt");
        std::cout << "Loaded model from test_model.pt" << std::endl;
        
        // Test 8: Version info
        std::cout << "\n8. LibTorch version info..." << std::endl;
        std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;
        
        std::cout << "\n✅ All LibTorch tests passed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 