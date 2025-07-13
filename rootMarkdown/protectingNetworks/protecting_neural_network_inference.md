# Protecting Neural Network Inference

```cpp
// Protect a neural network forward pass
auto protected_inference = [&](const std::vector<float>& input) -> std::vector<float> {
    // Create a wrapper for your neural network inference
    return protection.executeProtected<std::vector<float>>([&]() {
        return neural_network.forward(input);
    }).value;
};

// Use the protected inference function
std::vector<float> output = protected_inference(input_data);
```
