# Protecting a Simple Calculation

```cpp
// Define a simple function to protect
auto calculation = [](float x, float y) -> float {
    return x * y + std::sqrt(x) / y;  // Could have radiation-induced errors
};

// Protect it against radiation effects
float result = protection.executeProtected<float>([&]() {
    return calculation(3.14f, 2.71f);
}).value;
```
