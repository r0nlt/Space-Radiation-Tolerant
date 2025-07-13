# Using Adaptive Protection Strategy

```cpp
#include "rad_ml/neural/adaptive_protection.hpp"

// Create adaptive protection with default settings
neural::AdaptiveProtection protection;

// Configure for current environment
protection.setRadiationEnvironment(sim::createEnvironment(sim::Environment::MARS));
protection.setBaseProtectionLevel(neural::ProtectionLevel::MODERATE);

// Protect a neural network weight matrix
std::vector<float> weights = /* your neural network weights */;
auto protected_weights = protection.protectValue(weights);

// Later, recover the weights (with automatic error correction)
auto recovered_weights = protection.recoverValue(protected_weights);

// Check protection statistics
auto stats = protection.getProtectionStats();
std::cout << "Errors detected: " << stats.errors_detected << std::endl;
std::cout << "Errors corrected: " << stats.errors_corrected << std::endl;
```
