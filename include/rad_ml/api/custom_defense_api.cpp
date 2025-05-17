#include "custom_defense_api.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace rad_ml {
namespace api {
namespace examples {

// Example 1: Basic custom protection strategy implementation
class CustomBitInterleaving : public tmr::ProtectionStrategy {
   private:
    size_t interleave_factor_;

   public:
    CustomBitInterleaving(const DefenseConfig& config) : tmr::ProtectionStrategy()
    {
        // Get custom parameter or use default
        interleave_factor_ = 8;
        if (config.custom_params.count("interleave_factor")) {
            interleave_factor_ = std::stoi(config.custom_params.at("interleave_factor"));
        }
    }

    // Template method to protect a value
    template <typename T>
    tmr::ProtectionResult<T> protect(const T& value)
    {
        // In a real implementation, this would do bit interleaving
        // For demonstration, we'll just make a TMR-like copy
        tmr::ProtectionResult<T> result;
        result.value = value;
        result.error_detected = false;
        result.error_corrected = false;
        return result;
    }

    void updateEnvironment(const sim::RadiationEnvironment& env) override
    {
        // Adjust protection based on environment
        // For example, increase interleaving in harsher environments
        if (env == sim::RadiationEnvironment::JUPITER ||
            env == sim::RadiationEnvironment::SOLAR_PROBE) {
            interleave_factor_ = 16;  // Higher for harsh environments
        }
        else {
            interleave_factor_ = 8;  // Default for milder environments
        }
    }
};

// Register the custom strategy with the factory
bool registerCustomStrategies()
{
    return UnifiedDefenseSystem::registerCustomStrategy(
        "bit_interleaving", [](const DefenseConfig& config) {
            return std::make_unique<CustomBitInterleaving>(config);
        });
}

// Example 2: Simple neural network implementation
class SimpleNN {
   private:
    std::vector<float> weights1_;
    std::vector<float> weights2_;
    std::vector<float> biases1_;
    std::vector<float> biases2_;
    size_t input_size_;
    size_t hidden_size_;
    size_t output_size_;

   public:
    SimpleNN(size_t input_size, size_t hidden_size, size_t output_size)
        : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size)
    {
        // Initialize with random weights (simplified)
        weights1_.resize(input_size * hidden_size, 0.1f);
        weights2_.resize(hidden_size * output_size, 0.1f);
        biases1_.resize(hidden_size, 0.0f);
        biases2_.resize(output_size, 0.0f);
    }

    std::vector<float> forward(const std::vector<float>& input)
    {
        // Forward pass through network
        std::vector<float> hidden(hidden_size_, 0.0f);

        // Input to hidden layer
        for (size_t i = 0; i < hidden_size_; ++i) {
            for (size_t j = 0; j < input_size_; ++j) {
                hidden[i] += input[j] * weights1_[j * hidden_size_ + i];
            }
            hidden[i] += biases1_[i];
            // ReLU activation
            hidden[i] = std::max(0.0f, hidden[i]);
        }

        // Hidden to output layer
        std::vector<float> output(output_size_, 0.0f);
        for (size_t i = 0; i < output_size_; ++i) {
            for (size_t j = 0; j < hidden_size_; ++j) {
                output[i] += hidden[j] * weights2_[j * output_size_ + i];
            }
            output[i] += biases2_[i];
        }

        return output;
    }
};

// Example 3: Using the custom API with neural networks
void demonstrateCustomAPI()
{
    // Register our custom strategy
    registerCustomStrategies();

    // Create a defense configuration for Jupiter environment
    DefenseConfig jupiter_config = DefenseConfig::forEnvironment(sim::Environment::JUPITER);

    // Create a defense configuration with custom strategy
    DefenseConfig custom_config;
    custom_config.strategy = DefenseStrategy::CUSTOM;
    custom_config.custom_params["name"] = "bit_interleaving";
    custom_config.custom_params["interleave_factor"] = "16";

    // Create a neural network
    auto neural_network = std::make_unique<SimpleNN>(10, 20, 2);

    // Wrap it with radiation protection
    auto protected_nn = wrapExistingNN(std::move(neural_network), jupiter_config);

    // Use the protected neural network
    std::vector<float> input(10, 0.5f);
    auto output = protected_nn->forward(input);

    std::cout << "Output: ";
    for (auto val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Create directly with helper function
    auto another_nn = createRadiationHardenedNN<SimpleNN>(custom_config, 5, 10, 1);

    // Use the second network
    std::vector<float> another_input(5, 0.1f);
    auto another_output = another_nn->forward(another_input);

    std::cout << "Another output: " << another_output[0] << std::endl;

    // Create a unified defense system for protecting individual values
    UnifiedDefenseSystem defense_system(jupiter_config);

    // Protect individual values
    auto protected_value = defense_system.protectValue(3.14159f);

    // Execute a function with protection
    auto result = defense_system.executeProtected<float>([]() { return std::sqrt(2.0f); });

    std::cout << "Protected computation result: " << result.value << std::endl;
    if (result.error_detected) {
        std::cout << "Error was detected"
                  << (result.error_corrected ? " and corrected" : " but not corrected")
                  << std::endl;
    }
}

}  // namespace examples
}  // namespace api
}  // namespace rad_ml
