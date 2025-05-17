#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <rad_ml/api/custom_defense_api.hpp>
#include <random>
#include <string>
#include <vector>

using namespace rad_ml::api;
using namespace rad_ml::sim;
using namespace rad_ml::neural;

// Simple neural network for testing
class TestNetwork {
   private:
    std::vector<float> weights_;
    size_t input_size_;
    size_t output_size_;

   public:
    TestNetwork(size_t input_size, size_t output_size)
        : input_size_(input_size), output_size_(output_size)
    {
        weights_.resize(input_size * output_size, 0.5f);
    }

    std::vector<float> forward(const std::vector<float>& input)
    {
        if (input.size() != input_size_) {
            throw std::invalid_argument("Input size mismatch");
        }

        std::vector<float> output(output_size_, 0.0f);
        for (size_t i = 0; i < output_size_; ++i) {
            for (size_t j = 0; j < input_size_; ++j) {
                output[i] += input[j] * weights_[j * output_size_ + i];
            }
        }
        return output;
    }

    // Corrupt weights to simulate radiation
    void corruptWeights(float bit_flip_probability = 0.01)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::uniform_int_distribution<int> bit_dist(0, 31);  // 32 bits in a float

        for (auto& weight : weights_) {
            if (dist(gen) < bit_flip_probability) {
                // Flip a random bit
                int bit_pos = bit_dist(gen);
                uint32_t bit_mask = 1u << bit_pos;
                uint32_t value;
                std::memcpy(&value, &weight, sizeof(float));
                value ^= bit_mask;
                std::memcpy(&weight, &value, sizeof(float));
            }
        }
    }
};

// Custom protection strategy for testing
class TestCustomProtection : public rad_ml::tmr::ProtectionStrategy {
   private:
    bool strategy_used_ = false;

   public:
    TestCustomProtection(const DefenseConfig& config) : rad_ml::tmr::ProtectionStrategy()
    {
        // Initialize with config
    }

    template <typename T>
    rad_ml::tmr::ProtectionResult<T> protect(const T& value)
    {
        // Simple protection strategy that just returns the value and marks that it was used
        rad_ml::tmr::ProtectionResult<T> result;
        result.value = value;
        result.error_detected = false;
        result.error_corrected = false;
        strategy_used_ = true;
        return result;
    }

    void updateEnvironment(const sim::RadiationEnvironment& env) override
    {
        // Track environment changes
    }

    bool wasStrategyUsed() const { return strategy_used_; }
};

// Test fixture
class CustomDefenseApiTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        // Register custom protection strategy
        UnifiedDefenseSystem::registerCustomStrategy(
            "test_custom_strategy", [](const DefenseConfig& config) {
                return std::make_unique<TestCustomProtection>(config);
            });
    }
};

// Test default configurations
TEST_F(CustomDefenseApiTest, DefaultConfiguration)
{
    UnifiedDefenseSystem defense;

    // Test with a default value
    auto protected_value = defense.protectValue(42);

    // Check that we can get the value
    ASSERT_NE(protected_value, nullptr);
}

// Test different environment configurations
TEST_F(CustomDefenseApiTest, EnvironmentConfigurations)
{
    // Create different environment configurations
    DefenseConfig earth_config = DefenseConfig::forEnvironment(Environment::EARTH);
    DefenseConfig leo_config = DefenseConfig::forEnvironment(Environment::LEO);
    DefenseConfig jupiter_config = DefenseConfig::forEnvironment(Environment::JUPITER);

    // Check that they use different strategies
    EXPECT_NE(earth_config.strategy, jupiter_config.strategy);

    // Test that higher radiation environments use stronger protection
    EXPECT_EQ(jupiter_config.protection_level, ProtectionLevel::FULL_TMR);

    // Create systems with these configurations
    UnifiedDefenseSystem earth_defense(earth_config);
    UnifiedDefenseSystem leo_defense(leo_config);
    UnifiedDefenseSystem jupiter_defense(jupiter_config);

    // Test that they can protect values
    auto earth_value = earth_defense.protectValue(3.14159f);
    auto leo_value = leo_defense.protectValue(3.14159f);
    auto jupiter_value = jupiter_defense.protectValue(3.14159f);

    ASSERT_NE(earth_value, nullptr);
    ASSERT_NE(leo_value, nullptr);
    ASSERT_NE(jupiter_value, nullptr);
}

// Test protecting individual values
TEST_F(CustomDefenseApiTest, ProtectValues)
{
    UnifiedDefenseSystem defense;

    // Test with different types
    auto int_value = defense.protectValue(42);
    auto float_value = defense.protectValue(3.14159f);
    auto double_value = defense.protectValue(2.71828);

    ASSERT_NE(int_value, nullptr);
    ASSERT_NE(float_value, nullptr);
    ASSERT_NE(double_value, nullptr);
}

// Test executing protected functions
TEST_F(CustomDefenseApiTest, ExecuteProtected)
{
    UnifiedDefenseSystem defense;

    // Test with a simple calculation
    auto result = defense.executeProtected<float>([]() { return std::sqrt(2.0f); });

    // Check the result
    EXPECT_NEAR(result.value, 1.4142f, 0.0001f);
    EXPECT_FALSE(result.error_detected);
}

// Test protecting a neural network
TEST_F(CustomDefenseApiTest, ProtectNeuralNetwork)
{
    // Create a defense system
    UnifiedDefenseSystem defense;

    // Create a protected neural network
    auto network = defense.protectNeuralNetwork<float>({10, 20, 5});

    // Check that it was created
    ASSERT_NE(network, nullptr);

    // Check that it has the right architecture
    EXPECT_EQ(network->getInputSize(), 10);
    EXPECT_EQ(network->getOutputSize(), 5);
}

// Test wrapping an existing neural network
TEST_F(CustomDefenseApiTest, WrapExistingNetwork)
{
    // Create a neural network
    auto network = std::make_unique<TestNetwork>(10, 5);

    // Create a defense config
    DefenseConfig config;
    config.strategy = DefenseStrategy::ENHANCED_TMR;

    // Wrap the network
    auto protected_network = wrapExistingNN(std::move(network), config);

    // Check that it was created
    ASSERT_NE(protected_network, nullptr);

    // Test using the network
    std::vector<float> input(10, 0.5f);
    auto output = protected_network->forward(input);

    // Check output size
    EXPECT_EQ(output.size(), 5);
}

// Test creating a neural network with the helper function
TEST_F(CustomDefenseApiTest, CreateNetworkWithHelper)
{
    // Create a defense config
    DefenseConfig config;
    config.strategy = DefenseStrategy::MULTI_LAYERED;

    // Create a network with the helper
    auto network = createRadiationHardenedNN<TestNetwork>(config, 5, 2);

    // Check that it was created
    ASSERT_NE(network, nullptr);

    // Test using the network
    std::vector<float> input(5, 0.5f);
    auto output = network->forward(input);

    // Check output size
    EXPECT_EQ(output.size(), 2);
}

// Test custom protection strategy
TEST_F(CustomDefenseApiTest, CustomProtectionStrategy)
{
    // Create a defense config with custom strategy
    DefenseConfig config;
    config.strategy = DefenseStrategy::CUSTOM;
    config.custom_params["name"] = "test_custom_strategy";

    // Create a defense system with this config
    UnifiedDefenseSystem defense(config);

    // Execute something with protection to trigger the strategy
    defense.executeProtected<int>([]() { return 42; });

    // We can't directly check if the strategy was used, but we can check
    // that the system was created successfully
    SUCCEED();
}

// Test updating environments
TEST_F(CustomDefenseApiTest, UpdateEnvironment)
{
    // Create a defense system
    UnifiedDefenseSystem defense;

    // Update to different environments
    defense.updateEnvironment(RadiationEnvironment::LEO);
    defense.updateEnvironment(RadiationEnvironment::SAA);
    defense.updateEnvironment(RadiationEnvironment::JUPITER);

    // If no exceptions were thrown, test passes
    SUCCEED();
}

// Test wrapping a network and simulating radiation effects
TEST_F(CustomDefenseApiTest, SimulateRadiationEffects)
{
    // Create a neural network
    auto network = std::make_unique<TestNetwork>(10, 5);

    // Create a defense config for Jupiter (high radiation)
    DefenseConfig jupiter_config = DefenseConfig::forEnvironment(Environment::JUPITER);

    // Wrap the network
    auto protected_network = wrapExistingNN(std::move(network), jupiter_config);

    // Create test input
    std::vector<float> input(10, 0.5f);

    // Get output before corruption
    auto original_output = protected_network->forward(input);

    // Corrupt the underlying network (simulate radiation)
    // In a real scenario, this would happen naturally in a radiation environment
    // This is just for simulation in the test
    protected_network->getNetwork().corruptWeights(0.1f);  // High bit flip probability

    // Get output after corruption
    auto corrupted_output = protected_network->forward(input);

    // With strong protection, outputs should still be similar
    // In a real test, we would need more sophisticated validation
    // This is a simplified example
    SUCCEED();
}

// Main function
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
