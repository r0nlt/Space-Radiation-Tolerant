#pragma once

#include <functional>
#include <memory>
#include <rad_ml/api/rad_ml.hpp>
#include <rad_ml/core/redundancy/enhanced_tmr.hpp>
#include <rad_ml/core/redundancy/space_enhanced_tmr.hpp>
#include <rad_ml/core/redundancy/tmr.hpp>
#include <rad_ml/hw/hardware_acceleration.hpp>
#include <rad_ml/neural/adaptive_protection.hpp>
#include <rad_ml/neural/advanced_reed_solomon.hpp>
#include <rad_ml/neural/protected_neural_network.hpp>
#include <rad_ml/sim/mission_environment.hpp>
#include <rad_ml/tmr/physics_driven_protection.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad_ml {
namespace api {

/**
 * @brief DefenseStrategy enum representing different protection strategies
 */
enum class DefenseStrategy {
    STANDARD_TMR,          // Basic Triple Modular Redundancy
    ENHANCED_TMR,          // Enhanced TMR with checksums
    SPACE_OPTIMIZED_TMR,   // Space-optimized TMR
    REED_SOLOMON,          // Reed-Solomon error correction
    ADAPTIVE_PROTECTION,   // Adaptive protection based on environment
    PHYSICS_DRIVEN,        // Physics-based protection models
    HARDWARE_ACCELERATED,  // Hardware-accelerated protection
    MULTI_LAYERED,         // Combined multiple strategies
    CUSTOM                 // User-defined custom strategy
};

/**
 * @brief Configuration for a radiation defense system
 */
struct DefenseConfig {
    // Basic configuration
    DefenseStrategy strategy = DefenseStrategy::ENHANCED_TMR;
    sim::RadiationEnvironment environment = sim::RadiationEnvironment::LEO;
    neural::ProtectionLevel protection_level = neural::ProtectionLevel::ADAPTIVE_TMR;

    // Neural network specific
    bool protect_weights = true;
    bool protect_activations = true;
    bool protect_gradients = false;  // Only needed during training

    // Hardware acceleration
    bool use_hardware_acceleration = false;
    hw::AcceleratorType accelerator_type = hw::AcceleratorType::NONE;

    // Custom strategy parameters
    std::unordered_map<std::string, std::string> custom_params;

    // Monitoring and reporting
    bool enable_error_monitoring = true;
    bool collect_statistics = true;

    // Create configuration for a specific radiation environment
    static DefenseConfig forEnvironment(sim::Environment env)
    {
        DefenseConfig config;
        config.environment = sim::createEnvironment(env);

        // Select appropriate strategy based on environment
        switch (env) {
            case sim::Environment::JUPITER:
            case sim::Environment::SOLAR_STORM:
                // Harsh environments need stronger protection
                config.strategy = DefenseStrategy::MULTI_LAYERED;
                config.protection_level = neural::ProtectionLevel::FULL_TMR;
                break;

            case sim::Environment::SAA:
            case sim::Environment::MARS:
                // Moderate radiation environments
                config.strategy = DefenseStrategy::SPACE_OPTIMIZED_TMR;
                config.protection_level = neural::ProtectionLevel::SELECTIVE_TMR;
                break;

            default:
                // Default configuration for milder environments
                config.strategy = DefenseStrategy::ENHANCED_TMR;
                config.protection_level = neural::ProtectionLevel::ADAPTIVE_TMR;
                break;
        }

        return config;
    }
};

/**
 * @brief Factory class for creating custom defense strategies
 */
class DefenseFactory {
   public:
    using CreatorFunction =
        std::function<std::unique_ptr<tmr::ProtectionStrategy>(const DefenseConfig&)>;

    static DefenseFactory& instance()
    {
        static DefenseFactory factory;
        return factory;
    }

    bool registerStrategy(const std::string& name, CreatorFunction creator)
    {
        return strategies_.emplace(name, creator).second;
    }

    std::unique_ptr<tmr::ProtectionStrategy> createStrategy(const std::string& name,
                                                            const DefenseConfig& config)
    {
        auto it = strategies_.find(name);
        if (it != strategies_.end()) {
            return it->second(config);
        }
        return nullptr;
    }

   private:
    std::unordered_map<std::string, CreatorFunction> strategies_;
};

/**
 * @brief Main class for managing radiation defenses for neural networks
 */
class UnifiedDefenseSystem {
   public:
    /**
     * @brief Constructor with configuration
     */
    explicit UnifiedDefenseSystem(const DefenseConfig& config = DefenseConfig())
        : config_(config), hardware_accelerator_(nullptr)
    {
        // Initialize protection strategy
        initializeProtectionStrategy();

        // Initialize hardware acceleration if enabled
        if (config.use_hardware_acceleration) {
            hw::AcceleratorConfig hw_config;
            hw_config.type = config.accelerator_type;
            hardware_accelerator_ = std::make_unique<hw::HardwareAccelerator>(hw_config);
            if (hardware_accelerator_->is_available()) {
                hardware_accelerator_->initialize();
            }
            else {
                hardware_accelerator_.reset();  // Not available, don't use
            }
        }
    }

    /**
     * @brief Protect a neural network with the configured defense strategy
     *
     * @tparam T Numeric type (float, double)
     * @param network Neural network to protect
     * @return Protected neural network
     */
    template <typename T>
    std::unique_ptr<neural::ProtectedNeuralNetwork<T>> protectNeuralNetwork(
        const std::vector<size_t>& architecture)
    {
        auto network = std::make_unique<neural::ProtectedNeuralNetwork<T>>(
            architecture, config_.protection_level);

        // Apply additional protection based on configuration
        if (config_.strategy == DefenseStrategy::ADAPTIVE_PROTECTION) {
            network->enableAdaptiveProtection(config_.environment);
        }

        // Configure protection settings
        network->setProtectWeights(config_.protect_weights);
        network->setProtectActivations(config_.protect_activations);
        network->setProtectGradients(config_.protect_gradients);

        return network;
    }

    /**
     * @brief Protect a value with appropriate defense strategy
     *
     * @tparam T Value type
     * @param value Value to protect
     * @return Protected value
     */
    template <typename T>
    auto protectValue(const T& value)
    {
        switch (config_.strategy) {
            case DefenseStrategy::STANDARD_TMR:
                return make_tmr::standard<T>(value);

            case DefenseStrategy::ENHANCED_TMR:
                return make_tmr::enhanced<T>(value);

            case DefenseStrategy::SPACE_OPTIMIZED_TMR:
                // Use approximate TMR for floating point types
                if constexpr (std::is_floating_point_v<T>) {
                    return make_tmr::approximate<T>(value);
                }
                else {
                    return make_tmr::enhanced<T>(value);
                }

            case DefenseStrategy::MULTI_LAYERED:
                return make_tmr::hybrid<T>(value);

            default:
                return make_tmr::enhanced<T>(value);
        }
    }

    /**
     * @brief Execute a function with radiation protection
     *
     * @tparam T Return type
     * @param function Function to execute
     * @return Result with error information
     */
    template <typename T>
    auto executeProtected(std::function<T()> function)
    {
        return protection_strategy_->executeProtected<T>(function);
    }

    /**
     * @brief Update the radiation environment
     *
     * @param env New environment
     */
    void updateEnvironment(sim::RadiationEnvironment env)
    {
        config_.environment = env;
        if (protection_strategy_) {
            protection_strategy_->updateEnvironment(env);
        }
    }

    /**
     * @brief Register a custom protection strategy
     *
     * @param name Strategy name
     * @param creator Creator function
     * @return True if successfully registered
     */
    static bool registerCustomStrategy(const std::string& name,
                                       DefenseFactory::CreatorFunction creator)
    {
        return DefenseFactory::instance().registerStrategy(name, creator);
    }

   private:
    DefenseConfig config_;
    std::unique_ptr<tmr::ProtectionStrategy> protection_strategy_;
    std::unique_ptr<hw::HardwareAccelerator> hardware_accelerator_;

    void initializeProtectionStrategy()
    {
        // Create core::MaterialProperties from environment
        core::MaterialProperties material;
        material.radiation_tolerance = 50.0;  // Default aluminum equivalent

        // Create appropriate protection strategy
        switch (config_.strategy) {
            case DefenseStrategy::PHYSICS_DRIVEN:
                protection_strategy_ =
                    std::make_unique<tmr::PhysicsDrivenProtection>(material, config_.environment);
                break;

            case DefenseStrategy::CUSTOM:
                // Use the factory to create custom strategy
                if (!config_.custom_params.empty() &&
                    config_.custom_params.find("name") != config_.custom_params.end()) {
                    std::string name = config_.custom_params.at("name");
                    protection_strategy_ = DefenseFactory::instance().createStrategy(name, config_);
                }

                // Fallback if custom creation failed
                if (!protection_strategy_) {
                    protection_strategy_ = std::make_unique<tmr::PhysicsDrivenProtection>(
                        material, config_.environment);
                }
                break;

            default:
                // Default to physics-driven protection
                protection_strategy_ =
                    std::make_unique<tmr::PhysicsDrivenProtection>(material, config_.environment);
                break;
        }
    }
};

/**
 * @brief Wrapper for neural networks that provides integrated radiation protection
 *
 * @tparam NeuralNetwork The underlying neural network type to protect
 */
template <typename NeuralNetwork>
class RadiationHardenedNN {
   public:
    /**
     * @brief Create a radiation-hardened neural network
     *
     * @param network Underlying neural network
     * @param config Defense configuration
     */
    RadiationHardenedNN(std::unique_ptr<NeuralNetwork> network,
                        const DefenseConfig& config = DefenseConfig())
        : underlying_network_(std::move(network)), defense_system_(config)
    {
    }

    /**
     * @brief Create a radiation-hardened neural network
     *
     * @tparam Args Constructor argument types
     * @param config Defense configuration
     * @param args Constructor arguments for the neural network
     */
    template <typename... Args>
    RadiationHardenedNN(const DefenseConfig& config, Args&&... args)
        : underlying_network_(std::make_unique<NeuralNetwork>(std::forward<Args>(args)...)),
          defense_system_(config)
    {
    }

    /**
     * @brief Forward pass through the network with radiation protection
     *
     * @tparam InputType Input data type
     * @param input Input data
     * @return Protected output
     */
    template <typename InputType>
    auto forward(const InputType& input)
    {
        // Execute the forward pass with radiation protection
        return defense_system_
            .executeProtected<decltype(underlying_network_->forward(input))>(
                [this, &input]() { return underlying_network_->forward(input); })
            .value;
    }

    /**
     * @brief Get the underlying neural network
     *
     * @return Reference to underlying network
     */
    NeuralNetwork& getNetwork() { return *underlying_network_; }

    /**
     * @brief Get the defense system
     *
     * @return Reference to defense system
     */
    UnifiedDefenseSystem& getDefenseSystem() { return defense_system_; }

    /**
     * @brief Update the radiation environment
     *
     * @param env New environment
     */
    void updateEnvironment(sim::RadiationEnvironment env)
    {
        defense_system_.updateEnvironment(env);
    }

   private:
    std::unique_ptr<NeuralNetwork> underlying_network_;
    UnifiedDefenseSystem defense_system_;
};

// Helper functions to create radiation-hardened neural networks
template <typename NeuralNetwork, typename... Args>
auto createRadiationHardenedNN(const DefenseConfig& config, Args&&... args)
{
    return std::make_unique<RadiationHardenedNN<NeuralNetwork>>(config,
                                                                std::forward<Args>(args)...);
}

template <typename NeuralNetwork>
auto wrapExistingNN(std::unique_ptr<NeuralNetwork> network, const DefenseConfig& config)
{
    return std::make_unique<RadiationHardenedNN<NeuralNetwork>>(std::move(network), config);
}

}  // namespace api
}  // namespace rad_ml
