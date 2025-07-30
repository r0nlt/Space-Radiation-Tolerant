/**
 * @file pytorch_integration.hpp
 * @brief PyTorch integration for the rad_ml framework
 *
 * This header provides integration between PyTorch and the rad_ml framework,
 * allowing PyTorch models to benefit from radiation hardening and TMR protection.
 *
 * @author Rishab Nuguru
 * @copyright © 2025 Rishab Nuguru
 * @license AGPL v3 license
 */

#pragma once

#include <memory>
#include <rad_ml/core/logger.hpp>
#include <rad_ml/neural/adaptive_protection.hpp>
#include <rad_ml/tmr/adaptive_protection.hpp>

namespace rad_ml {
namespace pytorch {

/**
 * @brief PyTorch integration configuration
 */
struct PyTorchConfig {
    bool enable_tmr_protection = true;
    bool enable_radiation_hardening = true;
    neural::ProtectionLevel protection_level = neural::ProtectionLevel::MODERATE;
    tmr::ProtectionLevel tmr_strategy = tmr::ProtectionLevel::HYBRID_REDUNDANCY;

    // PyTorch-specific settings
    bool use_cuda_if_available = true;
    bool enable_gradient_protection = true;
    bool enable_weight_protection = true;
};

#ifdef RAD_ML_PYTORCH_ENABLED

// Forward declarations for PyTorch types
namespace torch {
class Tensor;
namespace nn {
class Module;
}
namespace optim {
class Optimizer;
}
}  // namespace torch

/**
 * @brief PyTorch tensor wrapper with radiation protection
 */
class ProtectedTensor {
   public:
    ProtectedTensor();
    explicit ProtectedTensor(const torch::Tensor& tensor);
    ~ProtectedTensor();

    // Copy and move operations
    ProtectedTensor(const ProtectedTensor& other);
    ProtectedTensor(ProtectedTensor&& other) noexcept;
    ProtectedTensor& operator=(const ProtectedTensor& other);
    ProtectedTensor& operator=(ProtectedTensor&& other) noexcept;

    // Access to underlying tensor
    torch::Tensor& tensor();
    const torch::Tensor& tensor() const;

    // Protection methods
    void enable_protection(neural::ProtectionLevel level);
    void disable_protection();
    bool is_protected() const;

    // Radiation hardening
    void apply_radiation_hardening();
    void validate_integrity();

    // TMR protection
    void enable_tmr_protection(tmr::ProtectionLevel strategy);
    void disable_tmr_protection();

   private:
    class Impl;
    std::unique_ptr<Impl> pImpl;

    void initialize_protection();
    void update_tmr_copies();
    torch::Tensor vote_tmr_copies();
};

/**
 * @brief Radiation-hardened PyTorch module base class
 */
class RadiationHardenedModule {
   public:
    RadiationHardenedModule();
    explicit RadiationHardenedModule(const PyTorchConfig& config);
    ~RadiationHardenedModule();

    // Protection methods
    void enable_protection(neural::ProtectionLevel level);
    void disable_protection();
    bool is_protected() const;

    // Radiation hardening
    void apply_radiation_hardening();
    void validate_model_integrity();

    // TMR protection
    void enable_tmr_protection(tmr::ProtectionLevel strategy);
    void disable_tmr_protection();

    // Training protection
    void protect_training_step();
    void validate_gradients();

   protected:
    PyTorchConfig config_;
    bool protection_enabled_;

    virtual torch::Tensor forward_protected(torch::Tensor input) = 0;
    virtual void apply_weight_protection() = 0;
    virtual void validate_weights() = 0;

   private:
    void initialize_protection();
    void protect_parameters();
    void validate_parameters();
};

#endif  // RAD_ML_PYTORCH_ENABLED

/**
 * @brief PyTorch integration manager
 */
class PyTorchIntegration {
   public:
    static PyTorchIntegration& get_instance();

    // Initialization
    void initialize(const PyTorchConfig& config = PyTorchConfig{});
    void shutdown();

    // Configuration
    const PyTorchConfig& get_config() const;
    void update_config(const PyTorchConfig& config);

#ifdef RAD_ML_PYTORCH_ENABLED
    // Model protection
    void protect_model(torch::nn::Module& model);
    void harden_model(torch::nn::Module& model);

    // Tensor protection
    ProtectedTensor create_protected_tensor(const torch::Tensor& tensor);
    void protect_tensor(torch::Tensor& tensor);

    // Training protection
    void protect_training_step(torch::nn::Module& model, torch::optim::Optimizer& optimizer);
    void validate_training_state(torch::nn::Module& model);
#endif

    // Utility methods
    bool is_pytorch_available() const;
    bool is_cuda_available() const;

   private:
    PyTorchIntegration() = default;
    ~PyTorchIntegration() = default;
    PyTorchIntegration(const PyTorchIntegration&) = delete;
    PyTorchIntegration& operator=(const PyTorchIntegration&) = delete;

    PyTorchConfig config_;
    bool initialized_;
    bool pytorch_available_;

    void check_pytorch_availability();
    void setup_protection_systems();
};

#ifdef RAD_ML_PYTORCH_ENABLED

// Utility functions
torch::Tensor apply_radiation_hardening(const torch::Tensor& tensor);
torch::Tensor apply_tmr_protection(const torch::Tensor& tensor, tmr::ProtectionLevel strategy);
bool validate_tensor_integrity(const torch::Tensor& tensor);

#endif  // RAD_ML_PYTORCH_ENABLED

}  // namespace pytorch
}  // namespace rad_ml
