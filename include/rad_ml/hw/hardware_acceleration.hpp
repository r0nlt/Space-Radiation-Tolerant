/**
 * Copyright (C) 2025 Rishab Nuguru
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <optional>
#include <array>

namespace rad_ml {
namespace hw {

/**
 * @brief Types of hardware accelerators that can be used with the framework
 * 
 * Based on radiation-tolerant computing platforms used in space missions
 */
enum class AcceleratorType {
    NONE,                 // No hardware acceleration
    RAD_HARD_CPU,         // Radiation-hardened CPU (e.g., RAD750, LEON4)
    RAD_HARD_GPU,         // Radiation-hardened GPU (e.g., HPSC)
    RAD_TOL_FPGA,         // Radiation-tolerant FPGA (e.g., Xilinx RT-FPGA)
    RAD_TOL_ASIC,         // Custom radiation-tolerant ASIC
    HYBRID_SYSTEM         // Mix of rad-hard and rad-tolerant components
};

/**
 * @brief Scrubbing strategy for FPGA-based accelerators
 * 
 * Based on techniques used in NASA and ESA FPGA deployments
 */
enum class ScrubbingStrategy {
    NONE,                 // No scrubbing (not recommended for space)
    PERIODIC,             // Regular interval scrubbing
    CONTINUOUS,           // Continuous background scrubbing
    TRIGGERED,            // Scrubbing triggered by error detection
    ADAPTIVE              // Adaptive rate based on environment
};

/**
 * @brief Hardware TMR implementation approach
 */
enum class HardwareTMRApproach {
    NONE,                 // No hardware TMR
    BLOCK_LEVEL,          // TMR applied to major blocks
    REGISTER_LEVEL,       // TMR applied to registers
    COMPLETE              // Full TMR of design
};

/**
 * @brief Configuration for hardware accelerator
 */
struct AcceleratorConfig {
    // Accelerator settings
    AcceleratorType type = AcceleratorType::NONE;
    bool enable_hw_ecc = true;            // Hardware ECC support
    HardwareTMRApproach tmr_approach = HardwareTMRApproach::NONE;
    
    // FPGA-specific settings
    ScrubbingStrategy scrubbing_strategy = ScrubbingStrategy::PERIODIC;
    double scrubbing_interval_sec = 10.0; // Typical scrubbing interval
    
    // Power settings
    double power_budget_watts = 5.0;      // Power budget for accelerator
    double performance_factor = 1.0;      // Performance scaling factor (0-1)
    
    // Statistics
    double bit_upset_tolerance = 0.0;     // Bits that can be upset without failure
    double seu_threshold = 0.0;           // SEUs per hour threshold
    
    // Get string representation of accelerator type
    std::string get_type_string() const {
        switch (type) {
            case AcceleratorType::NONE: return "None";
            case AcceleratorType::RAD_HARD_CPU: return "Rad-Hard CPU";
            case AcceleratorType::RAD_HARD_GPU: return "Rad-Hard GPU";
            case AcceleratorType::RAD_TOL_FPGA: return "Rad-Tolerant FPGA";
            case AcceleratorType::RAD_TOL_ASIC: return "Rad-Tolerant ASIC";
            case AcceleratorType::HYBRID_SYSTEM: return "Hybrid System";
            default: return "Unknown";
        }
    }
    
    // Get string representation of TMR approach
    std::string get_tmr_string() const {
        switch (tmr_approach) {
            case HardwareTMRApproach::NONE: return "None";
            case HardwareTMRApproach::BLOCK_LEVEL: return "Block-Level";
            case HardwareTMRApproach::REGISTER_LEVEL: return "Register-Level";
            case HardwareTMRApproach::COMPLETE: return "Complete";
            default: return "Unknown";
        }
    }
    
    // Get string representation of scrubbing strategy
    std::string get_scrubbing_string() const {
        switch (scrubbing_strategy) {
            case ScrubbingStrategy::NONE: return "None";
            case ScrubbingStrategy::PERIODIC: return "Periodic";
            case ScrubbingStrategy::CONTINUOUS: return "Continuous";
            case ScrubbingStrategy::TRIGGERED: return "Triggered";
            case ScrubbingStrategy::ADAPTIVE: return "Adaptive";
            default: return "Unknown";
        }
    }
};

/**
 * @brief Performance characteristics of a specific layer on hardware
 */
struct LayerPerformance {
    std::string layer_name;
    double execution_time_ms;
    double energy_usage_mj;
    double memory_usage_bytes;
    double reliability_score;  // 0.0-1.0 rating of resilience
};

/**
 * @brief Base class for hardware accelerator interface
 */
class HardwareAccelerator {
public:
    /**
     * @brief Create a hardware accelerator with specific configuration
     * 
     * @param config Accelerator configuration
     */
    explicit HardwareAccelerator(const AcceleratorConfig& config = AcceleratorConfig())
        : config_(config), error_counter_(0), last_error_time_(0) {
    }
    
    /**
     * @brief Destructor
     */
    virtual ~HardwareAccelerator() = default;
    
    /**
     * @brief Check if hardware acceleration is available
     * 
     * @return true if hardware accelerator is available
     */
    virtual bool is_available() const {
        // In real implementation, check hardware presence
        return config_.type != AcceleratorType::NONE;
    }
    
    /**
     * @brief Initialize the hardware accelerator
     * 
     * @return true if initialization successful
     */
    virtual bool initialize() {
        // Base implementation does nothing
        return is_available();
    }
    
    /**
     * @brief Execute a model inference on the hardware
     * 
     * @param input_data Input tensor data
     * @param output_buffer Buffer to store output
     * @return true if execution successful
     */
    virtual bool execute(const std::vector<float>& input_data, 
                        std::vector<float>& output_buffer) {
        // Base implementation does nothing
        return false;
    }
    
    /**
     * @brief Get performance metrics for the last execution
     * 
     * @return Map of metric name to value
     */
    virtual std::map<std::string, double> get_performance_metrics() const {
        std::map<std::string, double> metrics;
        metrics["execution_time_ms"] = 0.0;
        metrics["energy_usage_mj"] = 0.0;
        metrics["reliability_score"] = 0.0;
        return metrics;
    }
    
    /**
     * @brief Get layer-by-layer performance breakdown
     * 
     * @return Vector of layer performance data
     */
    virtual std::vector<LayerPerformance> get_layer_performance() const {
        return {};
    }
    
    /**
     * @brief Trigger hardware scrubbing (FPGA only)
     * 
     * @return true if scrubbing was performed
     */
    virtual bool trigger_scrubbing() {
        return false;
    }
    
    /**
     * @brief Get current hardware configuration
     * 
     * @return Current configuration
     */
    const AcceleratorConfig& get_config() const {
        return config_;
    }
    
    /**
     * @brief Update hardware configuration
     * 
     * @param config New configuration
     * @return true if update successful
     */
    virtual bool update_config(const AcceleratorConfig& config) {
        config_ = config;
        return true;
    }
    
    /**
     * @brief Get error statistics
     * 
     * @return Map of error type to count
     */
    virtual std::map<std::string, uint64_t> get_error_stats() const {
        std::map<std::string, uint64_t> stats;
        stats["total_errors"] = error_counter_;
        stats["uncorrectable_errors"] = 0;
        stats["corrected_errors"] = 0;
        return stats;
    }
    
    /**
     * @brief Get diagnostic information
     * 
     * @return String with diagnostic details
     */
    virtual std::string get_diagnostics() const {
        std::string result = "Hardware Accelerator Diagnostics\n";
        result += "--------------------------------\n";
        
        result += "Type: " + config_.get_type_string() + "\n";
        result += "Hardware ECC: " + std::string(config_.enable_hw_ecc ? "Enabled" : "Disabled") + "\n";
        result += "TMR Approach: " + config_.get_tmr_string() + "\n";
        result += "Scrubbing Strategy: " + config_.get_scrubbing_string() + "\n";
        
        if (config_.scrubbing_strategy != ScrubbingStrategy::NONE) {
            result += "Scrubbing Interval: " + std::to_string(config_.scrubbing_interval_sec) + " sec\n";
        }
        
        result += "Power Budget: " + std::to_string(config_.power_budget_watts) + " W\n";
        result += "Performance Factor: " + std::to_string(config_.performance_factor) + "\n";
        result += "Total Errors: " + std::to_string(error_counter_) + "\n";
        
        return result;
    }
    
    /**
     * @brief Calculate estimated reliability in current environment
     * 
     * @param seu_rate SEUs per bit per day
     * @return Reliability score (0.0-1.0)
     */
    virtual double calculate_reliability(double seu_rate) const {
        // Base reliability model
        // Different accelerator types have different inherent reliability
        double base_reliability = 0.0;
        
        switch (config_.type) {
            case AcceleratorType::RAD_HARD_CPU:
                base_reliability = 0.95;
                break;
            case AcceleratorType::RAD_HARD_GPU:
                base_reliability = 0.92;
                break;
            case AcceleratorType::RAD_TOL_FPGA:
                base_reliability = 0.90;
                break;
            case AcceleratorType::RAD_TOL_ASIC:
                base_reliability = 0.98;
                break;
            case AcceleratorType::HYBRID_SYSTEM:
                base_reliability = 0.93;
                break;
            default:
                base_reliability = 0.85;
                break;
        }
        
        // Hardware ECC improves reliability
        if (config_.enable_hw_ecc) {
            base_reliability += 0.03;
        }
        
        // TMR approach improves reliability
        switch (config_.tmr_approach) {
            case HardwareTMRApproach::COMPLETE:
                base_reliability += 0.05;
                break;
            case HardwareTMRApproach::REGISTER_LEVEL:
                base_reliability += 0.04;
                break;
            case HardwareTMRApproach::BLOCK_LEVEL:
                base_reliability += 0.02;
                break;
            default:
                break;
        }
        
        // FPGA scrubbing improves reliability
        if (config_.type == AcceleratorType::RAD_TOL_FPGA) {
            switch (config_.scrubbing_strategy) {
                case ScrubbingStrategy::CONTINUOUS:
                    base_reliability += 0.03;
                    break;
                case ScrubbingStrategy::PERIODIC:
                    base_reliability += 0.02;
                    break;
                case ScrubbingStrategy::TRIGGERED:
                    base_reliability += 0.01;
                    break;
                case ScrubbingStrategy::ADAPTIVE:
                    base_reliability += 0.04;
                    break;
                default:
                    break;
            }
        }
        
        // Environmental impact (SEU rate)
        double env_factor = std::exp(-10.0 * seu_rate); // Exponential degradation with SEU rate
        
        // Calculate final reliability (capped at 0.999)
        double reliability = std::min(0.999, base_reliability * env_factor);
        
        return reliability;
    }
    
protected:
    // Accelerator configuration
    AcceleratorConfig config_;
    
    // Error statistics
    uint64_t error_counter_;
    uint64_t last_error_time_;
    
    /**
     * @brief Log an error
     * 
     * @param error_type Type of error
     * @param details Additional details
     */
    virtual void log_error(const std::string& error_type, const std::string& details) {
        error_counter_++;
        last_error_time_ = std::chrono::system_clock::now().time_since_epoch().count();
        // In a real implementation, this would log to persistent storage
    }
};

/**
 * @brief Factory to create hardware accelerators based on available hardware
 */
class HardwareAcceleratorFactory {
public:
    /**
     * @brief Get the singleton instance
     * 
     * @return Reference to the singleton
     */
    static HardwareAcceleratorFactory& instance() {
        static HardwareAcceleratorFactory instance;
        return instance;
    }
    
    /**
     * @brief Detect available accelerators on the system
     * 
     * @return Map of accelerator type to availability
     */
    std::map<AcceleratorType, bool> detect_available_accelerators() {
        std::map<AcceleratorType, bool> result;
        
        // In simulation mode, preset available accelerators
        // In real implementation, would detect actual hardware
        result[AcceleratorType::NONE] = true;
        result[AcceleratorType::RAD_HARD_CPU] = false;
        result[AcceleratorType::RAD_HARD_GPU] = false;
        result[AcceleratorType::RAD_TOL_FPGA] = false;
        result[AcceleratorType::RAD_TOL_ASIC] = false;
        result[AcceleratorType::HYBRID_SYSTEM] = false;
        
        return result;
    }
    
    /**
     * @brief Create an accelerator of specified type
     * 
     * @param type Accelerator type
     * @param config Configuration
     * @return Unique pointer to the accelerator
     */
    std::unique_ptr<HardwareAccelerator> create_accelerator(
        AcceleratorType type,
        const AcceleratorConfig& config = AcceleratorConfig()) {
        
        // Create configuration with correct type
        AcceleratorConfig actual_config = config;
        actual_config.type = type;
        
        // Return base class for now
        // In real implementation, would return specific accelerator types
        return std::make_unique<HardwareAccelerator>(actual_config);
    }
    
    /**
     * @brief Create best available accelerator
     * 
     * @param config Base configuration to use
     * @return Unique pointer to the best available accelerator
     */
    std::unique_ptr<HardwareAccelerator> create_best_available(
        const AcceleratorConfig& config = AcceleratorConfig()) {
        
        auto available = detect_available_accelerators();
        
        // Try each accelerator type in priority order
        if (available[AcceleratorType::RAD_TOL_ASIC]) {
            return create_accelerator(AcceleratorType::RAD_TOL_ASIC, config);
        }
        
        if (available[AcceleratorType::RAD_HARD_GPU]) {
            return create_accelerator(AcceleratorType::RAD_HARD_GPU, config);
        }
        
        if (available[AcceleratorType::RAD_TOL_FPGA]) {
            return create_accelerator(AcceleratorType::RAD_TOL_FPGA, config);
        }
        
        if (available[AcceleratorType::RAD_HARD_CPU]) {
            return create_accelerator(AcceleratorType::RAD_HARD_CPU, config);
        }
        
        if (available[AcceleratorType::HYBRID_SYSTEM]) {
            return create_accelerator(AcceleratorType::HYBRID_SYSTEM, config);
        }
        
        // Fallback to no acceleration
        return create_accelerator(AcceleratorType::NONE, config);
    }
    
private:
    // Private constructor for singleton
    HardwareAcceleratorFactory() = default;
};

/**
 * @brief Integration of hardware accelerators with software TMR
 * 
 * This class combines hardware acceleration with the software TMR
 * framework to provide the best of both approaches.
 */
class TMRAcceleratorIntegration {
public:
    /**
     * @brief Create a new TMR accelerator integration
     * 
     * @param config Accelerator configuration
     */
    explicit TMRAcceleratorIntegration(const AcceleratorConfig& config = AcceleratorConfig())
        : accelerator_(HardwareAcceleratorFactory::instance().create_best_available(config)),
          use_sw_tmr_(true) {
    }
    
    /**
     * @brief Initialize the accelerator and TMR integration
     * 
     * @return true if initialization successful
     */
    bool initialize() {
        // Initialize hardware accelerator
        bool hw_init = accelerator_->initialize();
        
        // Determine if software TMR should be used based on hardware capabilities
        auto& config = accelerator_->get_config();
        
        // If hardware has complete TMR, we can disable software TMR
        if (config.tmr_approach == HardwareTMRApproach::COMPLETE) {
            use_sw_tmr_ = false;
        }
        
        return hw_init;
    }
    
    /**
     * @brief Execute inference with appropriate TMR strategy
     * 
     * @param input_data Input tensor data
     * @param output_buffer Buffer for output data
     * @return true if execution successful
     */
    bool execute(const std::vector<float>& input_data, 
                std::vector<float>& output_buffer) {
                
        if (!use_sw_tmr_) {
            // Use hardware TMR only
            return accelerator_->execute(input_data, output_buffer);
        }
        
        // Use software TMR with hardware acceleration
        std::array<std::vector<float>, 3> outputs;
        bool success = true;
        
        // Execute three times
        for (int i = 0; i < 3; ++i) {
            outputs[i].resize(output_buffer.size());
            success &= accelerator_->execute(input_data, outputs[i]);
        }
        
        // If any execution failed, return failure
        if (!success) {
            return false;
        }
        
        // Majority voting for each output element
        for (size_t j = 0; j < output_buffer.size(); ++j) {
            if (outputs[0][j] == outputs[1][j] || outputs[0][j] == outputs[2][j]) {
                output_buffer[j] = outputs[0][j];
            } else if (outputs[1][j] == outputs[2][j]) {
                output_buffer[j] = outputs[1][j];
            } else {
                // No majority, use average
                output_buffer[j] = (outputs[0][j] + outputs[1][j] + outputs[2][j]) / 3.0f;
            }
        }
        
        return true;
    }
    
    /**
     * @brief Get underlying hardware accelerator
     * 
     * @return Reference to hardware accelerator
     */
    HardwareAccelerator& get_accelerator() {
        return *accelerator_;
    }
    
    /**
     * @brief Get diagnostic information
     * 
     * @return String with diagnostic details
     */
    std::string get_diagnostics() const {
        std::string result = "TMR Accelerator Integration Diagnostics\n";
        result += "----------------------------------------\n";
        
        result += "Hardware Accelerator: " + 
                 accelerator_->get_config().get_type_string() + "\n";
        result += "Software TMR: " + 
                 std::string(use_sw_tmr_ ? "Enabled" : "Disabled") + "\n";
        result += "\n";
        result += accelerator_->get_diagnostics();
        
        return result;
    }
    
    /**
     * @brief Configure software TMR usage
     * 
     * @param use_sw_tmr Whether to use software TMR
     */
    void set_software_tmr(bool use_sw_tmr) {
        use_sw_tmr_ = use_sw_tmr;
    }
    
    /**
     * @brief Calculate combined system reliability
     * 
     * @param seu_rate SEUs per bit per day
     * @return System reliability score (0.0-1.0)
     */
    double calculate_system_reliability(double seu_rate) const {
        // Hardware reliability
        double hw_reliability = accelerator_->calculate_reliability(seu_rate);
        
        // Software TMR reliability model
        double sw_tmr_reliability = 0.0;
        if (use_sw_tmr_) {
            // Probability that at least 2 out of 3 modules are correct
            double p = hw_reliability; // Probability of single module being correct
            sw_tmr_reliability = 3 * p * p * (1 - p) + p * p * p;
        } else {
            sw_tmr_reliability = hw_reliability;
        }
        
        return sw_tmr_reliability;
    }
    
private:
    // Hardware accelerator
    std::unique_ptr<HardwareAccelerator> accelerator_;
    
    // Whether to use software TMR
    bool use_sw_tmr_;
};

} // namespace hw
} // namespace rad_ml 
