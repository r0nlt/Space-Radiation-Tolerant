#pragma once

/**
 * @file rad_ml.hpp
 * @brief Unified API for the Radiation-Tolerant Machine Learning Framework
 * 
 * This header provides a consolidated API for the major components of the rad_ml
 * framework, simplifying usage while maintaining access to advanced features.
 */

// Core headers
#include "../error/error_handling.hpp"
#include "../core/memory/unified_memory.hpp"
#include "../tmr/enhanced_tmr.hpp"
#include "../core/memory/memory_scrubber.hpp"

// TMR variants
#include "../tmr/tmr.hpp"
#include "../tmr/enhanced_stuck_bit_tmr.hpp"
#include "../tmr/health_weighted_tmr.hpp"
#include "../tmr/approximate_tmr.hpp"
#include "../tmr/temporal_redundancy.hpp"
#include "../tmr/hybrid_redundancy.hpp"

// Neural network components
#include "../neural/selective_hardening.hpp"
#include "../neural/error_predictor.hpp"
#include "../inference/model.hpp"

// Simulation and testing
#include "../sim/physics_radiation_simulator.hpp"
#include "../testing/mission_simulator.hpp"
#include "../testing/fault_injector.hpp"
#include "../testing/benchmark_framework.hpp"

// Memory management
#include "../memory/radiation_mapped_allocator.hpp"

// Advanced features
#include "../advanced/algorithmic_diversity.hpp"
#include "../advanced/error_prediction.hpp"
#include "../core/recovery/checkpoint_manager.hpp"
#include "../core/radiation/adaptive_protection.hpp"
#include "../power/power_aware_protection.hpp"

/**
 * @namespace rad_ml
 * @brief Root namespace for the Radiation-Tolerant Machine Learning Framework
 */
namespace rad_ml {

/**
 * @brief Framework version information
 */
struct Version {
    static constexpr int major = 2;
    static constexpr int minor = 0;
    static constexpr int patch = 0;
    
    /**
     * @brief Get version string
     * @return Version as string in format "major.minor.patch"
     */
    static constexpr const char* asString() {
        return "2.0.0";
    }
};

/**
 * @brief Initialize the framework with default configuration
 * 
 * @param enable_logging Whether to enable logging (default: true)
 * @param memory_protection_level Default memory protection level (default: NONE)
 * @return True if initialization successful
 */
inline bool initialize(
    bool enable_logging = true,
    memory::MemoryProtectionLevel memory_protection_level = memory::MemoryProtectionLevel::NONE) {
    
    try {
        // Set up error handling
        if (enable_logging) {
            error::ErrorHandler::setReportingLevel(error::ErrorSeverity::INFO);
        } else {
            error::ErrorHandler::setReportingLevel(error::ErrorSeverity::ERROR);
        }
        
        // Initialize memory manager
        auto& memory_manager = memory::UnifiedMemoryManager::getInstance();
        memory_manager.setDefaultProtectionLevel(memory_protection_level);
        
        return true;
    } catch (const std::exception& e) {
        if (enable_logging) {
            std::cerr << "Failed to initialize rad_ml framework: " << e.what() << std::endl;
        }
        return false;
    }
}

/**
 * @brief Shutdown the framework and perform cleanup
 * 
 * @param check_for_leaks Whether to check for memory leaks (default: true)
 * @return True if cleanup was successful
 */
inline bool shutdown(bool check_for_leaks = true) {
    try {
        auto& memory_manager = memory::UnifiedMemoryManager::getInstance();
        
        if (check_for_leaks) {
            memory_manager.checkForLeaks(true);
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during rad_ml framework shutdown: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Convenience namespace for TMR template aliases
 */
namespace tmr_types {
    /**
     * @brief Standard TMR with basic majority voting
     */
    template <typename T>
    using StandardTMR = tmr::TMR<T>;
    
    /**
     * @brief Enhanced TMR with CRC checking and health tracking
     */
    template <typename T>
    using EnhancedTMR = tmr::EnhancedTMR<T>;
    
    /**
     * @brief TMR with stuck bit detection and mitigation
     */
    template <typename T>
    using StuckBitTMR = tmr::EnhancedStuckBitTMR<T>;
    
    /**
     * @brief TMR with health-weighted voting for improved resilience
     */
    template <typename T>
    using HealthWeightedTMR = tmr::HealthWeightedTMR<T>;
    
    /**
     * @brief TMR with approximate comparisons for floating-point values
     */
    template <typename T>
    using ApproximateTMR = tmr::ApproximateTMR<T>;
    
    /**
     * @brief Hybrid redundancy combining spatial and temporal approaches
     */
    template <typename T>
    using HybridTMR = tmr::HybridRedundancy<T>;
}

/**
 * @brief Factory functions for creating TMR instances
 */
namespace make_tmr {
    /**
     * @brief Create a standard TMR instance
     * 
     * @tparam T Type to protect
     * @param initial_value Initial value
     * @return Shared pointer to TMR instance
     */
    template <typename T>
    inline std::shared_ptr<tmr::TMR<T>> standard(const T& initial_value = T()) {
        return tmr::TMRFactory::createTMR<T>(initial_value);
    }
    
    /**
     * @brief Create an enhanced TMR instance
     * 
     * @tparam T Type to protect
     * @param initial_value Initial value
     * @return Shared pointer to EnhancedTMR instance
     */
    template <typename T>
    inline std::shared_ptr<tmr::EnhancedTMR<T>> enhanced(const T& initial_value = T()) {
        return tmr::TMRFactory::createEnhancedTMR<T>(initial_value);
    }
    
    /**
     * @brief Create a stuck bit TMR instance
     * 
     * @tparam T Type to protect
     * @param initial_value Initial value
     * @return Shared pointer to EnhancedStuckBitTMR instance
     */
    template <typename T>
    inline std::shared_ptr<tmr::EnhancedStuckBitTMR<T>> stuckBit(const T& initial_value = T()) {
        return std::make_shared<tmr::EnhancedStuckBitTMR<T>>(initial_value);
    }
    
    /**
     * @brief Create a health-weighted TMR instance
     * 
     * @tparam T Type to protect
     * @param initial_value Initial value
     * @return Shared pointer to HealthWeightedTMR instance
     */
    template <typename T>
    inline std::shared_ptr<tmr::HealthWeightedTMR<T>> healthWeighted(const T& initial_value = T()) {
        return std::make_shared<tmr::HealthWeightedTMR<T>>(initial_value);
    }
    
    /**
     * @brief Create an approximate TMR instance for floating-point values
     * 
     * @tparam T Floating-point type to protect
     * @param initial_value Initial value
     * @param tolerance Comparison tolerance
     * @return Shared pointer to ApproximateTMR instance
     */
    template <typename T>
    inline std::shared_ptr<tmr::ApproximateTMR<T>> approximate(
        const T& initial_value = T(),
        T tolerance = static_cast<T>(0.0001)) {
        
        return std::make_shared<tmr::ApproximateTMR<T>>(
            initial_value, 
            tmr::ApproximationType::RELATIVE_DIFFERENCE,
            tolerance);
    }
    
    /**
     * @brief Create a hybrid redundancy instance
     * 
     * @tparam T Type to protect
     * @param initial_value Initial value
     * @return Shared pointer to HybridRedundancy instance
     */
    template <typename T>
    inline std::shared_ptr<tmr::HybridRedundancy<T>> hybrid(const T& initial_value = T()) {
        return std::make_shared<tmr::HybridRedundancy<T>>(initial_value);
    }
}

/**
 * @brief High-level memory management functions
 */
namespace memory_management {
    /**
     * @brief Allocate memory with radiation protection
     * 
     * @tparam T Type to allocate
     * @param count Number of elements
     * @param protection_level Protection level
     * @return Pointer to allocated memory
     */
    template <typename T>
    inline T* allocate(
        size_t count = 1,
        memory::MemoryProtectionLevel protection_level = memory::MemoryProtectionLevel::NONE) {
        
        if (count == 1) {
            return memory::UnifiedMemoryManager::getInstance().allocateObject<T>(
                memory::MemoryFlags::DEFAULT,
                protection_level,
                "rad_ml::memory_management::allocate"
            );
        } else {
            return memory::UnifiedMemoryManager::getInstance().allocateArray<T>(
                count,
                memory::MemoryFlags::DEFAULT,
                protection_level,
                "rad_ml::memory_management::allocate"
            );
        }
    }
    
    /**
     * @brief Deallocate memory
     * 
     * @param ptr Pointer to memory
     * @return True if deallocation was successful
     */
    inline bool deallocate(void* ptr) {
        return memory::UnifiedMemoryManager::getInstance().deallocate(ptr);
    }
    
    /**
     * @brief Create a radiation-tolerant smart pointer
     * 
     * @tparam T Type to manage
     * @tparam Args Constructor argument types
     * @param args Constructor arguments
     * @return RadiationTolerantPtr managing the object
     */
    template <typename T, typename... Args>
    inline memory::RadiationTolerantPtr<T> makeRadTolerant(Args&&... args) {
        return memory::makeRadTolerant<T>(std::forward<Args>(args)...);
    }
    
    /**
     * @brief Create a protected radiation-tolerant smart pointer
     * 
     * @tparam T Type to manage
     * @tparam Args Constructor argument types
     * @param protection_level Protection level
     * @param args Constructor arguments
     * @return RadiationTolerantPtr managing the protected object
     */
    template <typename T, typename... Args>
    inline memory::RadiationTolerantPtr<T> makeProtected(
        memory::MemoryProtectionLevel protection_level,
        Args&&... args) {
        
        return memory::makeRadTolerantProtected<T>(
            protection_level, 
            std::forward<Args>(args)...
        );
    }
    
    /**
     * @brief Check for memory leaks
     * 
     * @param report_to_log Whether to report leaks to the error log
     * @return Number of detected leaks
     */
    inline size_t checkForLeaks(bool report_to_log = true) {
        return memory::UnifiedMemoryManager::getInstance().checkForLeaks(report_to_log);
    }
    
    /**
     * @brief Get memory statistics
     * 
     * @return Current memory statistics
     */
    inline memory::MemoryStats getStats() {
        return memory::UnifiedMemoryManager::getInstance().getStats();
    }
}

/**
 * @brief High-level error handling functions
 */
namespace error_handling {
    /**
     * @brief Set the error reporting level
     * 
     * @param level Minimum severity level to log
     */
    inline void setReportingLevel(error::ErrorSeverity level) {
        error::ErrorHandler::setReportingLevel(level);
    }
    
    /**
     * @brief Create a custom error logger
     * 
     * @tparam Logger Custom logger type implementing IErrorLogger
     * @tparam Args Constructor argument types
     * @param args Constructor arguments
     */
    template <typename Logger, typename... Args>
    inline void setCustomLogger(Args&&... args) {
        error::ErrorHandler::setLogger(
            std::make_shared<Logger>(std::forward<Args>(args)...)
        );
    }
    
    /**
     * @brief Log an error
     * 
     * @param code Error code
     * @param category Error category
     * @param severity Error severity
     * @param message Error message
     * @param details Additional details (optional)
     */
    inline void logError(
        error::ErrorCode code,
        error::ErrorCategory category,
        error::ErrorSeverity severity,
        const std::string& message,
        const std::optional<std::string>& details = std::nullopt) {
        
        error::ErrorHandler::logError(
            error::ErrorInfo(
                code, category, severity, message, 
                std::source_location::current(), details
            )
        );
    }
    
    /**
     * @brief Create a success result
     * 
     * @tparam T Value type
     * @param value Value
     * @return Success result with given value
     */
    template <typename T>
    inline error::Result<T> makeSuccess(T value) {
        return error::Result<T>::success(std::move(value));
    }
    
    /**
     * @brief Create an error result
     * 
     * @tparam T Value type
     * @param code Error code
     * @param category Error category
     * @param severity Error severity
     * @param message Error message
     * @param details Additional details (optional)
     * @return Error result with given error information
     */
    template <typename T>
    inline error::Result<T> makeError(
        error::ErrorCode code,
        error::ErrorCategory category,
        error::ErrorSeverity severity,
        std::string message,
        std::optional<std::string> details = std::nullopt) {
        
        return error::Result<T>::error(
            code, category, severity, 
            std::move(message), std::source_location::current(), 
            std::move(details)
        );
    }
}

/**
 * @brief Neural network protection helpers
 */
namespace neural {
    /**
     * @brief Create a neural network with selective hardening
     * 
     * @tparam Network Neural network type
     * @tparam Args Constructor argument types
     * @param strategy Hardening strategy
     * @param protection_level Protection level
     * @param args Constructor arguments
     * @return Protected neural network instance
     */
    template <typename Network, typename... Args>
    inline std::unique_ptr<neural::SelectiveHardening<Network>> createProtectedNetwork(
        neural::HardeningStrategy strategy,
        neural::ProtectionLevel protection_level,
        Args&&... args) {
        
        auto network = std::make_unique<Network>(std::forward<Args>(args)...);
        auto protected_network = std::make_unique<neural::SelectiveHardening<Network>>(
            std::move(network), strategy, protection_level
        );
        
        return protected_network;
    }
    
    /**
     * @brief Create an error predictor for adaptive protection
     * 
     * @param model_path Path to pre-trained error prediction model
     * @return Error predictor instance
     */
    inline std::unique_ptr<neural::ErrorPredictor> createErrorPredictor(
        const std::string& model_path = "") {
        
        auto predictor = std::make_unique<neural::ErrorPredictor>();
        
        if (!model_path.empty()) {
            predictor->loadModel(model_path);
        }
        
        return predictor;
    }
}

/**
 * @brief Simulation and testing helpers
 */
namespace simulation {
    /**
     * @brief Create a physics-based radiation simulator
     * 
     * @param environment Radiation environment
     * @param intensity Radiation intensity (0.0-1.0)
     * @return Radiation simulator instance
     */
    inline std::unique_ptr<sim::PhysicsRadiationSimulator> createRadiationSimulator(
        sim::RadiationEnvironment environment = sim::RadiationEnvironment::EARTH_ORBIT,
        double intensity = 0.5) {
        
        auto simulator = std::make_unique<sim::PhysicsRadiationSimulator>(environment);
        simulator->setIntensity(intensity);
        
        return simulator;
    }
    
    /**
     * @brief Create a mission simulator for testing
     * 
     * @param mission_type Type of mission to simulate
     * @param duration_days Duration in days
     * @return Mission simulator instance
     */
    inline std::unique_ptr<testing::MissionSimulator> createMissionSimulator(
        mission::MissionType mission_type,
        size_t duration_days = 30) {
        
        auto simulator = std::make_unique<testing::MissionSimulator>();
        simulator->configureMission(mission_type, duration_days);
        
        return simulator;
    }
    
    /**
     * @brief Create a fault injector for testing
     * 
     * @param fault_rate Average faults per second
     * @return Fault injector instance
     */
    inline std::unique_ptr<testing::FaultInjector> createFaultInjector(double fault_rate = 0.01) {
        auto injector = std::make_unique<testing::FaultInjector>();
        injector->setFaultRate(fault_rate);
        
        return injector;
    }
}

} // namespace rad_ml 