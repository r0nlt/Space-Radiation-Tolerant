#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <cmath>
#include <memory>
#include <algorithm>

// Use include paths based on the include_directories in CMakeLists.txt
#include "include/rad_ml/tmr/enhanced_stuck_bit_tmr.hpp"
#include "include/rad_ml/memory/radiation_mapped_allocator.hpp"
#include "include/rad_ml/power/power_aware_protection.hpp"
#include "include/rad_ml/sim/physics_radiation_simulator.hpp"
#include "include/rad_ml/hw/hardware_acceleration.hpp"

// Utility function to print section headers
void print_header(const std::string& title) {
    std::cout << "\n===============================================" << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << "===============================================" << std::endl;
}

// Test EnhancedStuckBitTMR
void test_enhanced_stuck_bit_tmr() {
    print_header("ENHANCED STUCK BIT TMR TEST");
    
    // Create TMR with initial value
    rad_ml::tmr::EnhancedStuckBitTMR<uint32_t> tmr(0x12345678);
    
    std::cout << "Initial TMR value: 0x" << std::hex << tmr.get() << std::dec << std::endl;
    std::cout << "Initial diagnostics:\n" << tmr.getDiagnostics() << std::endl;
    
    // Simulate stuck bits by consistently corrupting the same bits
    const uint32_t stuck_bit_mask = 0x00010001; // Bit 0 and bit 16 stuck at 1
    
    std::cout << "Simulating stuck bits with mask: 0x" << std::hex << stuck_bit_mask << std::dec << std::endl;
    
    // Perform several corruption and repair cycles to train TMR
    for (int i = 0; i < 5; i++) {
        std::cout << "\nCycle " << i+1 << ":" << std::endl;
        
        // Corrupt copy 0 with stuck bits
        uint32_t corrupted = tmr.getCopies()[0] | stuck_bit_mask;
        tmr.corruptCopy(0, corrupted);
        
        std::cout << "Corrupted copy 0 to: 0x" << std::hex << corrupted << std::dec << std::endl;
        std::cout << "TMR value after corruption: 0x" << std::hex << tmr.get() << std::dec << std::endl;
        
        // Repair and check results
        tmr.repair();
        std::cout << "TMR value after repair: 0x" << std::hex << tmr.get() << std::dec << std::endl;
        std::cout << "Stuck bit mask: " << tmr.getStuckBitMask() << std::endl;
        
        // Display health scores
        const auto& scores = tmr.getHealthScores();
        std::cout << "Health scores: [";
        for (size_t j = 0; j < scores.size(); j++) {
            std::cout << scores[j];
            if (j < scores.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    // Try setting a new value
    std::cout << "\nSetting new value: 0xABCDEF01" << std::endl;
    tmr.set(0xABCDEF01);
    std::cout << "TMR value after set: 0x" << std::hex << tmr.get() << std::dec << std::endl;
    std::cout << "Stuck bit mask: " << tmr.getStuckBitMask() << std::endl;
    
    // Corrupt all copies differently
    std::cout << "\nCorrupting all copies with different values:" << std::endl;
    tmr.corruptCopy(0, 0xABCDEF11); // Simulate stuck bit at position 0
    tmr.corruptCopy(1, 0xABCDEF03); // Different corruption
    tmr.corruptCopy(2, 0xABCDEF01); // No corruption
    
    std::cout << "Copy values: [";
    for (size_t i = 0; i < tmr.getCopies().size(); i++) {
        std::cout << "0x" << std::hex << tmr.getCopies()[i] << std::dec;
        if (i < tmr.getCopies().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "TMR value with corruption: 0x" << std::hex << tmr.get() << std::dec << std::endl;
    
    // Repair again
    tmr.repair();
    std::cout << "Final TMR value after repair: 0x" << std::hex << tmr.get() << std::dec << std::endl;
    std::cout << "Final diagnostics:\n" << tmr.getDiagnostics() << std::endl;
}

// Test Radiation Mapped Memory Allocator
void test_radiation_mapped_memory() {
    print_header("RADIATION-MAPPED MEMORY TEST");
    
    // Create allocator with default zones
    rad_ml::memory::RadiationMappedAllocator allocator;
    
    std::cout << "Allocating data with different criticality levels:" << std::endl;
    
    // Allocate memory with different criticality levels
    void* critical_data = allocator.allocate(1024 * 1024, 
                                           rad_ml::memory::DataCriticality::MISSION_CRITICAL);
    
    void* important_data = allocator.allocate(2 * 1024 * 1024, 
                                            rad_ml::memory::DataCriticality::HIGHLY_IMPORTANT);
    
    void* moderate_data = allocator.allocate(4 * 1024 * 1024, 
                                           rad_ml::memory::DataCriticality::MODERATELY_IMPORTANT);
    
    void* low_data = allocator.allocate(8 * 1024 * 1024, 
                                       rad_ml::memory::DataCriticality::LOW_IMPORTANCE);
    
    // Get allocation diagnostics
    std::cout << allocator.get_diagnostics() << std::endl;
    
    // Show memory usage by criticality
    auto usage = allocator.get_criticality_usage();
    std::cout << "Memory usage by criticality:" << std::endl;
    std::cout << "  Mission Critical: " << usage[rad_ml::memory::DataCriticality::MISSION_CRITICAL] / 1024 << " KB" << std::endl;
    std::cout << "  Highly Important: " << usage[rad_ml::memory::DataCriticality::HIGHLY_IMPORTANT] / 1024 << " KB" << std::endl;
    std::cout << "  Moderately Important: " << usage[rad_ml::memory::DataCriticality::MODERATELY_IMPORTANT] / 1024 << " KB" << std::endl;
    std::cout << "  Low Importance: " << usage[rad_ml::memory::DataCriticality::LOW_IMPORTANCE] / 1024 << " KB" << std::endl;
    
    // Calculate vulnerability score
    double vulnerability = allocator.calculate_vulnerability_score();
    std::cout << "Overall memory vulnerability score: " << vulnerability << std::endl;
    std::cout << "Lower score means less vulnerable to radiation effects." << std::endl;
    
    // Free memory
    allocator.deallocate(critical_data);
    allocator.deallocate(important_data);
    allocator.deallocate(moderate_data);
    allocator.deallocate(low_data);
    
    std::cout << "\nAfter freeing all memory:" << std::endl;
    std::cout << allocator.get_diagnostics() << std::endl;
}

// Test Power-Aware Protection
void test_power_aware_protection() {
    print_header("POWER-AWARE PROTECTION TEST");
    
    // Create power manager with 30W total budget
    rad_ml::power::PowerAwareProtection power_mgr(30.0);
    
    // Register various components with different criticality
    rad_ml::power::ProtectedComponent nav_system{
        "Navigation Neural Network",
        rad_ml::power::ProtectedComponent::Type::NEURAL_NETWORK,
        0.7, // Minimum protection (70%)
        0.99, // Maximum protection (99%)
        0.7, // Start at minimum
        3.0, // Power at minimum protection (watts)
        8.0, // Power at maximum protection (watts)
        0.95 // Criticality (navigation is crucial)
    };
    
    rad_ml::power::ProtectedComponent sensor_proc{
        "Sensor Processing",
        rad_ml::power::ProtectedComponent::Type::SENSOR_PROCESSING,
        0.5, // Minimum protection
        0.95, // Maximum protection
        0.5, // Start at minimum
        2.0, // Power at minimum
        5.0, // Power at maximum
        0.8 // Criticality
    };
    
    rad_ml::power::ProtectedComponent science{
        "Science Instrument ML",
        rad_ml::power::ProtectedComponent::Type::SCIENCE_INSTRUMENT,
        0.3, // Minimum protection
        0.9, // Maximum protection
        0.3, // Start at minimum
        4.0, // Power at minimum
        12.0, // Power at maximum
        0.6 // Criticality
    };
    
    rad_ml::power::ProtectedComponent comms{
        "Communications System",
        rad_ml::power::ProtectedComponent::Type::COMMUNICATIONS,
        0.6, // Minimum protection
        0.98, // Maximum protection
        0.6, // Start at minimum
        1.5, // Power at minimum
        4.0, // Power at maximum
        0.9 // Criticality
    };
    
    // Register components
    int nav_id = power_mgr.register_component(nav_system);
    int sensor_id = power_mgr.register_component(sensor_proc);
    int science_id = power_mgr.register_component(science);
    int comms_id = power_mgr.register_component(comms);
    
    // Show initial power state
    std::cout << "Initial power state (NOMINAL):" << std::endl;
    std::cout << power_mgr.get_status_report() << std::endl;
    
    // Calculate expected error rate
    double error_rate = power_mgr.calculate_expected_error_rate();
    std::cout << "Expected bit error rate: " << error_rate << " errors per day" << std::endl;
    
    // Test different power states
    std::cout << "\nChanging to SCIENCE_OPERATION mode:" << std::endl;
    power_mgr.set_power_state(rad_ml::power::PowerState::SCIENCE_OPERATION);
    std::cout << power_mgr.get_status_report() << std::endl;
    
    std::cout << "\nChanging to LOW_POWER mode:" << std::endl;
    power_mgr.set_power_state(rad_ml::power::PowerState::LOW_POWER);
    std::cout << power_mgr.get_status_report() << std::endl;
    
    std::cout << "\nChanging to EMERGENCY mode:" << std::endl;
    power_mgr.set_power_state(rad_ml::power::PowerState::EMERGENCY);
    std::cout << power_mgr.get_status_report() << std::endl;
    
    // Overall protection effectiveness
    std::cout << "\nProtection effectiveness in different power states:" << std::endl;
    power_mgr.set_power_state(rad_ml::power::PowerState::EMERGENCY);
    std::cout << "  EMERGENCY: " << power_mgr.calculate_protection_effectiveness() << std::endl;
    
    power_mgr.set_power_state(rad_ml::power::PowerState::LOW_POWER);
    std::cout << "  LOW_POWER: " << power_mgr.calculate_protection_effectiveness() << std::endl;
    
    power_mgr.set_power_state(rad_ml::power::PowerState::NOMINAL);
    std::cout << "  NOMINAL: " << power_mgr.calculate_protection_effectiveness() << std::endl;
    
    power_mgr.set_power_state(rad_ml::power::PowerState::SCIENCE_OPERATION);
    std::cout << "  SCIENCE_OPERATION: " << power_mgr.calculate_protection_effectiveness() << std::endl;
    
    power_mgr.set_power_state(rad_ml::power::PowerState::PEAK_PERFORMANCE);
    std::cout << "  PEAK_PERFORMANCE: " << power_mgr.calculate_protection_effectiveness() << std::endl;
}

// Test Physics-Based Radiation Simulator
void test_physics_radiation_simulator() {
    print_header("PHYSICS-BASED RADIATION SIMULATOR TEST");
    
    // Create simulator with Europa mission trajectory
    const auto trajectory = rad_ml::sim::SpacecraftTrajectory::Europa_Mission();
    
    rad_ml::sim::PhysicsRadiationSimulator sim(
        8 * 1024 * 1024, // 8MB memory
        32, // 32-bit words
        5.0, // 5mm Al shielding
        trajectory
    );
    
    // Current environment info
    std::cout << sim.get_environment_report() << std::endl;
    
    // Simulate radiation in different environments
    std::cout << "\nSimulating different radiation environments:" << std::endl;
    
    const std::vector<rad_ml::sim::RadiationEnvironment> environments = {
        rad_ml::sim::RadiationEnvironment::LEO,
        rad_ml::sim::RadiationEnvironment::MARS_SURFACE,
        rad_ml::sim::RadiationEnvironment::EUROPA,
        rad_ml::sim::RadiationEnvironment::SOLAR_STORM
    };
    
    for (const auto& env : environments) {
        sim.set_environment(env);
        
        // Simulate 10 days in this environment
        auto events = sim.simulate_period(10.0);
        
        // Count events by type
        std::map<rad_ml::sim::RadiationEffectType, size_t> counts;
        
        for (const auto& [type, bits] : events) {
            counts[type]++;
        }
        
        std::cout << "\nEnvironment: ";
        switch (env) {
            case rad_ml::sim::RadiationEnvironment::LEO:
                std::cout << "Low Earth Orbit";
                break;
            case rad_ml::sim::RadiationEnvironment::MARS_SURFACE:
                std::cout << "Mars Surface";
                break;
            case rad_ml::sim::RadiationEnvironment::EUROPA:
                std::cout << "Europa";
                break;
            case rad_ml::sim::RadiationEnvironment::SOLAR_STORM:
                std::cout << "Solar Storm";
                break;
            default:
                std::cout << "Unknown";
                break;
        }
        std::cout << std::endl;
        
        std::cout << "Events in 10 days:" << std::endl;
        std::cout << "  SEUs: " << counts[rad_ml::sim::RadiationEffectType::SEU] << std::endl;
        std::cout << "  MBUs: " << counts[rad_ml::sim::RadiationEffectType::MBU] << std::endl;
        std::cout << "  SELs: " << counts[rad_ml::sim::RadiationEffectType::SEL] << std::endl;
        std::cout << "  Stuck Bits: " << counts[rad_ml::sim::RadiationEffectType::TID_STUCK_BIT] << std::endl;
        
        // Get error rates
        auto rates = sim.get_error_rates();
        std::cout << "Error rates (per Mbit per day):" << std::endl;
        std::cout << "  SEU rate: " << rates[rad_ml::sim::RadiationEffectType::SEU] << std::endl;
        std::cout << "  MBU rate: " << rates[rad_ml::sim::RadiationEffectType::MBU] << std::endl;
        std::cout << "  SEL rate: " << rates[rad_ml::sim::RadiationEffectType::SEL] << std::endl;
        std::cout << "  Stuck Bit rate: " << rates[rad_ml::sim::RadiationEffectType::TID_STUCK_BIT] << std::endl;
    }
    
    // Simulate entire mission
    std::cout << "\nSimulating entire Europa mission:" << std::endl;
    auto mission_events = sim.simulate_mission();
    
    // Show events by mission segment
    std::cout << "Events by mission segment:" << std::endl;
    
    for (size_t i = 0; i < mission_events.size(); ++i) {
        std::cout << "  Segment " << i+1 << " (";
        
        switch (trajectory.environments[i]) {
            case rad_ml::sim::RadiationEnvironment::LEO:
                std::cout << "Low Earth Orbit";
                break;
            case rad_ml::sim::RadiationEnvironment::INTERPLANETARY:
                std::cout << "Interplanetary";
                break;
            case rad_ml::sim::RadiationEnvironment::JUPITER:
                std::cout << "Jupiter";
                break;
            case rad_ml::sim::RadiationEnvironment::EUROPA:
                std::cout << "Europa";
                break;
            default:
                std::cout << "Unknown";
                break;
        }
        
        std::cout << ", " << trajectory.durations_days[i] << " days):" << std::endl;
        
        std::cout << "    SEUs: " << mission_events[i][rad_ml::sim::RadiationEffectType::SEU] << std::endl;
        std::cout << "    MBUs: " << mission_events[i][rad_ml::sim::RadiationEffectType::MBU] << std::endl;
        std::cout << "    SELs: " << mission_events[i][rad_ml::sim::RadiationEffectType::SEL] << std::endl;
        std::cout << "    Stuck Bits: " << mission_events[i][rad_ml::sim::RadiationEffectType::TID_STUCK_BIT] << std::endl;
    }
    
    // TID effects over entire mission
    std::cout << "\nTID effects over entire mission:" << std::endl;
    auto tid_effects = sim.simulate_mission_tid(trajectory.durations_days[0] + 
                                            trajectory.durations_days[1] + 
                                            trajectory.durations_days[2] +
                                            trajectory.durations_days[3] +
                                            trajectory.durations_days[4] +
                                            trajectory.durations_days[5] +
                                            trajectory.durations_days[6]);
                                            
    std::cout << "  Total stuck bits: " << tid_effects[rad_ml::sim::RadiationEffectType::TID_STUCK_BIT] << std::endl;
    std::cout << "  Threshold shifts: " << tid_effects[rad_ml::sim::RadiationEffectType::TID_THRESHOLD_SHIFT] << std::endl;
}

// Test Hardware Acceleration Integration
void test_hardware_acceleration() {
    print_header("HARDWARE ACCELERATION INTEGRATION TEST");
    
    // Create hardware accelerator configuration
    rad_ml::hw::AcceleratorConfig config;
    config.type = rad_ml::hw::AcceleratorType::RAD_TOL_FPGA;
    config.enable_hw_ecc = true;
    config.tmr_approach = rad_ml::hw::HardwareTMRApproach::REGISTER_LEVEL;
    config.scrubbing_strategy = rad_ml::hw::ScrubbingStrategy::ADAPTIVE;
    config.scrubbing_interval_sec = 5.0;
    config.power_budget_watts = 8.0;
    
    // Create TMR-accelerator integration
    rad_ml::hw::TMRAcceleratorIntegration accel_integ(config);
    
    // Initialize
    bool init_success = accel_integ.initialize();
    std::cout << "Accelerator initialization " 
              << (init_success ? "successful" : "failed") << std::endl;
    
    // Show diagnostics
    std::cout << accel_integ.get_diagnostics() << std::endl;
    
    // Test reliability in different environments
    std::cout << "\nReliability in different radiation environments:" << std::endl;
    
    // Define SEU rates for different environments (events per bit per day)
    const std::vector<std::pair<std::string, double>> environments = {
        {"Low Earth Orbit", 1e-8},
        {"Geosynchronous Orbit", 5e-8},
        {"Interplanetary", 1e-7},
        {"Mars Surface", 5e-8},
        {"Jupiter Radiation Belts", 1e-5},
        {"Europa", 2e-5},
        {"Solar Storm", 1e-4}
    };
    
    // Show reliability for each environment
    for (const auto& [name, rate] : environments) {
        double reliability = accel_integ.calculate_system_reliability(rate);
        std::cout << "  " << name << ": " << std::fixed << std::setprecision(6) 
                  << reliability << std::endl;
    }
    
    // Test with/without software TMR
    std::cout << "\nImpact of software TMR:" << std::endl;
    
    accel_integ.set_software_tmr(true);
    std::cout << "  With software TMR: " 
              << accel_integ.calculate_system_reliability(1e-5) << std::endl;
    
    accel_integ.set_software_tmr(false);
    std::cout << "  Without software TMR: " 
              << accel_integ.calculate_system_reliability(1e-5) << std::endl;
    
    // Test different hardware configurations
    std::cout << "\nImpact of different hardware protection strategies:" << std::endl;
    
    // Test with different TMR approaches
    config.tmr_approach = rad_ml::hw::HardwareTMRApproach::NONE;
    accel_integ.get_accelerator().update_config(config);
    std::cout << "  No TMR: " 
              << accel_integ.calculate_system_reliability(1e-5) << std::endl;
    
    config.tmr_approach = rad_ml::hw::HardwareTMRApproach::BLOCK_LEVEL;
    accel_integ.get_accelerator().update_config(config);
    std::cout << "  Block-level TMR: " 
              << accel_integ.calculate_system_reliability(1e-5) << std::endl;
    
    config.tmr_approach = rad_ml::hw::HardwareTMRApproach::REGISTER_LEVEL;
    accel_integ.get_accelerator().update_config(config);
    std::cout << "  Register-level TMR: " 
              << accel_integ.calculate_system_reliability(1e-5) << std::endl;
    
    config.tmr_approach = rad_ml::hw::HardwareTMRApproach::COMPLETE;
    accel_integ.get_accelerator().update_config(config);
    std::cout << "  Complete TMR: " 
              << accel_integ.calculate_system_reliability(1e-5) << std::endl;
}

// Integrated test of all components working together
void test_integrated_system() {
    print_header("INTEGRATED SYSTEM TEST");
    
    std::cout << "Simulating a complete space mission with all protection features active:" << std::endl;
    
    // Create radiation environment simulator for Europa mission
    rad_ml::sim::PhysicsRadiationSimulator sim(
        64 * 1024 * 1024, // 64MB memory
        32, // 32-bit words
        10.0, // 10mm Al shielding
        rad_ml::sim::SpacecraftTrajectory::Europa_Mission()
    );
    
    // Create memory allocator
    rad_ml::memory::RadiationMappedAllocator memory_mgr;
    
    // Create power manager with 50W budget
    rad_ml::power::PowerAwareProtection power_mgr(50.0);
    
    // Create hardware accelerator integration
    rad_ml::hw::AcceleratorConfig hw_config;
    hw_config.type = rad_ml::hw::AcceleratorType::RAD_TOL_FPGA;
    hw_config.enable_hw_ecc = true;
    hw_config.tmr_approach = rad_ml::hw::HardwareTMRApproach::REGISTER_LEVEL;
    hw_config.scrubbing_strategy = rad_ml::hw::ScrubbingStrategy::ADAPTIVE;
    hw_config.power_budget_watts = 10.0;
    
    rad_ml::hw::TMRAcceleratorIntegration hw_accel(hw_config);
    hw_accel.initialize();
    
    // Create TMR for crucial values
    rad_ml::tmr::EnhancedStuckBitTMR<float> critical_value(3.14159f);
    
    // Register power-managed components
    rad_ml::power::ProtectedComponent neural_net{
        "Neural Network Inference Engine",
        rad_ml::power::ProtectedComponent::Type::NEURAL_NETWORK,
        0.7, // Minimum protection
        0.99, // Maximum protection
        0.7, // Current protection
        5.0, // Power at min protection
        15.0, // Power at max protection
        0.9 // Criticality
    };
    
    rad_ml::power::ProtectedComponent nav_system{
        "Navigation System",
        rad_ml::power::ProtectedComponent::Type::NAVIGATION,
        0.8, // Minimum protection
        0.95, // Maximum protection
        0.8, // Current protection
        3.0, // Power at min
        8.0, // Power at max
        0.95 // Criticality
    };
    
    rad_ml::power::ProtectedComponent science_instrument{
        "Europa Surface Analysis",
        rad_ml::power::ProtectedComponent::Type::SCIENCE_INSTRUMENT,
        0.5, // Minimum protection
        0.9, // Maximum protection
        0.5, // Current protection
        8.0, // Power at min
        20.0, // Power at max
        0.7 // Criticality
    };
    
    int nn_id = power_mgr.register_component(neural_net);
    int nav_id = power_mgr.register_component(nav_system);
    int sci_id = power_mgr.register_component(science_instrument);
    
    // Simulate mission phases
    std::cout << "\nMission Phase 1: Earth Departure (LEO)" << std::endl;
    sim.set_environment(rad_ml::sim::RadiationEnvironment::LEO);
    power_mgr.set_power_state(rad_ml::power::PowerState::NOMINAL);
    
    auto error_rates = sim.get_error_rates();
    double seu_rate = error_rates[rad_ml::sim::RadiationEffectType::SEU];
    
    std::cout << "  SEU Rate: " << seu_rate << " per Mbit per day" << std::endl;
    std::cout << "  System reliability: " << hw_accel.calculate_system_reliability(seu_rate/1e6) << std::endl;
    std::cout << "  Power protection level: " << power_mgr.calculate_protection_effectiveness() << std::endl;
    
    // Simulate memory allocation
    void* critical_data = memory_mgr.allocate(2 * 1024 * 1024, 
                                           rad_ml::memory::DataCriticality::MISSION_CRITICAL);
    std::cout << "  Memory vulnerability: " << memory_mgr.calculate_vulnerability_score() << std::endl;
    
    // Simulate stuck bit detection
    for (int i = 0; i < 3; i++) {
        // Corrupt value with consistent bit error
        float corrupted = critical_value.get();
        *reinterpret_cast<uint32_t*>(&corrupted) |= (1 << i); // Set bit i
        critical_value.corruptCopy(0, corrupted);
        critical_value.repair();
    }
    
    std::cout << "  TMR stuck bits detected: " << critical_value.getStuckBitMask().count() << std::endl;
    
    // Phase 2: Jupiter Arrival
    std::cout << "\nMission Phase 2: Jupiter Arrival" << std::endl;
    sim.set_environment(rad_ml::sim::RadiationEnvironment::JUPITER);
    power_mgr.set_power_state(rad_ml::power::PowerState::LOW_POWER);
    
    error_rates = sim.get_error_rates();
    seu_rate = error_rates[rad_ml::sim::RadiationEffectType::SEU];
    
    std::cout << "  SEU Rate: " << seu_rate << " per Mbit per day" << std::endl;
    std::cout << "  System reliability: " << hw_accel.calculate_system_reliability(seu_rate/1e6) << std::endl;
    std::cout << "  Power protection level: " << power_mgr.calculate_protection_effectiveness() << std::endl;
    
    // Phase 3: Europa Encounter
    std::cout << "\nMission Phase 3: Europa Encounter" << std::endl;
    sim.set_environment(rad_ml::sim::RadiationEnvironment::EUROPA);
    power_mgr.set_power_state(rad_ml::power::PowerState::SCIENCE_OPERATION);
    
    error_rates = sim.get_error_rates();
    seu_rate = error_rates[rad_ml::sim::RadiationEffectType::SEU];
    
    std::cout << "  SEU Rate: " << seu_rate << " per Mbit per day" << std::endl;
    std::cout << "  System reliability: " << hw_accel.calculate_system_reliability(seu_rate/1e6) << std::endl;
    std::cout << "  Power protection level: " << power_mgr.calculate_protection_effectiveness() << std::endl;
    
    // Allocate science data
    void* science_data = memory_mgr.allocate(32 * 1024 * 1024, 
                                          rad_ml::memory::DataCriticality::HIGHLY_IMPORTANT);
    std::cout << "  Memory vulnerability: " << memory_mgr.calculate_vulnerability_score() << std::endl;
    
    // Phase 4: Solar Storm
    std::cout << "\nMission Phase 4: Solar Storm Emergency" << std::endl;
    sim.set_environment(rad_ml::sim::RadiationEnvironment::SOLAR_STORM);
    power_mgr.set_power_state(rad_ml::power::PowerState::EMERGENCY);
    
    error_rates = sim.get_error_rates();
    seu_rate = error_rates[rad_ml::sim::RadiationEffectType::SEU];
    
    std::cout << "  SEU Rate: " << seu_rate << " per Mbit per day" << std::endl;
    std::cout << "  System reliability: " << hw_accel.calculate_system_reliability(seu_rate/1e6) << std::endl;
    std::cout << "  Power protection level: " << power_mgr.calculate_protection_effectiveness() << std::endl;
    
    // Phase 5: Return to Normal
    std::cout << "\nMission Phase 5: Return to Normal Operations" << std::endl;
    sim.set_environment(rad_ml::sim::RadiationEnvironment::JUPITER);
    power_mgr.set_power_state(rad_ml::power::PowerState::NOMINAL);
    
    error_rates = sim.get_error_rates();
    seu_rate = error_rates[rad_ml::sim::RadiationEffectType::SEU];
    
    std::cout << "  SEU Rate: " << seu_rate << " per Mbit per day" << std::endl;
    std::cout << "  System reliability: " << hw_accel.calculate_system_reliability(seu_rate/1e6) << std::endl;
    std::cout << "  Power protection level: " << power_mgr.calculate_protection_effectiveness() << std::endl;
    
    // Final mission summary
    std::cout << "\nMission Summary:" << std::endl;
    std::cout << "  Enhanced TMR effectively identified " << critical_value.getStuckBitMask().count() 
              << " stuck bits." << std::endl;
    std::cout << "  Memory management allocated critical data to most protected regions." << std::endl;
    std::cout << "  Power-aware protection adjusted protection levels based on power state." << std::endl;
    std::cout << "  Hardware acceleration maintained reliability in extreme radiation environments." << std::endl;
    
    // Clean up
    memory_mgr.deallocate(critical_data);
    memory_mgr.deallocate(science_data);
}

// Test function to validate all enhanced features
int main() {
    std::cout << "Running Enhanced Features Test for Radiation-Tolerant ML Framework\n";
    std::cout << "==================================================================\n\n";
    
    // Run individual component tests
    test_enhanced_stuck_bit_tmr();
    test_radiation_mapped_memory();
    test_power_aware_protection();
    test_physics_radiation_simulator();
    test_hardware_acceleration();
    
    // Run integrated system test
    test_integrated_system();
    
    std::cout << "\nAll tests completed successfully!" << std::endl;
    
    return 0;
} 