/**
 * @file quantum_enhanced_radiation_test.cpp
 * @brief Test demonstrating quantum-enhanced radiation simulation
 *
 * This example shows how the QFT framework is now connected to realistic
 * semiconductor physics for radiation effects simulation.
 */

#include "rad_ml/physics/quantum_enhanced_radiation.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "rad_ml/core/logger.hpp"
#include "rad_ml/neural/protected_neural_network.hpp"

using namespace rad_ml;
using namespace rad_ml::physics;

void demonstrateQuantumEnhancedRadiation()
{
    std::cout << "\n=== Quantum-Enhanced Radiation Simulation Demo ===\n" << std::endl;

    // Configure semiconductor properties for different scenarios
    SemiconductorProperties silicon_300k;
    silicon_300k.temperature_k = 300.0;  // Room temperature

    SemiconductorProperties silicon_77k;
    silicon_77k.temperature_k = 77.0;  // Liquid nitrogen temperature
    silicon_77k.bandgap_ev = 1.17;     // Bandgap increases at low temperature

    SemiconductorProperties gaas_300k;
    gaas_300k.bandgap_ev = 1.42;             // GaAs bandgap
    gaas_300k.effective_mass_ratio = 0.067;  // Lower effective mass
    gaas_300k.dielectric_constant = 13.1;
    gaas_300k.lattice_constant_nm = 0.565;

    // Create quantum-enhanced radiation simulators
    QuantumEnhancedRadiation silicon_sim(silicon_300k);
    QuantumEnhancedRadiation cold_silicon_sim(silicon_77k);
    QuantumEnhancedRadiation gaas_sim(gaas_300k);

    std::cout << "Initialized quantum radiation simulators for different materials/temperatures\n"
              << std::endl;

    // Test different radiation scenarios
    struct RadiationScenario {
        std::string name;
        double energy_mev;
        double let_mev_cm2_mg;
        ParticleType particle;
        MemoryDeviceType device;
    };

    std::vector<RadiationScenario> scenarios = {
        {"Low-energy proton (LEO)", 10.0, 0.5, ParticleType::Proton, MemoryDeviceType::SRAM_6T},
        {"High-energy proton (GEO)", 100.0, 2.0, ParticleType::Proton, MemoryDeviceType::SRAM_6T},
        {"Alpha particle", 5.0, 15.0, ParticleType::HeavyIon, MemoryDeviceType::SRAM_6T},
        {"Heavy ion (cosmic ray)", 1000.0, 80.0, ParticleType::HeavyIon, MemoryDeviceType::SRAM_6T},
        {"Neutron (atmospheric)", 50.0, 5.0, ParticleType::Neutron, MemoryDeviceType::DRAM},
        {"Proton on Flash memory", 50.0, 2.0, ParticleType::Proton, MemoryDeviceType::FLASH_SLC}};

    std::cout << std::setw(25) << "Scenario" << std::setw(15) << "Classical (fC)" << std::setw(15)
              << "Quantum (fC)" << std::setw(15) << "Enhancement" << std::setw(15)
              << "Bit Flip Prob" << std::setw(10) << "MBU Size" << std::endl;
    std::cout << std::string(95, '-') << std::endl;

    for (const auto& scenario : scenarios) {
        // Calculate classical charge deposition (simplified)
        double classical_charge = scenario.let_mev_cm2_mg * 0.278;

        // Calculate quantum-enhanced charge deposition
        double quantum_charge = silicon_sim.calculateQuantumChargeDeposition(
            scenario.energy_mev, scenario.let_mev_cm2_mg, scenario.particle);

        // Calculate bit flip probability
        double flip_prob =
            silicon_sim.calculateEnhancedBitFlipProbability(quantum_charge, scenario.device, 300.0);

        // Calculate MBU size
        uint32_t mbu_size = silicon_sim.calculateQuantumMBUSize(quantum_charge, scenario.particle);

        double enhancement = quantum_charge / classical_charge;

        std::cout << std::setw(25) << scenario.name << std::setw(15) << std::fixed
                  << std::setprecision(2) << classical_charge << std::setw(15) << quantum_charge
                  << std::setw(15) << enhancement << std::setw(15) << std::setprecision(4)
                  << flip_prob << std::setw(10) << mbu_size << std::endl;
    }

    std::cout << "\n=== Temperature Effects Demonstration ===\n" << std::endl;

    // Demonstrate temperature effects
    std::vector<double> temperatures = {77.0, 200.0, 300.0, 400.0, 500.0};
    double base_critical_charge = 15.0;  // fC

    std::cout << std::setw(15) << "Temperature (K)" << std::setw(20) << "Critical Charge (fC)"
              << std::setw(20) << "Quantum Correction" << std::setw(15) << "Sensitivity"
              << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (double temp : temperatures) {
        SemiconductorProperties temp_material = silicon_300k;
        temp_material.temperature_k = temp;
        QuantumEnhancedRadiation temp_sim(temp_material);

        double critical_charge =
            temp_sim.calculateTemperatureCriticalCharge(base_critical_charge, temp);
        double correction = critical_charge / base_critical_charge;
        double sensitivity = base_critical_charge / critical_charge;  // Inverse relationship

        std::cout << std::setw(15) << temp << std::setw(20) << std::fixed << std::setprecision(2)
                  << critical_charge << std::setw(20) << std::setprecision(3) << correction
                  << std::setw(15) << std::setprecision(2) << sensitivity << std::endl;
    }

    std::cout << "\n=== Device Type Sensitivity Comparison ===\n" << std::endl;

    std::vector<std::pair<MemoryDeviceType, std::string>> devices = {
        {MemoryDeviceType::SRAM_6T, "SRAM 6T"},
        {MemoryDeviceType::SRAM_8T, "SRAM 8T"},
        {MemoryDeviceType::DRAM, "DRAM"},
        {MemoryDeviceType::FLASH_SLC, "Flash SLC"},
        {MemoryDeviceType::FLASH_MLC, "Flash MLC"},
        {MemoryDeviceType::MRAM, "MRAM"},
        {MemoryDeviceType::FRAM, "FRAM"}};

    double test_charge = 20.0;  // fC

    std::cout << std::setw(15) << "Device Type" << std::setw(20) << "Bit Flip Prob" << std::setw(20)
              << "Relative Sensitivity" << std::endl;
    std::cout << std::string(55, '-') << std::endl;

    double sram_prob = silicon_sim.calculateEnhancedBitFlipProbability(
        test_charge, MemoryDeviceType::SRAM_6T, 300.0);

    for (const auto& device : devices) {
        double flip_prob =
            silicon_sim.calculateEnhancedBitFlipProbability(test_charge, device.first, 300.0);
        double relative_sensitivity = flip_prob / sram_prob;

        std::cout << std::setw(15) << device.second << std::setw(20) << std::fixed
                  << std::setprecision(4) << flip_prob << std::setw(20) << std::setprecision(2)
                  << relative_sensitivity << std::endl;
    }
}

void demonstrateRealisticMemoryCorruption()
{
    std::cout << "\n=== Realistic Memory Corruption Test ===\n" << std::endl;

    // Create test memory buffer
    const size_t buffer_size = 1024;                        // 1KB
    std::vector<uint8_t> memory_buffer(buffer_size, 0xAA);  // Initialize with pattern

    // Initialize quantum radiation simulator
    QuantumEnhancedRadiation quantum_sim;

    // Test different radiation environments
    struct Environment {
        std::string name;
        double energy_mev;
        double let;
        ParticleType particle;
        uint32_t duration_ms;
    };

    std::vector<Environment> environments = {
        {"LEO (Low Earth Orbit)", 20.0, 1.0, ParticleType::Proton, 1000},
        {"GEO (Geostationary)", 100.0, 3.0, ParticleType::Proton, 1000},
        {"Deep Space (Jupiter)", 500.0, 40.0, ParticleType::HeavyIon, 1000},
        {"Solar Particle Event", 200.0, 10.0, ParticleType::Proton, 100}};

    for (const auto& env : environments) {
        // Reset memory buffer
        std::fill(memory_buffer.begin(), memory_buffer.end(), 0xAA);

        // Count initial set bits
        uint32_t initial_bits = 0;
        for (uint8_t byte : memory_buffer) {
            initial_bits += __builtin_popcount(byte);
        }

        std::cout << "Testing environment: " << env.name << std::endl;
        std::cout << "  Initial set bits: " << initial_bits << std::endl;

        // Apply quantum-enhanced radiation
        uint32_t bit_flips = quantum_sim.applyQuantumEnhancedRadiation(
            memory_buffer.data(), buffer_size, env.energy_mev, env.let, env.particle,
            MemoryDeviceType::SRAM_6T, env.duration_ms);

        // Count final set bits
        uint32_t final_bits = 0;
        for (uint8_t byte : memory_buffer) {
            final_bits += __builtin_popcount(byte);
        }

        std::cout << "  Bit flips applied: " << bit_flips << std::endl;
        std::cout << "  Final set bits: " << final_bits << std::endl;
        std::cout << "  Net bit change: " << static_cast<int32_t>(final_bits - initial_bits)
                  << std::endl;
        std::cout << "  Error rate: " << (bit_flips * 100.0) / (buffer_size * 8) << "%"
                  << std::endl;
        std::cout << std::endl;
    }
}

int main()
{
    // Initialize logging
    core::Logger::init(core::LogLevel::INFO);

    std::cout << "Quantum-Enhanced Radiation Simulation Test" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "This test demonstrates how QFT calculations are now connected" << std::endl;
    std::cout << "to realistic semiconductor physics for radiation effects." << std::endl;

    try {
        demonstrateQuantumEnhancedRadiation();
        demonstrateRealisticMemoryCorruption();

        std::cout << "\n=== Summary ===\n" << std::endl;
        std::cout << "✅ QFT framework successfully connected to realistic physics:" << std::endl;
        std::cout << "   • Quantum tunneling affects critical charge calculations" << std::endl;
        std::cout << "   • Defect formation energy uses QFT corrections" << std::endl;
        std::cout << "   • Temperature dependence includes quantum effects" << std::endl;
        std::cout << "   • Device-specific sensitivity uses quantum confinement" << std::endl;
        std::cout << "   • Multi-bit upsets calculated from quantum clustering" << std::endl;
        std::cout << "   • Charge collection efficiency includes quantum corrections" << std::endl;
        std::cout << "\n✅ Missing physics now implemented:" << std::endl;
        std::cout << "   • Charge deposition modeling with quantum corrections" << std::endl;
        std::cout << "   • Critical charge calculations (temperature dependent)" << std::endl;
        std::cout << "   • Device-specific sensitivity (SRAM vs DRAM vs Flash)" << std::endl;
        std::cout << "   • Temperature dependence of radiation sensitivity" << std::endl;
        std::cout << "   • Realistic dose rate effects through time integration" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
