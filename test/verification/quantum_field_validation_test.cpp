#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
// Include our headers
#include <rad_ml/physics/field_theory.hpp>
#include <rad_ml/physics/quantum_field_theory.hpp>
#include <rad_ml/physics/quantum_models.hpp>

using namespace rad_ml::physics;

// Define material test cases
struct MaterialTestCase {
    std::string name;
    CrystalLattice lattice;
    double temperature;
    double radiation_dose;
};

// Define test scenarios
struct TestScenario {
    std::string name;
    double pka_energy;
    QFTParameters qft_params;
};

// Compute performance metrics
struct PerformanceMetrics {
    double classical_total_defects;
    double quantum_total_defects;
    double percent_difference;
    double tunneling_contribution;
    double zero_point_contribution;
    double execution_time_ms;
};

// Note: Using validated simulateDisplacementCascade() function from quantum_models.hpp
// This replaces the old local implementation with proper physics-based calculation

// Validate displacement energy against expected ranges
bool validateDisplacementEnergy(const std::string& material_name, CrystalLattice::Type crystal_type,
                                double displacement_energy)
{
    // Expected ranges based on validated crystal lattice properties
    double min_expected = 0.0, max_expected = 0.0;

    switch (crystal_type) {
        case CrystalLattice::Type::FCC:
            min_expected = 25.0;  // eV
            max_expected = 40.0;  // eV
            break;
        case CrystalLattice::Type::BCC:
            min_expected = 40.0;  // eV
            max_expected = 90.0;  // eV
            break;
        case CrystalLattice::Type::DIAMOND:
            min_expected = 30.0;  // eV
            max_expected = 40.0;  // eV (for Si, Ge, GaAs)
            break;
    }

    bool is_valid = (displacement_energy >= min_expected && displacement_energy <= max_expected);

    std::cout << "    Displacement energy: " << displacement_energy << " eV ";
    std::cout << "(expected: " << min_expected << "-" << max_expected << " eV) ";
    std::cout << (is_valid ? "✓ VALID" : "✗ INVALID") << std::endl;

    return is_valid;
}

// Run test for a single material and scenario
PerformanceMetrics runTest(const MaterialTestCase& material, const TestScenario& scenario)
{
    PerformanceMetrics metrics;

    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Calculate displacement energy using validated physics function
    // Default to Proton for backward compatibility, but should be parameterized in real usage
    double displacement_energy =
        calculateDisplacementEnergy(material.lattice, scenario.qft_params, ParticleType::Proton);

    // Validate displacement energy against expected ranges
    validateDisplacementEnergy(material.name, material.lattice.type, displacement_energy);

    // Simulate displacement cascade using classical model
    DefectDistribution classical_defects =
        simulateDisplacementCascade(material.lattice, scenario.pka_energy, scenario.qft_params,
                                    displacement_energy, ParticleType::Proton);

    // Count total classical defects
    metrics.classical_total_defects = 0.0;
    for (const auto& [particleType, values] : classical_defects.interstitials) {
        for (const auto& val : values) {
            metrics.classical_total_defects += val;
        }
    }
    for (const auto& [particleType, values] : classical_defects.vacancies) {
        for (const auto& val : values) {
            metrics.classical_total_defects += val;
        }
    }
    for (const auto& [particleType, values] : classical_defects.clusters) {
        for (const auto& val : values) {
            metrics.classical_total_defects += val;
        }
    }

    // Apply quantum corrections
    DefectDistribution quantum_defects = applyQuantumFieldCorrections(
        classical_defects, material.lattice, scenario.qft_params, material.temperature);

    // Count total quantum-corrected defects
    metrics.quantum_total_defects = 0.0;
    for (const auto& [particleType, values] : quantum_defects.interstitials) {
        for (const auto& val : values) {
            metrics.quantum_total_defects += val;
        }
    }
    for (const auto& [particleType, values] : quantum_defects.vacancies) {
        for (const auto& val : values) {
            metrics.quantum_total_defects += val;
        }
    }
    for (const auto& [particleType, values] : quantum_defects.clusters) {
        for (const auto& val : values) {
            metrics.quantum_total_defects += val;
        }
    }

    // Calculate percentage difference
    if (metrics.classical_total_defects > 0.0) {
        metrics.percent_difference =
            (metrics.quantum_total_defects - metrics.classical_total_defects) /
            metrics.classical_total_defects * 100.0;
    }
    else {
        // Avoid division by zero
        metrics.percent_difference = metrics.quantum_total_defects > 0 ? 100.0 : 0.0;
    }

    // Estimate tunneling contribution (simplified calculation)
    double formation_energy = 4.0;  // typical value in eV
    metrics.tunneling_contribution =
        calculateQuantumTunnelingProbability(formation_energy, material.temperature,
                                             scenario.qft_params) *
        100.0;

    // Estimate zero-point energy contribution (simplified calculation)
    double classical_energy = formation_energy;
    double quantum_energy = calculateQuantumCorrectedDefectEnergy(
        material.temperature, formation_energy, scenario.qft_params);

    // Safe calculation of zero-point contribution
    if (std::abs(classical_energy) > 1e-10) {
        metrics.zero_point_contribution =
            (quantum_energy - classical_energy) / classical_energy * 100.0;
    }
    else {
        metrics.zero_point_contribution = 0.0;
    }

    // Record end time and calculate execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    metrics.execution_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    return metrics;
}

int main()
{
    std::cout << "Quantum Field Theory Framework Enhancement Validation Test" << std::endl;
    std::cout << "=======================================================" << std::endl;

    // Define materials to test with CORRECT crystal structures
    std::vector<MaterialTestCase> materials = {
        {"Silicon", CrystalLattice(CrystalLattice::Type::DIAMOND, 5.431), 300.0,
         1e3},  // Silicon is Diamond structure
        {"Germanium", CrystalLattice(CrystalLattice::Type::DIAMOND, 5.658), 300.0,
         1e3},  // Germanium is Diamond structure
        {"GaAs", CrystalLattice(CrystalLattice::Type::DIAMOND, 5.653), 300.0,
         1e3},  // GaAs is zinc blende (Diamond-like)
        {"Silicon (Low Temp)", CrystalLattice(CrystalLattice::Type::DIAMOND, 5.431), 77.0, 1e3},
        {"Silicon (High Temp)", CrystalLattice(CrystalLattice::Type::DIAMOND, 5.431), 500.0, 1e3},
        {"Aluminum (FCC)", CrystalLattice(CrystalLattice::Type::FCC, 4.05), 300.0,
         1e3},  // Test actual FCC material
        {"Iron (BCC)", CrystalLattice(CrystalLattice::Type::BCC, 2.867), 300.0,
         1e3}};  // Test actual BCC material

    // Define test scenarios
    std::vector<TestScenario> scenarios;

    // Scenario 1: Standard conditions
    TestScenario standard;
    standard.name = "Standard";
    standard.pka_energy = 1000.0;  // 1 keV
    standard.qft_params.hbar = 6.582119569e-16;

    // Set particle-specific properties for all relevant particle types
    standard.qft_params.masses[ParticleType::Proton] = 1.67262192369e-27;
    standard.qft_params.masses[ParticleType::Electron] = 9.1093837015e-31;
    standard.qft_params.masses[ParticleType::Neutron] = 1.67492749804e-27;
    standard.qft_params.masses[ParticleType::Photon] = 0.0;  // Massless

    standard.qft_params.coupling_constants[ParticleType::Proton] = 0.1;
    standard.qft_params.coupling_constants[ParticleType::Electron] = 0.15;
    standard.qft_params.coupling_constants[ParticleType::Neutron] = 0.08;
    standard.qft_params.coupling_constants[ParticleType::Photon] = 0.05;

    standard.qft_params.potential_coefficient = 0.5;
    standard.qft_params.lattice_spacing = 0.1;
    standard.qft_params.time_step = 1.0e-18;
    standard.qft_params.dimensions = 3;
    scenarios.push_back(standard);

    // Scenario 2: High energy radiation
    TestScenario high_energy = standard;
    high_energy.name = "High Energy";
    high_energy.pka_energy = 10000.0;  // 10 keV
    scenarios.push_back(high_energy);

    // Scenario 3: Quantum-dominant regime
    TestScenario quantum_dominant = standard;
    quantum_dominant.name = "Quantum Dominant";
    quantum_dominant.qft_params.hbar = 6.582119569e-16 * 10;  // Exaggerated for testing
    scenarios.push_back(quantum_dominant);

    // Prepare results file
    std::ofstream results_file("quantum_enhancement_results.csv");
    results_file << "Material,Scenario,Classical Defects,Quantum Defects,Percent Difference,"
                 << "Tunneling Contribution (%),Zero-Point Contribution (%),Execution Time (ms)"
                 << std::endl;

    // Run tests for all materials and scenarios
    for (const auto& material : materials) {
        std::cout << "\nTesting material: " << material.name << std::endl;

        for (const auto& scenario : scenarios) {
            std::cout << "  Scenario: " << scenario.name << "... ";

            PerformanceMetrics metrics = runTest(material, scenario);

            // Write results to file
            results_file << material.name << "," << scenario.name << ","
                         << metrics.classical_total_defects << "," << metrics.quantum_total_defects
                         << "," << metrics.percent_difference << ","
                         << metrics.tunneling_contribution << "," << metrics.zero_point_contribution
                         << "," << metrics.execution_time_ms << std::endl;

            // Print summary
            std::cout << "Complete. Defect difference: " << std::fixed << std::setprecision(2)
                      << metrics.percent_difference << "%" << std::endl;
        }
    }

    results_file.close();

    std::cout << "\nQuantum enhancement validation test completed." << std::endl;
    std::cout << "Results saved to quantum_enhancement_results.csv" << std::endl;

    return 0;
}
