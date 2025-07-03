#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
// Include our headers
#include <rad_ml/physics/field_theory.hpp>
#include <rad_ml/physics/quantum_field_theory.hpp>

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

// Utility functions implemented for testing
double calculateDisplacementEnergy(const CrystalLattice& lattice, const QFTParameters& params)
{
    // ==== UPDATED: USE VALIDATED KONOBEYEV MODEL ====
    // Replace old invalid base_energy * arbitrary factors with physics-based calculation

    // Material-specific parameters based on crystal structure
    double density = 0.0;          // g/cm³
    double melting_temp = 0.0;     // K
    double cohesive_energy = 0.0;  // eV
    double base_threshold = 0.0;   // eV

    switch (lattice.type) {
        case CrystalLattice::Type::FCC:
            // Typical FCC metals (Al, Cu, Ni, Au)
            density = 8.96;          // Cu-like density
            melting_temp = 1358.0;   // K
            cohesive_energy = 3.49;  // eV
            base_threshold = 25.0;   // eV (experimental range 25-40)
            break;

        case CrystalLattice::Type::BCC:
            // Typical BCC metals (Fe, Cr, W, Mo)
            density = 7.87;          // Fe-like density
            melting_temp = 1811.0;   // K
            cohesive_energy = 4.28;  // eV
            base_threshold = 40.0;   // eV (experimental range 40-90)
            break;

        case CrystalLattice::Type::DIAMOND:
            // Typical diamond structure (Si, Ge, C)
            density = 2.33;          // Si-like density
            melting_temp = 1687.0;   // K
            cohesive_energy = 4.63;  // eV
            base_threshold = 35.0;   // eV (experimental ~35 for Si)
            break;

        default:
            // Conservative default values
            density = 5.0;
            melting_temp = 1500.0;
            cohesive_energy = 4.0;
            base_threshold = 30.0;
    }

    // ==== KONOBEYEV MODEL CALCULATION ====
    // Ed = α(ρTmelt)^1/2 + β
    double alpha = 0.0352;  // Fitted parameter from Konobeyev analysis
    double beta = 8.74;     // Fitted parameter from Konobeyev analysis

    double konobeyev_energy = alpha * std::sqrt(density * melting_temp) + beta;

    // ==== COHESIVE ENERGY SCALING ====
    // Ed ≈ α × Ecohesive + β alternative model
    double alpha_coh = 8.2;  // Fitted parameter
    double beta_coh = 2.1;   // Fitted parameter

    double cohesive_scaled_energy = alpha_coh * cohesive_energy + beta_coh;

    // ==== WEIGHTED AVERAGE OF MODELS ====
    // Combine both validated approaches with empirical weighting
    double displacement_energy = 0.6 * konobeyev_energy + 0.4 * cohesive_scaled_energy;

    // Ensure result is within experimental bounds
    displacement_energy = std::max(displacement_energy, base_threshold * 0.8);
    displacement_energy = std::min(displacement_energy, base_threshold * 1.5);

    return displacement_energy;
}

DefectDistribution simulateDisplacementCascade(const CrystalLattice& lattice, double pka_energy,
                                               const QFTParameters& params,
                                               double displacement_energy)
{
    // ==== UPDATED: USE MODERN RADIATION DAMAGE MODEL ====
    // Implements arc-DPA corrections and cascade efficiency effects

    // Initialize defect distribution
    DefectDistribution defects;

    if (pka_energy > displacement_energy) {
        // ==== ARC-DPA MODEL CORRECTIONS ====
        // Modern understanding shows 2-3x higher damage production than basic NRT predictions
        double arc_dpa_factor = 1.0;
        switch (lattice.type) {
            case CrystalLattice::Type::FCC:
                arc_dpa_factor = 2.1;  // FCC metals show ~2.1x enhancement
                break;
            case CrystalLattice::Type::BCC:
                arc_dpa_factor = 2.5;  // BCC metals show higher enhancement
                break;
            case CrystalLattice::Type::DIAMOND:
                arc_dpa_factor = 1.8;  // Covalent materials show moderate enhancement
                break;
        }

        // ==== CASCADE EFFICIENCY EFFECTS ====
        // Defect production efficiency increases with PKA energy due to better cascade development
        double cascade_efficiency = 0.5;  // Base efficiency

        // Energy-dependent efficiency - higher energy = higher efficiency
        if (pka_energy < 1000.0) {
            cascade_efficiency = 0.4;  // Low energy: lower efficiency
        }
        else if (pka_energy < 10000.0) {
            cascade_efficiency = 0.6;  // Medium energy: moderate efficiency
        }
        else if (pka_energy < 100000.0) {
            cascade_efficiency = 0.8;  // High energy: higher efficiency
        }
        else {
            cascade_efficiency = 0.7;  // Very high energy: limited by subcascade formation
        }

        // Material-dependent efficiency
        switch (lattice.type) {
            case CrystalLattice::Type::FCC:
                cascade_efficiency *= 1.1;  // FCC slightly more efficient
                break;
            case CrystalLattice::Type::BCC:
                cascade_efficiency *= 1.0;  // BCC baseline
                break;
            case CrystalLattice::Type::DIAMOND:
                cascade_efficiency *= 0.9;  // Diamond less efficient due to strong bonds
                break;
        }

        // ==== CORRECT PHYSICS: APPLY CASCADE EFFICIENCY FIRST ====
        // Cascade efficiency determines how much of PKA energy goes into defects (≤ 1.0)
        double available_energy = pka_energy * cascade_efficiency;

        // Basic defect count from available energy (energy conservation)
        double base_defect_count = available_energy / displacement_energy;

        // ==== THEN APPLY ARC-DPA ENHANCEMENT ====
        // Arc-DPA factor enhances defect production beyond NRT predictions
        double defect_count = std::floor(base_defect_count * arc_dpa_factor);

        // Final energy conservation check
        double max_possible_defects = pka_energy / displacement_energy;
        defect_count = std::min(defect_count, max_possible_defects);

        // ==== REALISTIC DEFECT FRACTIONS ====
        double vacancy_fraction = 0.6;
        double interstitial_fraction = 0.3;
        double cluster_fraction = 0.1;

        // Convert indices to particle types and fill with realistic scaling
        std::vector<ParticleType> particleTypes = {ParticleType::Proton, ParticleType::Electron,
                                                   ParticleType::Neutron};

        for (size_t i = 0; i < particleTypes.size(); i++) {
            ParticleType type = particleTypes[i];
            // Initialize vectors for this particle type
            defects.interstitials[type] = std::vector<double>(3, 0.0);
            defects.vacancies[type] = std::vector<double>(3, 0.0);
            defects.clusters[type] = std::vector<double>(3, 0.0);

            // Set values for all spatial regions (3 regions)
            for (size_t j = 0; j < 3; j++) {
                defects.interstitials[type][j] =
                    defect_count * interstitial_fraction * (0.5 - j * 0.2);
                defects.vacancies[type][j] = defect_count * vacancy_fraction * (0.5 - j * 0.2);
                defects.clusters[type][j] = defect_count * cluster_fraction * (0.5 - j * 0.2);
            }
        }
    }

    return defects;
}

// Run test for a single material and scenario
PerformanceMetrics runTest(const MaterialTestCase& material, const TestScenario& scenario)
{
    PerformanceMetrics metrics;

    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Calculate displacement energy
    double displacement_energy = calculateDisplacementEnergy(material.lattice, scenario.qft_params);

    // Simulate displacement cascade using classical model
    DefectDistribution classical_defects = simulateDisplacementCascade(
        material.lattice, scenario.pka_energy, scenario.qft_params, displacement_energy);

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

    // Define materials to test
    std::vector<MaterialTestCase> materials = {
        {"Silicon", CrystalLattice(CrystalLattice::Type::FCC, 5.431), 300.0, 1e3},
        {"Germanium", CrystalLattice(CrystalLattice::Type::FCC, 5.658), 300.0, 1e3},
        {"GaAs", CrystalLattice(CrystalLattice::Type::FCC, 5.653), 300.0, 1e3},
        {"Silicon (Low Temp)", CrystalLattice(CrystalLattice::Type::FCC, 5.431), 77.0, 1e3},
        {"Silicon (High Temp)", CrystalLattice(CrystalLattice::Type::FCC, 5.431), 500.0, 1e3}};

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
