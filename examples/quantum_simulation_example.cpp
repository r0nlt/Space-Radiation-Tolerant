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
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>

// Simple struct definitions for testing - in real code, these would be in headers
struct CrystalLattice {
    enum Type { FCC_TYPE, BCC, DIAMOND };
    
    Type type;
    double lattice_constant;
    double barrier_height;
    
    CrystalLattice(Type t = DIAMOND, double lc = 5.43, double bh = 1.0) 
        : type(t), lattice_constant(lc), barrier_height(bh) {}
};

// Defect distribution map
using DefectDistribution = std::map<std::string, double>;

// QFT parameters for quantum field calculations
struct QFTParameters {
    double hbar;               // Reduced Planck constant (eV·s)
    double mass;               // Effective mass (kg)
    double coupling_constant;  // Coupling constant for interactions
    double potential_coefficient; // Potential energy coefficient
    double lattice_spacing;    // Lattice spacing (nm)
    double time_step;          // Simulation time step (s)
    int dimensions;            // Number of spatial dimensions
    
    QFTParameters() 
        : hbar(6.582119569e-16), 
          mass(1.0e-30),
          coupling_constant(0.1),
          potential_coefficient(0.5),
          lattice_spacing(1.0),
          time_step(1.0e-18),
          dimensions(3) {}
};

// Configuration for quantum corrections
struct QuantumCorrectionConfig {
    bool enable_quantum_corrections;        // Master switch for quantum corrections
    double temperature_threshold;           // Apply quantum corrections below this temperature (K)
    double feature_size_threshold;          // Apply quantum corrections below this feature size (nm)
    double radiation_intensity_threshold;   // Apply quantum corrections above this radiation level
    bool force_quantum_corrections;         // Force quantum corrections regardless of thresholds
};

// Forward declarations
double calculateQuantumTunnelingProbability(
    double barrier_height,
    double mass,
    double hbar,
    double temperature);

double solveKleinGordonEquation(
    double hbar,
    double mass,
    double potential_coeff,
    double coupling_constant,
    double lattice_spacing,
    double time_step);

double calculateZeroPointEnergyContribution(
    double hbar,
    double mass,
    double lattice_constant,
    double temperature);

DefectDistribution applyQuantumFieldCorrections(
    const DefectDistribution& defects,
    const CrystalLattice& crystal,
    const QFTParameters& qft_params,
    double temperature);

QFTParameters createQFTParameters(const CrystalLattice& crystal, double feature_size_nm = 100.0);

QuantumCorrectionConfig getDefaultQuantumConfig();

bool shouldApplyQuantumCorrections(
    double temperature,
    double feature_size,
    double radiation_intensity,
    const QuantumCorrectionConfig& config);

DefectDistribution applyQuantumCorrectionsToSimulation(
    const DefectDistribution& defects,
    const CrystalLattice& crystal,
    double temperature,
    double feature_size_nm,
    double radiation_intensity,
    const QuantumCorrectionConfig& config = QuantumCorrectionConfig());

void compareDistributions(
    const DefectDistribution& classical,
    const DefectDistribution& quantum);

// Simplified radiation simulation class
class RadiationSimulator {
private:
    // Random number generator
    std::mt19937 rng;
    std::normal_distribution<double> normal_dist;
    
    // Simulation parameters
    double temperature_K;
    double feature_size_nm;
    double radiation_intensity;
    double simulation_time_s;
    CrystalLattice material;
    
    // Quantum configuration
    QuantumCorrectionConfig quantum_config;
    bool use_quantum_corrections;
    
    // Results
    DefectDistribution classical_defects;
    DefectDistribution quantum_defects;
    double simulation_error;
    
public:
    // Constructor with default parameters
    RadiationSimulator(
        const CrystalLattice& mat = CrystalLattice(),
        double temp = 300.0,
        double size = 45.0,
        double rad_intensity = 1e5,
        double sim_time = 1.0)
        : material(mat),
          temperature_K(temp),
          feature_size_nm(size),
          radiation_intensity(rad_intensity),
          simulation_time_s(sim_time),
          use_quantum_corrections(true),
          simulation_error(0.0),
          normal_dist(0.0, 1.0)
    {
        // Initialize random number generator
        std::random_device rd;
        rng = std::mt19937(rd());
        
        // Initialize quantum configuration with default values
        quantum_config = getDefaultQuantumConfig();
    }
    
    // Run the simulation
    void runSimulation() {
        std::cout << "Running radiation simulation with:" << std::endl;
        std::cout << "  - Temperature: " << temperature_K << " K" << std::endl;
        std::cout << "  - Feature size: " << feature_size_nm << " nm" << std::endl;
        std::cout << "  - Radiation intensity: " << radiation_intensity << " rad/s" << std::endl;
        std::cout << "  - Simulation time: " << simulation_time_s << " s" << std::endl;
        
        // Run classical simulation
        runClassicalSimulation();
        
        // Apply quantum corrections if enabled
        if (use_quantum_corrections) {
            applyQuantumCorrections();
        }
        
        // Calculate error based on known experimental data
        calculateSimulationError();
    }
    
    // Enable or disable quantum corrections
    void enableQuantumCorrections(bool enable) {
        use_quantum_corrections = enable;
    }
    
    // Set temperature
    void setTemperature(double temp) {
        temperature_K = temp;
    }
    
    // Set feature size
    void setFeatureSize(double size) {
        feature_size_nm = size;
    }
    
    // Set simulation time
    void setSimulationTime(double time) {
        simulation_time_s = time;
    }
    
    // Get simulation results
    const DefectDistribution& getClassicalDefects() const {
        return classical_defects;
    }
    
    const DefectDistribution& getQuantumDefects() const {
        return use_quantum_corrections ? quantum_defects : classical_defects;
    }
    
    double getSimulationError() const {
        return simulation_error;
    }
    
    // Print simulation results
    void printResults() const {
        std::cout << "Simulation Results:" << std::endl;
        std::cout << std::string(30, '-') << std::endl;
        
        std::cout << "Classical Model:" << std::endl;
        for (const auto& defect : classical_defects) {
            std::cout << "  " << std::left << std::setw(15) << defect.first 
                     << std::fixed << std::setprecision(2) << defect.second << std::endl;
        }
        std::cout << std::endl;
        
        if (use_quantum_corrections) {
            std::cout << "Quantum-Corrected Model:" << std::endl;
            for (const auto& defect : quantum_defects) {
                std::cout << "  " << std::left << std::setw(15) << defect.first 
                         << std::fixed << std::setprecision(2) << defect.second << std::endl;
            }
            std::cout << std::endl;
            
            // Calculate improvement
            double total_classical = 0.0;
            double total_quantum = 0.0;
            
            for (const auto& defect : classical_defects) {
                total_classical += defect.second;
            }
            
            for (const auto& defect : quantum_defects) {
                total_quantum += defect.second;
            }
            
            double percent_diff = ((total_quantum - total_classical) / total_classical) * 100.0;
            std::cout << "Quantum correction effect: " << std::showpos << std::fixed 
                     << std::setprecision(2) << percent_diff << "%" << std::noshowpos << std::endl;
        }
        
        std::cout << "Simulation error: " << std::fixed << std::setprecision(2) 
                 << simulation_error * 100.0 << "%" << std::endl;
        std::cout << std::endl;
    }
    
    // Export results to CSV
    void exportResults(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return;
        }
        
        // Write header
        file << "Defect Type,Classical Model,Quantum Model,Difference (%)\n";
        
        // Write data
        for (const auto& defect : classical_defects) {
            const std::string& defect_type = defect.first;
            double classical_value = defect.second;
            double quantum_value = use_quantum_corrections ? 
                quantum_defects.at(defect_type) : classical_value;
            
            double percent_diff = ((quantum_value - classical_value) / classical_value) * 100.0;
            
            file << defect_type << ","
                << classical_value << ","
                << quantum_value << ","
                << percent_diff << "\n";
        }
        
        // Add simulation parameters
        file << "\nSimulation Parameters\n";
        file << "Temperature (K)," << temperature_K << "\n";
        file << "Feature Size (nm)," << feature_size_nm << "\n";
        file << "Radiation Intensity (rad/s)," << radiation_intensity << "\n";
        file << "Simulation Time (s)," << simulation_time_s << "\n";
        file << "Quantum Corrections," << (use_quantum_corrections ? "Enabled" : "Disabled") << "\n";
        file << "Simulation Error (%)," << simulation_error * 100.0 << "\n";
        
        file.close();
        std::cout << "Results exported to " << filename << std::endl;
    }
    
private:
    // Run classical radiation simulation
    void runClassicalSimulation() {
        // Clear previous results
        classical_defects.clear();
        
        // In a real implementation, this would be a complex physics-based simulation
        // Here we just use a simplified model for demonstration
        
        // Calculate base defect rates based on radiation intensity and time
        double total_dose = radiation_intensity * simulation_time_s;
        
        // Simple linear model with temperature dependence
        double temp_factor = std::exp(-temperature_K / 600.0); // Higher temp = fewer stable defects
        
        // Simulate different types of defects
        double vacancy_rate = 0.5 * total_dose * temp_factor;
        double interstitial_rate = 0.4 * total_dose * temp_factor;
        double complex_rate = 0.1 * total_dose * std::pow(temp_factor, 2.0); // More affected by temp
        double dislocation_rate = 0.05 * total_dose * (1.0 - temp_factor); // Less affected by temp
        
        // Add random variation
        vacancy_rate *= (1.0 + 0.1 * normal_dist(rng));
        interstitial_rate *= (1.0 + 0.1 * normal_dist(rng));
        complex_rate *= (1.0 + 0.2 * normal_dist(rng));
        dislocation_rate *= (1.0 + 0.1 * normal_dist(rng));
        
        // Store results
        classical_defects["vacancy"] = vacancy_rate;
        classical_defects["interstitial"] = interstitial_rate;
        classical_defects["complex"] = complex_rate;
        classical_defects["dislocation"] = dislocation_rate;
        
        std::cout << "Classical simulation completed." << std::endl;
    }
    
    // Apply quantum corrections to the classical results
    void applyQuantumCorrections() {
        // Apply quantum field theory corrections
        quantum_defects = applyQuantumCorrectionsToSimulation(
            classical_defects,
            material,
            temperature_K,
            feature_size_nm,
            radiation_intensity,
            quantum_config
        );
        
        std::cout << "Quantum corrections applied." << std::endl;
    }
    
    // Calculate simulation error based on "known" experimental data
    void calculateSimulationError() {
        // In a real application, this would compare against experimental data
        // For this example, we'll create synthetic "experimental data" with quantum effects built in
        
        DefectDistribution experimental_data;
        
        // Create experimental data that includes quantum effects
        // We'll assume quantum effects are more prominent at low temperatures
        double quantum_factor = 1.0;
        
        if (temperature_K < 150.0) {
            quantum_factor += 0.3 * (150.0 - temperature_K) / 150.0;
        }
        
        // Smaller feature sizes also increase quantum effects
        if (feature_size_nm < 20.0) {
            quantum_factor += 0.2 * (20.0 - feature_size_nm) / 20.0;
        }
        
        // Generate experimental data based on quantum effects
        for (const auto& defect : classical_defects) {
            std::string defect_type = defect.first;
            double rate = defect.second;
            
            // Different defect types have different quantum sensitivities
            double type_factor = 1.0;
            if (defect_type == "vacancy") type_factor = 0.8;
            else if (defect_type == "interstitial") type_factor = 1.2;
            else if (defect_type == "complex") type_factor = 1.5;
            
            // Add random experimental variation
            double exp_rate = rate * (1.0 + (quantum_factor - 1.0) * type_factor);
            exp_rate *= (1.0 + 0.05 * normal_dist(rng)); // 5% measurement error
            
            experimental_data[defect_type] = exp_rate;
        }
        
        // Compare with either classical or quantum-corrected results
        const DefectDistribution& simulation_data = use_quantum_corrections ? 
            quantum_defects : classical_defects;
        
        // Calculate mean squared error
        double error_sum = 0.0;
        double value_sum = 0.0;
        
        for (const auto& defect : experimental_data) {
            std::string defect_type = defect.first;
            double exp_value = defect.second;
            double sim_value = simulation_data.at(defect_type);
            
            error_sum += std::pow(exp_value - sim_value, 2.0);
            value_sum += std::pow(exp_value, 2.0);
        }
        
        // Normalize error
        simulation_error = std::sqrt(error_sum / value_sum);
        
        std::cout << "Simulation error calculation completed." << std::endl;
    }
};

// Implementations of quantum field theory functions
// These are simplified versions of what would be in the real implementation
double calculateQuantumTunnelingProbability(
    double barrier_height,
    double mass,
    double hbar,
    double temperature) {
    
    // WKB approximation for tunneling through a barrier
    const double barrier_width = 1.0; // nm
    const double kb = 8.617333262e-5; // Boltzmann constant in eV/K
    
    // Calculate thermal energy
    double thermal_energy = kb * temperature;
    
    // Basic tunneling probability
    double exponent = -2.0 * barrier_width * std::sqrt(2.0 * mass * barrier_height) / hbar;
    // Normalize to a reasonable range - real physical value is extremely small
    double base_probability = 0.01 * (1.0 - std::min(1.0, temperature / 300.0));
    
    // Temperature correction (higher temperature reduces tunneling significance)
    double temp_factor = std::exp(-thermal_energy / (2.0 * barrier_height));
    
    // Scale to a realistic enhancement range (0-10% at most)
    return 0.1 * base_probability * temp_factor;
}

double solveKleinGordonEquation(
    double hbar,
    double mass,
    double potential_coeff,
    double coupling_constant,
    double lattice_spacing,
    double time_step) {
    
    // Simplified Klein-Gordon equation solution
    // In a full implementation, this would involve solving the differential equation
    
    // For testing, return a small enhancement factor based on the feature size
    double size_factor = (1.0 / lattice_spacing) * 0.01;
    
    // Scale to a reasonable correction range (bigger effect at smaller sizes)
    return size_factor; // 0-5% correction based on size
}

double calculateZeroPointEnergyContribution(
    double hbar,
    double mass,
    double lattice_constant,
    double temperature) {
    
    // Simple harmonic oscillator zero-point energy: E₀ = ħω/2
    // Temperature scaling (ZPE becomes more significant at lower temperatures)
    const double kb = 8.617333262e-5; // Boltzmann constant in eV/K
    
    // ZPE significance increases at low temperature
    double zpe_significance = 0.005 * (300.0 / std::max(temperature, 10.0));
    
    // Cap at reasonable values (0.5-5% effect)
    return std::min(0.05, zpe_significance);
}

DefectDistribution applyQuantumFieldCorrections(
    const DefectDistribution& defects,
    const CrystalLattice& crystal,
    const QFTParameters& qft_params,
    double temperature) {
    
    // Start with a copy of the original defects
    DefectDistribution corrected_defects = defects;
    
    // Calculate quantum corrections
    double tunneling_probability = calculateQuantumTunnelingProbability(
        crystal.barrier_height,
        qft_params.mass,
        qft_params.hbar,
        temperature
    );
    
    // Apply Klein-Gordon correction factor
    double kg_correction = solveKleinGordonEquation(
        qft_params.hbar,
        qft_params.mass,
        qft_params.potential_coefficient,
        qft_params.coupling_constant,
        qft_params.lattice_spacing,
        qft_params.time_step
    );
    
    // Calculate zero-point energy contribution
    double zpe_contribution = calculateZeroPointEnergyContribution(
        qft_params.hbar,
        qft_params.mass,
        crystal.lattice_constant,
        temperature
    );
    
    // Apply corrections to each defect type
    for (auto& defect_pair : corrected_defects) {
        std::string defect_type = defect_pair.first;
        double& defect_count = defect_pair.second;
        
        // Apply different correction factors based on defect type
        if (defect_type == "vacancy") {
            // Vacancies are less affected by tunneling
            defect_count *= (1.0 + 0.5 * tunneling_probability + 0.7 * kg_correction);
        } else if (defect_type == "interstitial") {
            // Interstitials are strongly affected by tunneling
            defect_count *= (1.0 + 1.5 * tunneling_probability + 0.9 * kg_correction);
        } else if (defect_type == "complex") {
            // Complex defects show intermediate behavior
            defect_count *= (1.0 + tunneling_probability + kg_correction);
        } else {
            // Default correction
            defect_count *= (1.0 + 0.8 * tunneling_probability + 0.8 * kg_correction);
        }
        
        // Add zero-point energy contribution
        defect_count *= (1.0 + zpe_contribution);
    }
    
    // Log the correction factors (in a real implementation, this would use proper logging)
    std::cout << "Applied quantum corrections with factors: " << std::endl;
    std::cout << "  - Tunneling probability: " << tunneling_probability << std::endl;
    std::cout << "  - Klein-Gordon correction: " << kg_correction << std::endl;
    std::cout << "  - Zero-point energy contribution: " << zpe_contribution << std::endl;
    
    return corrected_defects;
}

QFTParameters createQFTParameters(const CrystalLattice& crystal, double feature_size_nm) {
    QFTParameters params;
    
    // Physics constants
    params.hbar = 6.582119569e-16;  // reduced Planck constant (eV·s)
    
    // Material-specific parameters
    // Mass depends on the material type
    switch (crystal.type) {
        case CrystalLattice::FCC_TYPE:
            params.mass = 1.0e-30; // Default FCC mass value
            break;
        case CrystalLattice::BCC:
            params.mass = 1.1e-30; // BCC materials have slightly different effective mass
            break;
        case CrystalLattice::DIAMOND:
            params.mass = 0.9e-30; // Diamond lattice materials
            break;
        default:
            params.mass = 1.0e-30; // Default value
    }
    
    // Scaling parameters based on material properties
    params.coupling_constant = 0.1 * (crystal.lattice_constant / 5.0); // Scale with lattice constant
    params.potential_coefficient = 0.5;
    
    // Feature size impacts lattice spacing parameter
    params.lattice_spacing = feature_size_nm / 100.0; // Convert to appropriate scale
    
    // Simulation parameters
    params.time_step = 1.0e-18; // fs time scale
    params.dimensions = 3;
    
    return params;
}

QuantumCorrectionConfig getDefaultQuantumConfig() {
    QuantumCorrectionConfig config;
    config.enable_quantum_corrections = true;
    config.temperature_threshold = 150.0;      // K - quantum effects start becoming significant
    config.feature_size_threshold = 20.0;      // nm - quantum effects important at smaller scales
    config.radiation_intensity_threshold = 1e5; // rad/s - high radiation levels
    config.force_quantum_corrections = false;
    return config;
}

bool shouldApplyQuantumCorrections(
    double temperature,
    double feature_size,
    double radiation_intensity,
    const QuantumCorrectionConfig& config) {
    
    if (!config.enable_quantum_corrections) {
        return false;
    }
    
    if (config.force_quantum_corrections) {
        return true;
    }
    
    // Apply based on thresholds
    bool temperature_criterion = temperature < config.temperature_threshold;
    bool feature_size_criterion = feature_size < config.feature_size_threshold;
    bool radiation_criterion = radiation_intensity > config.radiation_intensity_threshold;
    
    // Apply corrections if any criterion is met
    return temperature_criterion || feature_size_criterion || radiation_criterion;
}

DefectDistribution applyQuantumCorrectionsToSimulation(
    const DefectDistribution& defects,
    const CrystalLattice& crystal,
    double temperature,
    double feature_size_nm,
    double radiation_intensity,
    const QuantumCorrectionConfig& config) {
    
    // Check if we should apply quantum corrections
    if (!shouldApplyQuantumCorrections(temperature, feature_size_nm, radiation_intensity, config)) {
        std::cout << "Quantum corrections will NOT be applied (thresholds not met)" << std::endl;
        return defects; // Return original defects without quantum corrections
    }
    
    std::cout << "Quantum corrections will be applied" << std::endl;
    
    // Create QFT parameters based on material properties
    QFTParameters qft_params = createQFTParameters(crystal, feature_size_nm);
    
    // Apply quantum field corrections
    DefectDistribution corrected_defects = applyQuantumFieldCorrections(
        defects, crystal, qft_params, temperature);
    
    return corrected_defects;
}

// Main function with examples
int main() {
    std::cout << "=== Quantum Field Theory Integration Example ===" << std::endl;
    std::cout << std::string(50, '=') << std::endl << std::endl;
    
    // Define materials for testing
    CrystalLattice silicon(CrystalLattice::DIAMOND, 5.431, 1.1);
    CrystalLattice germanium(CrystalLattice::DIAMOND, 5.658, 0.67);
    CrystalLattice gaas(CrystalLattice::DIAMOND, 5.653, 0.84);
    
    // Example 1: Room temperature silicon (no quantum effects expected)
    std::cout << "Example 1: Room temperature silicon (300K)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    // Create simulator with default settings
    RadiationSimulator simulator(silicon, 300.0, 45.0, 1e5, 1.0);
    
    // Run with and without quantum corrections
    std::cout << "Running with quantum corrections:" << std::endl;
    simulator.enableQuantumCorrections(true);
    simulator.runSimulation();
    simulator.printResults();
    
    std::cout << "Running without quantum corrections:" << std::endl;
    simulator.enableQuantumCorrections(false);
    simulator.runSimulation();
    simulator.printResults();
    
    // Example 2: Low temperature silicon (quantum effects should be visible)
    std::cout << "Example 2: Low temperature silicon (77K)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    simulator.setTemperature(77.0);
    
    std::cout << "Running with quantum corrections:" << std::endl;
    simulator.enableQuantumCorrections(true);
    simulator.runSimulation();
    simulator.printResults();
    
    std::cout << "Running without quantum corrections:" << std::endl;
    simulator.enableQuantumCorrections(false);
    simulator.runSimulation();
    simulator.printResults();
    
    // Example 3: Nanoscale device (quantum effects should be visible)
    std::cout << "Example 3: Silicon nanoscale device (10nm)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    simulator.setTemperature(300.0);
    simulator.setFeatureSize(10.0);
    
    std::cout << "Running with quantum corrections:" << std::endl;
    simulator.enableQuantumCorrections(true);
    simulator.runSimulation();
    simulator.printResults();
    
    std::cout << "Running without quantum corrections:" << std::endl;
    simulator.enableQuantumCorrections(false);
    simulator.runSimulation();
    simulator.printResults();
    
    // Example 4: Extreme conditions (both low temperature and small size)
    std::cout << "Example 4: Extreme conditions (4.2K, 5nm)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    simulator.setTemperature(4.2);
    simulator.setFeatureSize(5.0);
    
    std::cout << "Running with quantum corrections:" << std::endl;
    simulator.enableQuantumCorrections(true);
    simulator.runSimulation();
    simulator.printResults();
    
    std::cout << "Running without quantum corrections:" << std::endl;
    simulator.enableQuantumCorrections(false);
    simulator.runSimulation();
    simulator.printResults();
    
    // Export results to CSV
    simulator.exportResults("extreme_conditions_results.csv");
    
    std::cout << "Example completed successfully." << std::endl;
    
    return 0;
} 
