/**
 * Implementation of Quantum Models
 * 
 * This file implements functions declared in quantum_models.hpp
 */

#include <rad_ml/physics/quantum_models.hpp>
#include <rad_ml/physics/quantum_field_theory.hpp>
#include <cmath>
#include <random>

namespace rad_ml {
namespace physics {

double calculateQuantumDecoherence(
    const DefectDistribution& defects,
    double temperature,
    const ExtendedQFTParameters& params) {
    
    // Simple decoherence model based on temperature and defect concentration
    double total_defects = 0.0;
    for (auto val : defects.interstitials) total_defects += val;
    for (auto val : defects.vacancies) total_defects += val;
    for (auto val : defects.clusters) total_defects += val;
    
    // Decoherence rate increases with temperature and defect concentration
    return params.decoherence_rate * (1.0 + temperature / 300.0) * 
           (1.0 + total_defects * params.dissipation_coefficient);
}

double calculateQuantumTransitionProbability(
    double incident_energy,
    double temperature,
    const QFTParameters& params) {
    
    // Simplified model for quantum transition probability
    // Higher probabilities at low temperatures and high incident energies
    const double kB = 8.617333262e-5; // Boltzmann constant in eV/K
    double thermal_energy = kB * temperature;
    
    // Calculate transition probability using quantum mechanics principles
    double transition_prob = 1.0 - std::exp(-incident_energy / (thermal_energy + params.hbar * 1e15));
    
    // Bound the result to [0, 1]
    return std::max(0.0, std::min(1.0, transition_prob));
}

double calculateDisplacementEnergy(
    const CrystalLattice& crystal,
    const QFTParameters& params) {
    
    // Base displacement energy depends on lattice type
    double base_energy = 0.0;
    switch (crystal.type) {
        case CrystalLattice::FCC_TYPE:
            base_energy = 15.0 + 2.5 * crystal.lattice_constant; // eV
            break;
        case CrystalLattice::BCC:
            base_energy = 10.0 + 3.0 * crystal.lattice_constant; // eV
            break;
        case CrystalLattice::DIAMOND:
            base_energy = 20.0 + 4.0 * crystal.lattice_constant; // eV
            break;
        default:
            base_energy = 25.0; // Default value
    }
    
    // Apply quantum correction
    double quantum_correction = calculateZeroPointEnergyContribution(
        params.hbar, params.mass, crystal.lattice_constant, 300.0);
    
    return base_energy + quantum_correction;
}

DefectDistribution simulateDisplacementCascade(
    const CrystalLattice& crystal,
    double pka_energy,
    const QFTParameters& params,
    double displacement_energy) {
    
    // Initialize defect distribution
    DefectDistribution defects;
    
    // Simple model for defect production:
    // Number of defects scales with PKA energy and inversely with displacement energy
    if (pka_energy > displacement_energy) {
        double defect_count = std::floor(0.8 * pka_energy / displacement_energy);
        
        // Distribute defects among different types
        // Spatial distribution follows cascade morphology
        double vacancy_fraction = 0.6;
        double interstitial_fraction = 0.3;
        double cluster_fraction = 0.1;
        
        // Fill in the defects vectors with appropriate distribution
        defects.interstitials.clear();
        defects.vacancies.clear();
        defects.clusters.clear();
        
        // Region 1 (core)
        defects.vacancies.push_back(defect_count * vacancy_fraction * 0.6);
        defects.interstitials.push_back(defect_count * interstitial_fraction * 0.4);
        defects.clusters.push_back(defect_count * cluster_fraction * 0.7);
        
        // Region 2 (intermediate)
        defects.vacancies.push_back(defect_count * vacancy_fraction * 0.3);
        defects.interstitials.push_back(defect_count * interstitial_fraction * 0.4);
        defects.clusters.push_back(defect_count * cluster_fraction * 0.2);
        
        // Region 3 (periphery)
        defects.vacancies.push_back(defect_count * vacancy_fraction * 0.1);
        defects.interstitials.push_back(defect_count * interstitial_fraction * 0.2);
        defects.clusters.push_back(defect_count * cluster_fraction * 0.1);
    }
    
    return defects;
}

} // namespace physics
} // namespace rad_ml 