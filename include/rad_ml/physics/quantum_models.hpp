/**
 * Quantum Models for Radiation Effects
 * 
 * This file contains quantum models for radiation effects simulation.
 * It extends the core quantum field theory models.
 */

#pragma once

#include <string>
#include <map>
#include <complex>
#include <vector>
#include <Eigen/Dense>
#include <rad_ml/physics/field_theory.hpp>
#include <rad_ml/physics/quantum_field_theory.hpp>

namespace rad_ml {
namespace physics {

// Using the types already defined in quantum_field_theory.hpp
// No need to redefine CrystalLattice, QFTParameters, or other classes

// Extended QFT parameters with additional fields
struct ExtendedQFTParameters : public QFTParameters {
    double decoherence_rate;
    double dissipation_coefficient;
    
    ExtendedQFTParameters() 
        : QFTParameters(),
          decoherence_rate(1e-12),
          dissipation_coefficient(0.01) {}
};

// Additional quantum model utilities and extensions

/**
 * Calculate quantum decoherence effects on defect distribution
 * @param defects Defect distribution 
 * @param temperature Temperature in Kelvin
 * @param params Extended QFT parameters
 * @return Decoherence rate
 */
double calculateQuantumDecoherence(
    const DefectDistribution& defects,
    double temperature,
    const ExtendedQFTParameters& params);

/**
 * Calculate radiation-induced quantum transition probability
 * @param incident_energy Incident radiation energy in eV
 * @param temperature Temperature in Kelvin
 * @param params QFT parameters
 * @return Transition probability
 */
double calculateQuantumTransitionProbability(
    double incident_energy,
    double temperature,
    const QFTParameters& params);

// Additional utility functions for quantum modeling of radiation effects

/**
 * Calculate displacement energy based on quantum effects
 * @param crystal Crystal lattice structure
 * @param params DFT parameters
 * @return Displacement energy in eV
 */
double calculateDisplacementEnergy(
    const CrystalLattice& crystal,
    const QFTParameters& params);

/**
 * Simulate displacement cascade with quantum effects
 * @param crystal Crystal structure
 * @param pka_energy Primary knock-on atom energy in eV
 * @param params QFT parameters
 * @param displacement_energy Displacement energy threshold
 * @return Resulting defect distribution
 */
DefectDistribution simulateDisplacementCascade(
    const CrystalLattice& crystal,
    double pka_energy,
    const QFTParameters& params,
    double displacement_energy);

/**
 * Factory methods for crystal lattice creation
 */
namespace CrystalLatticeFactory {
    inline CrystalLattice FCC(double lattice_constant, double barrier_height = 1.0) {
        return CrystalLattice(CrystalLattice::FCC_TYPE, lattice_constant, barrier_height);
    }

    inline CrystalLattice BCC(double lattice_constant, double barrier_height = 1.0) {
        return CrystalLattice(CrystalLattice::BCC, lattice_constant, barrier_height);
    }

    inline CrystalLattice Diamond(double lattice_constant, double barrier_height = 1.0) {
        return CrystalLattice(CrystalLattice::DIAMOND, lattice_constant, barrier_height);
    }
}

} // namespace physics
} // namespace rad_ml 