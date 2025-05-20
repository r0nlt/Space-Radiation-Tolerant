/**
 * Quantum Models for Radiation Effects
 *
 * This file contains quantum models for radiation effects simulation.
 * It extends the core quantum field theory models.
 */

#pragma once

#include <Eigen/Dense>
#include <complex>
#include <map>
#include <memory>
#include <optional>
#include <rad_ml/physics/field_theory.hpp>
#include <rad_ml/physics/quantum_field_theory.hpp>
#include <string>
#include <variant>
#include <vector>

namespace rad_ml {
namespace physics {

// Using the types already defined in quantum_field_theory.hpp
// No need to redefine CrystalLattice, QFTParameters, or other classes

// Extended QFT parameters with additional fields
struct ExtendedQFTParameters : public QFTParameters {
    std::unordered_map<ParticleType, double> decoherence_rates;
    std::unordered_map<ParticleType, double> dissipation_coefficients;

    ExtendedQFTParameters() : QFTParameters()
    {
        // Initialize with default values for common particles
        decoherence_rates[ParticleType::Proton] = 1e-12;
        decoherence_rates[ParticleType::Electron] = 1e-12;
        decoherence_rates[ParticleType::Neutron] = 1e-12;
        decoherence_rates[ParticleType::Photon] = 1e-12;

        dissipation_coefficients[ParticleType::Proton] = 0.01;
        dissipation_coefficients[ParticleType::Electron] = 0.01;
        dissipation_coefficients[ParticleType::Neutron] = 0.01;
        dissipation_coefficients[ParticleType::Photon] = 0.01;
    }

    // For backward compatibility - get decoherence rate for a specific particle type
    double getDecoherenceRate(ParticleType type = ParticleType::Proton) const
    {
        auto it = decoherence_rates.find(type);
        return (it != decoherence_rates.end()) ? it->second : 1e-12;
    }

    // For backward compatibility - get dissipation coefficient for a specific particle type
    double getDissipationCoefficient(ParticleType type = ParticleType::Proton) const
    {
        auto it = dissipation_coefficients.find(type);
        return (it != dissipation_coefficients.end()) ? it->second : 0.01;
    }
};

// Additional quantum model utilities and extensions

/**
 * Calculate quantum decoherence effects on defect distribution
 *
 * @param defects Defect distribution
 * @param temperature Temperature in Kelvin
 * @param params Extended QFT parameters
 * @param particle_type Particle type to consider
 * @return Decoherence rate
 */
double calculateQuantumDecoherence(const DefectDistribution& defects, double temperature,
                                   const ExtendedQFTParameters& params,
                                   ParticleType particle_type = ParticleType::Proton);

/**
 * Calculate radiation-induced quantum transition probability
 *
 * @param incident_energy Incident radiation energy in eV
 * @param temperature Temperature in Kelvin
 * @param params QFT parameters
 * @param particle_type Particle type to consider
 * @return Transition probability
 */
double calculateQuantumTransitionProbability(double incident_energy, double temperature,
                                             const QFTParameters& params,
                                             ParticleType particle_type = ParticleType::Proton);

// Additional utility functions for quantum modeling of radiation effects

/**
 * Calculate displacement energy based on quantum effects
 *
 * @param crystal Crystal lattice structure
 * @param params DFT parameters
 * @param particle_type Particle type to consider
 * @return Displacement energy in eV
 */
double calculateDisplacementEnergy(const CrystalLattice& crystal, const QFTParameters& params,
                                   ParticleType particle_type = ParticleType::Proton);

/**
 * Simulate displacement cascade with quantum effects
 *
 * @param crystal Crystal structure
 * @param pka_energy Primary knock-on atom energy in eV
 * @param params QFT parameters
 * @param displacement_energy Displacement energy threshold
 * @param particle_type Particle type to consider
 * @return Resulting defect distribution
 */
DefectDistribution simulateDisplacementCascade(const CrystalLattice& crystal, double pka_energy,
                                               const QFTParameters& params,
                                               double displacement_energy,
                                               ParticleType particle_type = ParticleType::Proton);

/**
 * Factory methods for crystal lattice creation
 */
namespace CrystalLatticeFactory {
inline CrystalLattice FCC(double lattice_constant, double barrier_height = 1.0)
{
    return CrystalLattice(CrystalLattice::Type::FCC, lattice_constant, barrier_height);
}

inline CrystalLattice BCC(double lattice_constant, double barrier_height = 1.0)
{
    return CrystalLattice(CrystalLattice::Type::BCC, lattice_constant, barrier_height);
}

inline CrystalLattice Diamond(double lattice_constant, double barrier_height = 1.0)
{
    return CrystalLattice(CrystalLattice::Type::DIAMOND, lattice_constant, barrier_height);
}
}  // namespace CrystalLatticeFactory

/**
 * Create a multi-particle field with appropriate physics
 *
 * @param grid_dimensions Grid dimensions for the field
 * @param lattice_spacing Lattice spacing
 * @param particle_type Type of particle
 * @param params QFT parameters
 * @return Unique pointer to a quantum field
 */
std::unique_ptr<QuantumField<3>> createParticleField(const std::vector<int>& grid_dimensions,
                                                     double lattice_spacing,
                                                     ParticleType particle_type,
                                                     const QFTParameters& params);

/**
 * Simulate interaction between multiple particle fields
 *
 * @param fields Vector of quantum fields for different particles
 * @param params QFT parameters
 * @param steps Number of evolution steps
 * @return Changes in field energies
 */
std::vector<double> simulateMultiParticleInteraction(
    std::vector<std::reference_wrapper<QuantumField<3>>> fields, const QFTParameters& params,
    int steps);

}  // namespace physics
}  // namespace rad_ml
