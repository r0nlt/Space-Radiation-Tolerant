/**
 * Radiation Transport Equation Models
 *
 * This file contains models for solving the Boltzmann transport equation
 * to simulate radiation transport through materials.
 */

#pragma once

#include <Eigen/Dense>
#include <rad_ml/physics/quantum_field_theory.hpp>

namespace rad_ml {
namespace physics {

// Use the ParticleType from quantum_field_theory.hpp

/**
 * Cross-section data for materials
 */
struct CrossSectionData {
    double total;        // Total cross-section
    double elastic;      // Elastic scattering cross-section
    double inelastic;    // Inelastic scattering cross-section
    double z_effective;  // Effective atomic number
};

/**
 * Solution to the transport equation
 */
struct TransportSolution {
    Eigen::Tensor<double, 3> fluence;  // Fluence tensor Φ(x,Ω,E)
    double convergence_error;          // Convergence error
};

/**
 * Setup radiation source based on environment
 *
 * @param env The radiation environment parameters
 * @param spatial_points Number of spatial points in the grid
 * @param angular_points Number of angular points in the grid
 * @param energy_bins Number of energy bins
 * @param particle_type Type of particle to simulate
 * @return 3D tensor of radiation source
 */
Eigen::Tensor<double, 3> setupRadiationSource(const struct RadiationEnvironment& env,
                                              int spatial_points, int angular_points,
                                              int energy_bins,
                                              ParticleType particle_type = ParticleType::Proton);

/**
 * Generate material-specific cross-section tensors
 *
 * @param material Properties of the material
 * @param energy_bins Number of energy bins
 * @param particle_type Type of particle to simulate
 * @return 2D tensor of cross sections
 */
Eigen::Tensor<double, 2> generateMaterialCrossSections(
    const struct MaterialProperties& material, int energy_bins,
    ParticleType particle_type = ParticleType::Proton);

/**
 * Generate material-specific scattering cross-section tensors
 *
 * @param material Properties of the material
 * @param angular_points Number of angular points in the grid
 * @param energy_bins Number of energy bins
 * @param particle_type Type of particle to simulate
 * @return 4D tensor of scattering cross sections
 */
Eigen::Tensor<double, 4> generateScatteringCrossSections(
    const struct MaterialProperties& material, int angular_points, int energy_bins,
    ParticleType particle_type = ParticleType::Proton);

/**
 * Solve the Boltzmann transport equation
 *
 * @param initial_fluence Initial fluence distribution
 * @param source Radiation source term
 * @param sigma_t Total cross-section tensor
 * @param sigma_s Scattering cross-section tensor
 * @return Solution to the transport equation
 */
TransportSolution solveTransportEquation(const Eigen::Tensor<double, 3>& initial_fluence,
                                         const Eigen::Tensor<double, 3>& source,
                                         const Eigen::Tensor<double, 2>& sigma_t,
                                         const Eigen::Tensor<double, 4>& sigma_s);

/**
 * Calculate dose distribution from fluence
 *
 * @param fluence Fluence tensor
 * @param material_density Density of the material
 * @return 1D tensor of dose distribution
 */
Eigen::Tensor<double, 1> calculateDoseDistribution(const Eigen::Tensor<double, 3>& fluence,
                                                   double material_density);

/**
 * Calculate average attenuation for a specific particle type
 *
 * @param fluence Fluence tensor
 * @param particle_type Type of particle
 * @return Average attenuation factor
 */
double calculateAverageAttenuation(const Eigen::Tensor<double, 3>& fluence,
                                   ParticleType particle_type);

}  // namespace physics
}  // namespace rad_ml
