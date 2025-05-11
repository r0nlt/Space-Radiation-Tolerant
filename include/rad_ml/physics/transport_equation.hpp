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
/**
 * Radiation Transport Equation Models
 * 
 * This file contains models for solving the Boltzmann transport equation
 * to simulate radiation transport through materials.
 */

#pragma once

#include <Eigen/Dense>

namespace rad_ml {
namespace physics {

/**
 * Enum for particle types
 */
enum ParticleType {
    ProtonParticle,
    ElectronParticle,
    NeutronParticle,
    PhotonParticle,
    HeavyIonParticle
};

/**
 * Cross-section data for materials
 */
struct CrossSectionData {
    double total;            // Total cross-section
    double elastic;          // Elastic scattering cross-section
    double inelastic;        // Inelastic scattering cross-section
    double z_effective;      // Effective atomic number
};

/**
 * Solution to the transport equation
 */
struct TransportSolution {
    Eigen::Tensor<double, 3> fluence;    // Fluence tensor Φ(x,Ω,E)
    double convergence_error;            // Convergence error
};

/**
 * Setup radiation source based on environment
 */
Eigen::Tensor<double, 3> setupRadiationSource(
    const struct RadiationEnvironment& env,
    int spatial_points,
    int angular_points,
    int energy_bins);

/**
 * Generate material-specific cross-section tensors
 */
Eigen::Tensor<double, 2> generateMaterialCrossSections(
    const struct MaterialProperties& material,
    int energy_bins);

/**
 * Generate material-specific scattering cross-section tensors
 */
Eigen::Tensor<double, 4> generateScatteringCrossSections(
    const struct MaterialProperties& material,
    int angular_points,
    int energy_bins);

/**
 * Solve the Boltzmann transport equation
 */
TransportSolution solveTransportEquation(
    const Eigen::Tensor<double, 3>& initial_fluence,
    const Eigen::Tensor<double, 3>& source,
    const Eigen::Tensor<double, 2>& sigma_t,
    const Eigen::Tensor<double, 4>& sigma_s);

/**
 * Calculate dose distribution from fluence
 */
Eigen::Tensor<double, 1> calculateDoseDistribution(
    const Eigen::Tensor<double, 3>& fluence,
    double material_density);

/**
 * Calculate average attenuation for a specific particle type
 */
double calculateAverageAttenuation(
    const Eigen::Tensor<double, 3>& fluence,
    ParticleType particle_type);

} // namespace physics
} // namespace rad_ml 
