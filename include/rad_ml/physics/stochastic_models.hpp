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
 * Stochastic Models for Radiation Effects
 * 
 * This file contains stochastic models for simulating
 * the evolution of radiation-induced defects in materials.
 */

#pragma once

#include <vector>
#include <functional>
#include <Eigen/Dense>

namespace rad_ml {
namespace physics {

/**
 * Material parameters for stochastic models
 */
struct MaterialParameters {
    double diffusion_coefficient;   // m²/s
    double recombination_radius;    // Angstrom
    double migration_energy;        // eV
    double displacement_energy;     // eV
};

/**
 * Results from stochastic simulation
 */
struct SimulationResults {
    Eigen::VectorXd final_concentration;
    double statistical_error;
};

/**
 * Create drift term function for stochastic differential equation
 */
std::function<Eigen::VectorXd(const Eigen::VectorXd&, double, double)> 
createDriftTerm(const MaterialParameters& params, double generation_rate);

/**
 * Create diffusion term function for stochastic differential equation
 */
std::function<Eigen::MatrixXd(const Eigen::VectorXd&, double, double)> 
createDiffusionTerm(const MaterialParameters& params, double temperature);

/**
 * Calculate generation rate based on environment and material
 */
double calculateGenerationRate(
    const struct RadiationEnvironment& env, 
    const struct MaterialProperties& material);

/**
 * Solve stochastic differential equation for defect evolution
 */
SimulationResults solveStochasticDE(
    const Eigen::VectorXd& initial_concentrations,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&, double, double)>& drift_term,
    const std::function<Eigen::MatrixXd(const Eigen::VectorXd&, double, double)>& diffusion_term,
    int time_steps,
    double simulation_time,
    double temperature,
    double applied_stress);

} // namespace physics
} // namespace rad_ml 
