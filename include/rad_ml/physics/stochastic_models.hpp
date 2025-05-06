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