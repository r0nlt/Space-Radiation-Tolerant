/**
 * Field Theory Models for Radiation Effects
 * 
 * This file contains field theory models for simulating
 * the evolution of radiation-induced defects in materials.
 */

#pragma once

#include <vector>
#include <Eigen/Dense>

namespace rad_ml {
namespace physics {

/**
 * 3D grid for spatial discretization
 */
class Grid3D {
public:
    int size_x;
    int size_y;
    int size_z;
    double spacing;
    
    /**
     * Constructor with grid dimensions and spacing
     */
    Grid3D(int size_x, int size_y, int size_z, double spacing);
};

/**
 * Template for 3D field defined on a grid
 */
template<typename T = double>
class Field3D {
public:
    /**
     * Constructor with grid
     */
    Field3D(const Grid3D& grid);
    
    /**
     * Access operator for 3D indices
     */
    T& operator()(int i, int j, int k);
    
    /**
     * Const access operator for 3D indices
     */
    const T& operator()(int i, int j, int k) const;
    
    /**
     * Set all field values to zero
     */
    void setZero();
};

/**
 * Parameters for field theory calculations
 */
struct FieldParameters {
    double kappa;                   // Gradient energy coefficient
    std::vector<std::vector<double>> gamma;  // Interaction matrix
};

/**
 * Free energy functional for field theory calculations
 */
class FreeEnergyFunctional {
public:
    /**
     * Constructor with parameters
     */
    FreeEnergyFunctional(const FieldParameters& params);
    
    /**
     * Calculate functional derivatives δF/δC_i
     */
    std::vector<Field3D<double>> calculateDerivatives(const std::vector<Field3D<double>>& fields);
};

/**
 * Results from time evolution calculation
 */
struct TimeEvolutionResults {
    double total_defects_initial;
    double total_defects_final;
    std::vector<Field3D<double>> final_fields;
};

/**
 * Calculate gradient energy coefficient from material properties
 */
double calculateGradientEnergyCoefficient(const struct MaterialProperties& material);

/**
 * Create interaction matrix from defect formation energies
 */
std::vector<std::vector<double>> createInteractionMatrix(const std::vector<double>& defect_formation_energies);

/**
 * Initialize defect fields based on environment and material
 */
void initializeDefectFields(
    std::vector<Field3D<double>>& fields, 
    const struct RadiationEnvironment& env, 
    const struct MaterialProperties& material);

/**
 * Solve field equations for time evolution
 */
TimeEvolutionResults solveFieldEquations(
    std::vector<Field3D<double>>& fields,
    const std::vector<Field3D<double>>& derivatives,
    double radiation_dose,
    double temperature,
    double applied_stress);

/**
 * Calculate defect clustering ratio
 */
double calculateClusteringRatio(const std::vector<Field3D<double>>& fields);

} // namespace physics
} // namespace rad_ml 