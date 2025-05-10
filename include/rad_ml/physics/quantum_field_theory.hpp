/**
 * Quantum Field Theory Models
 * 
 * Core implementation of quantum field theory models for radiation effects.
 */

#pragma once

#include <string>
#include <map>
#include <complex>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <rad_ml/physics/field_theory.hpp>

namespace rad_ml {
namespace physics {

// Forward declarations for types used in this interface
struct CrystalLattice {
    enum Type { FCC_TYPE, BCC, DIAMOND };
    
    Type type;
    double lattice_constant;
    double barrier_height;
    
    CrystalLattice(Type t = DIAMOND, double lc = 5.43, double bh = 1.0) 
        : type(t), lattice_constant(lc), barrier_height(bh) {}
};

// Proper defect distribution structure with vectors
struct DefectDistribution {
    std::vector<double> interstitials = {1.0, 2.0, 3.0};
    std::vector<double> vacancies = {1.0, 2.0, 3.0};
    std::vector<double> clusters = {0.5, 1.0, 1.5};
};

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

/**
 * Class representing a quantum field on a lattice
 */
template<int Dimensions = 3>
class QuantumField {
public:
    using ComplexMatrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;
    using RealMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    
    /**
     * Constructor with grid dimensions
     */
    QuantumField(const std::vector<int>& grid_dimensions, double lattice_spacing);
    
    /**
     * Initialize field with Gaussian random values
     */
    void initializeGaussian(double mean, double stddev);
    
    /**
     * Initialize field with coherent state
     */
    void initializeCoherentState(double amplitude, double phase);
    
    /**
     * Calculate kinetic energy term in Hamiltonian
     */
    RealMatrix calculateKineticTerm() const;
    
    /**
     * Calculate potential energy term in Hamiltonian
     */
    RealMatrix calculatePotentialTerm(const QFTParameters& params) const;
    
    /**
     * Calculate total energy of the field
     */
    double calculateTotalEnergy(const QFTParameters& params) const;
    
    /**
     * Time evolution using split-operator method
     */
    void evolve(const QFTParameters& params, int steps);
    
    /**
     * Calculate field correlation function
     */
    RealMatrix calculateCorrelationFunction(int max_distance) const;
    
    /**
     * Get field value at position
     */
    std::complex<double> getFieldAt(const std::vector<int>& position) const;
    
    /**
     * Set field value at position
     */
    void setFieldAt(const std::vector<int>& position, const std::complex<double>& value);
};

/**
 * Klein-Gordon equation for scalar fields
 */
class KleinGordonEquation {
public:
    /**
     * Constructor with parameters
     */
    KleinGordonEquation(const QFTParameters& params);
    
    /**
     * Calculate field evolution for one time step
     */
    void evolveField(QuantumField<3>& field) const;
    
    /**
     * Calculate field propagator
     */
    Eigen::MatrixXcd calculatePropagator(double momentum_squared) const;
};

/**
 * Dirac equation for spinor fields
 */
class DiracEquation {
public:
    /**
     * Constructor with parameters
     */
    DiracEquation(const QFTParameters& params);
    
    /**
     * Calculate field evolution for one time step
     */
    void evolveField(QuantumField<3>& field) const;
    
    /**
     * Calculate field propagator
     */
    Eigen::MatrixXcd calculatePropagator(const Eigen::Vector3d& momentum) const;

};

/**
 * Maxwell equations for electromagnetic fields
 */
class MaxwellEquations {
public:
    /**
     * Constructor with parameters
     */
    MaxwellEquations(const QFTParameters& params);
    
    /**
     * Calculate field evolution for one time step
     */
    void evolveField(QuantumField<3>& electric_field, QuantumField<3>& magnetic_field) const;
};

/**
 * Calculate quantum correction to defect formation energy
 * @param temperature Temperature in Kelvin
 * @param defect_energy Classical defect formation energy
 * @param params QFT parameters
 * @return Quantum corrected defect formation energy
 */
double calculateQuantumCorrectedDefectEnergy(
    double temperature,
    double defect_energy,
    const QFTParameters& params);

/**
 * Calculate quantum tunneling probability for defect migration
 * @param barrier_height Migration energy barrier in eV
 * @param temperature Temperature in Kelvin
 * @param params QFT parameters
 * @return Tunneling probability
 */
double calculateQuantumTunnelingProbability(
    double barrier_height,
    double temperature,
    const QFTParameters& params);

/**
 * Apply quantum field corrections to radiation damage model
 * @param defects Defect distribution from classical model
 * @param crystal Crystal lattice
 * @param params QFT parameters
 * @param temperature Temperature in Kelvin
 * @return Quantum-corrected defect distribution
 */
DefectDistribution applyQuantumFieldCorrections(
    const DefectDistribution& defects,
    const CrystalLattice& crystal,
    const QFTParameters& params,
    double temperature);

// Core quantum field theory functions
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

} // namespace physics
} // namespace rad_ml 