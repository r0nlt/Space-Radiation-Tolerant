/**
 * Quantum Field Theory Models
 *
 * Core implementation of quantum field theory models for radiation effects.
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <complex>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <rad_ml/physics/field_theory.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad_ml {
namespace physics {

/**
 * Type-safe enum class for particle types
 */
enum class ParticleType {
    Proton,
    Electron,
    Neutron,
    Photon,
    HeavyIon,
    Positron,
    Muon,
    Neutrino,
    // Add more particle types as needed
};

/**
 * Class representing physical properties of a particle
 */
class Particle {
   public:
    /**
     * Construct a particle with basic properties
     *
     * @param type The particle type
     * @param mass The particle mass in kg
     * @param charge The particle charge in elementary charge units
     * @param spin The particle spin quantum number
     */
    Particle(ParticleType type, double mass, double charge, double spin)
        : type_(type), mass_(mass), charge_(charge), spin_(spin)
    {
    }

    /**
     * Factory methods for creating common particles
     */
    static Particle createProton()
    {
        return Particle(ParticleType::Proton, 1.6726219e-27, 1.0, 0.5);
    }

    static Particle createElectron()
    {
        return Particle(ParticleType::Electron, 9.1093837e-31, -1.0, 0.5);
    }

    static Particle createNeutron()
    {
        return Particle(ParticleType::Neutron, 1.6749275e-27, 0.0, 0.5);
    }

    static Particle createPhoton() { return Particle(ParticleType::Photon, 0.0, 0.0, 1.0); }

    /**
     * Getters for particle properties
     */
    ParticleType type() const { return type_; }
    double mass() const { return mass_; }
    double charge() const { return charge_; }
    double spin() const { return spin_; }

   private:
    ParticleType type_;
    double mass_;
    double charge_;
    double spin_;
};

// Forward declarations for types used in this interface
struct CrystalLattice {
    enum class Type { FCC, BCC, DIAMOND };

    Type type;
    double lattice_constant;
    double barrier_height;

    CrystalLattice(Type t = Type::DIAMOND, double lc = 5.43, double bh = 1.0)
        : type(t), lattice_constant(lc), barrier_height(bh)
    {
    }
};

// Proper defect distribution structure with vectors
struct DefectDistribution {
    std::unordered_map<ParticleType, std::vector<double>> interstitials;
    std::unordered_map<ParticleType, std::vector<double>> vacancies;
    std::unordered_map<ParticleType, std::vector<double>> clusters;

    // Default constructor initializes with values for a proton (backward compatibility)
    DefectDistribution()
    {
        interstitials[ParticleType::Proton] = {1.0, 2.0, 3.0};
        vacancies[ParticleType::Proton] = {1.0, 2.0, 3.0};
        clusters[ParticleType::Proton] = {0.5, 1.0, 1.5};
    }
};

// QFT parameters for quantum field calculations
struct QFTParameters {
    double hbar;                                      // Reduced Planck constant (eV·s)
    std::unordered_map<ParticleType, double> masses;  // Effective masses (kg)
    std::unordered_map<ParticleType, double>
        coupling_constants;        // Coupling constants for interactions
    double potential_coefficient;  // Potential energy coefficient
    double lattice_spacing;        // Lattice spacing (nm)
    double time_step;              // Simulation time step (s)
    double omega;                  // Angular frequency (rad/s)
    int dimensions;                // Number of spatial dimensions

    // New configurable parameters (previously hardcoded)
    double phase_evolution_rate = 1.0e15;  // Default oscillation rate (Hz)
    double damping_factor = 0.001;         // Default damping per second
    double temperature_threshold = 150.0;  // K
    double temperature_scaling_factor = 100.0;
    double position_factor_amplitude = 0.01;  // Amplitude of position-dependent variations

    QFTParameters()
        : hbar(6.582119569e-16),
          potential_coefficient(0.5),
          lattice_spacing(1.0),
          time_step(1.0e-18),
          omega(1.0e15),  // Default angular frequency
          dimensions(3)
    {
        // Initialize with default values for backward compatibility
        masses[ParticleType::Proton] = 1.6726219e-27;
        masses[ParticleType::Electron] = 9.1093837e-31;
        masses[ParticleType::Neutron] = 1.6749275e-27;
        masses[ParticleType::Photon] = 1.0e-30;          // Non-zero for numerical stability
        masses[ParticleType::HeavyIon] = 5.0e-27;        // Typical heavy ion mass (e.g., Carbon-12)
        masses[ParticleType::Positron] = 9.1093837e-31;  // Same as electron
        masses[ParticleType::Muon] = 1.8835316e-28;      // Muon mass
        masses[ParticleType::Neutrino] = 1.0e-36;        // Small but non-zero for stability

        coupling_constants[ParticleType::Proton] = 0.1;
        coupling_constants[ParticleType::Electron] = 0.1;
        coupling_constants[ParticleType::Neutron] = 0.1;
        coupling_constants[ParticleType::Photon] = 0.1;
        coupling_constants[ParticleType::HeavyIon] = 0.15;  // Stronger coupling for heavy ions
        coupling_constants[ParticleType::Positron] = 0.1;   // Same as electron
        coupling_constants[ParticleType::Muon] = 0.12;      // Slightly different for muon
        coupling_constants[ParticleType::Neutrino] = 0.01;  // Weak coupling for neutrinos
    }

    // For backward compatibility - get mass for a specific particle type
    double getMass(ParticleType type = ParticleType::Proton) const
    {
        auto it = masses.find(type);
        if (it == masses.end()) {
            throw std::runtime_error("Mass not defined for particle type: " +
                                     std::to_string(static_cast<int>(type)));
        }
        return it->second;
    }

    // For backward compatibility - get coupling constant for a specific particle type
    double getCouplingConstant(ParticleType type = ParticleType::Proton) const
    {
        auto it = coupling_constants.find(type);
        if (it == coupling_constants.end()) {
            throw std::runtime_error("Coupling constant not defined for particle type: " +
                                     std::to_string(static_cast<int>(type)));
        }
        return it->second;
    }
};

/**
 * Class representing a quantum field on a lattice
 */
template <int Dimensions = 3>
class QuantumField {
   public:
    using ComplexMatrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;
    using RealMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    /**
     * Constructor with grid dimensions
     *
     * @param grid_dimensions The dimensions of the grid
     * @param lattice_spacing The spacing between lattice points
     * @param particle_type The type of particle (default: Proton)
     */
    QuantumField(const std::vector<int>& grid_dimensions, double lattice_spacing,
                 ParticleType particle_type = ParticleType::Proton);

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
     *
     * @param params QFT parameters
     * @param particle_type Optional particle type (uses field's particle type if not specified)
     */
    RealMatrix calculatePotentialTerm(
        const QFTParameters& params,
        std::optional<ParticleType> particle_type = std::nullopt) const;

    /**
     * Calculate total energy of the field
     *
     * @param params QFT parameters
     * @param particle_type Optional particle type (uses field's particle type if not specified)
     */
    double calculateTotalEnergy(const QFTParameters& params,
                                std::optional<ParticleType> particle_type = std::nullopt) const;

    /**
     * Time evolution using split-operator method
     *
     * @param params QFT parameters
     * @param steps Number of time steps to evolve
     * @param particle_type Optional particle type (uses field's particle type if not specified)
     */
    void evolve(const QFTParameters& params, int steps,
                std::optional<ParticleType> particle_type = std::nullopt);

    /**
     * Calculate field correlation function
     */
    RealMatrix calculateCorrelationFunction(int max_distance) const;

    /**
     * Get field value at multi-dimensional position (bounds-checked).
     */
    std::complex<double> getFieldAt(const std::vector<int>& position) const;

    /**
     * Set field value at multi-dimensional position (bounds-checked).
     */
    void setFieldAt(const std::vector<int>& position, const std::complex<double>& value);

    /**
     * Direct flat-index access (no allocation, no bounds check). Use in hot loops.
     */
    const std::complex<double>& operator[](size_t flat_index) const { return field_data_[flat_index]; }
    std::complex<double>& operator[](size_t flat_index) { return field_data_[flat_index]; }

    /**
     * Convert (i,j,k) to flat index for a 3D field. No bounds check.
     */
    size_t flatIndex(int i, int j, int k) const {
        return static_cast<size_t>(i) * dimensions_[1] * dimensions_[2]
             + static_cast<size_t>(j) * dimensions_[2]
             + static_cast<size_t>(k);
    }

    /**
     * Total number of lattice sites.
     */
    size_t size() const { return field_data_.size(); }

    /**
     * Compute the discrete Laplacian at (i,j,k) with periodic BC.
     * Returns (sum of 6 neighbors - 6*center) / dx^2.
     */
    std::complex<double> laplacian3D(int i, int j, int k) const;

    /**
     * Compute the gradient energy density 0.5 * sum_d |phi(x+d)-phi(x)|^2 / dx^2
     * at site (i,j,k) using forward differences with periodic BC.
     */
    double gradientEnergyDensity(int i, int j, int k) const;

    /**
     * Get the particle type of this field
     */
    ParticleType getParticleType() const { return particle_type_; }

    /**
     * Set the particle type of this field
     */
    void setParticleType(ParticleType type) { particle_type_ = type; }

    /**
     * Get the dimensions of this field
     */
    const std::vector<int>& getDimensions() const { return dimensions_; }

    double getLatticeSpacing() const { return lattice_spacing_; }

    /**
     * @brief Optimized quantum field computation
     *
     * @param input_field Input field values
     * @param output_field Output field values (will be computed)
     */
    void computeOptimized(const std::vector<std::complex<double>>& input_field,
                          std::vector<std::complex<double>>& output_field) const;

    /**
     * @brief Compute Laplacian for a field at a specific index
     */
    std::complex<double> computeLaplacian(const std::vector<std::complex<double>>& field,
                                          size_t index) const;

    /**
     * @brief Standard computation (fallback)
     */
    void computeStandard(const std::vector<std::complex<double>>& input_field,
                         std::vector<std::complex<double>>& output_field) const;

#ifdef __AVX2__
    /**
     * @brief SIMD-optimized computation using AVX2
     */
    void computeSIMD(const std::vector<std::complex<double>>& input_field,
                     std::vector<std::complex<double>>& output_field) const;
#endif

   private:
    ParticleType particle_type_;
    double lattice_spacing_;
    std::vector<int> dimensions_;
    std::vector<std::complex<double>> field_data_;
    mutable std::mt19937 random_generator_{std::random_device{}()};

    /**
     * Calculate 1D array index from multi-dimensional position
     */
    std::size_t calculateIndex(const std::vector<int>& position) const;
};

/**
 * Klein-Gordon equation for scalar fields
 */
class KleinGordonEquation {
   public:
    /**
     * Constructor with parameters
     *
     * @param params QFT parameters
     * @param particle_type Particle type this equation applies to
     */
    KleinGordonEquation(const QFTParameters& params,
                        ParticleType particle_type = ParticleType::Proton);

    /**
     * Symplectic (leapfrog) field evolution for one time step.
     * Uses the conjugate momentum field pi for energy-conserving updates.
     * On first call, pi is initialized to zero (field at rest).
     *
     * @param field The quantum field to evolve
     */
    void evolveField(QuantumField<3>& field);

    /**
     * Calculate field propagator
     *
     * @param momentum_squared The squared momentum
     * @param particle_type Optional particle type override
     */
    Eigen::MatrixXcd calculatePropagator(
        double momentum_squared, std::optional<ParticleType> particle_type = std::nullopt) const;

    /**
     * Compute the full Hamiltonian H = ½Σ|π|² + ½Σ|∇φ|²/dx² + ½m²Σ|φ|²
     * This is the exactly-conserved quantity for the symplectic integrator.
     */
    double computeHamiltonian(const QuantumField<3>& field) const;

    /**
     * Get the particle type this equation applies to
     */
    ParticleType getParticleType() const { return particle_type_; }

   private:
    const QFTParameters& params_;
    ParticleType particle_type_;
    std::vector<std::complex<double>> pi_field_;
    bool initialized_ = false;
};

/**
 * Dirac equation for spinor fields
 */
class DiracEquation {
   public:
    /**
     * Constructor with parameters
     *
     * @param params QFT parameters
     * @param particle_type Particle type this equation applies to
     */
    DiracEquation(const QFTParameters& params, ParticleType particle_type = ParticleType::Electron);

    /**
     * Calculate field evolution for one time step
     *
     * @param field The quantum field to evolve
     */
    void evolveField(QuantumField<3>& field) const;

    /**
     * Calculate field propagator
     *
     * @param momentum The momentum vector
     * @param particle_type Optional particle type override
     */
    Eigen::MatrixXcd calculatePropagator(
        const Eigen::Vector3d& momentum,
        std::optional<ParticleType> particle_type = std::nullopt) const;

    /**
     * Get the particle type this equation applies to
     */
    ParticleType getParticleType() const { return particle_type_; }

   private:
    const QFTParameters& params_;
    ParticleType particle_type_;
};

/**
 * Maxwell equations for electromagnetic fields
 */
class MaxwellEquations {
   public:
    /**
     * Constructor with parameters
     *
     * @param params QFT parameters
     */
    MaxwellEquations(const QFTParameters& params);

    /**
     * Symplectic (leapfrog) field evolution for one time step.
     * Each field evolves as a massless wave equation with velocity
     * fields stored internally for energy conservation.
     */
    void evolveField(QuantumField<3>& electric_field, QuantumField<3>& magnetic_field);

    /**
     * Compute the full Hamiltonian H = sum of kinetic + gradient energy
     * for both E and B fields. This is the exactly-conserved quantity.
     */
    double computeHamiltonian(const QuantumField<3>& electric_field,
                              const QuantumField<3>& magnetic_field) const;

   private:
    const QFTParameters& params_;
    std::vector<std::complex<double>> e_velocity_;
    std::vector<std::complex<double>> b_velocity_;
    bool initialized_ = false;
};

/**
 * Calculate quantum correction to defect formation energy
 * @param temperature Temperature in Kelvin
 * @param defect_energy Classical defect formation energy
 * @param params QFT parameters
 * @param particle_type Particle type to consider
 * @return Quantum corrected defect formation energy
 */
double calculateQuantumCorrectedDefectEnergy(double temperature, double defect_energy,
                                             const QFTParameters& params,
                                             ParticleType particle_type = ParticleType::Proton);

/**
 * Calculate quantum tunneling probability for defect migration
 * @param barrier_height Migration energy barrier in eV
 * @param temperature Temperature in Kelvin
 * @param params QFT parameters
 * @param particle_type Particle type to consider
 * @return Tunneling probability
 */
double calculateQuantumTunnelingProbability(double barrier_height, double temperature,
                                            const QFTParameters& params,
                                            ParticleType particle_type = ParticleType::Proton);

/**
 * Apply quantum field corrections to radiation damage model
 * @param defects Defect distribution from classical model
 * @param crystal Crystal lattice
 * @param params QFT parameters
 * @param temperature Temperature in Kelvin
 * @param particle_types Vector of particle types to consider (if empty, all particles in defects
 * are considered)
 * @return Quantum-corrected defect distribution
 */
DefectDistribution applyQuantumFieldCorrections(
    const DefectDistribution& defects, const CrystalLattice& crystal, const QFTParameters& params,
    double temperature, const std::vector<ParticleType>& particle_types = {});

// Core quantum field theory functions
double calculateQuantumTunnelingProbability(double barrier_height, double mass, double hbar,
                                            double temperature);

double solveKleinGordonEquation(double hbar, double mass, double potential_coeff,
                                double coupling_constant, double lattice_spacing, double time_step);

double calculateZeroPointEnergyContribution(double hbar, double mass, double lattice_constant,
                                            double temperature);

}  // namespace physics
}  // namespace rad_ml
