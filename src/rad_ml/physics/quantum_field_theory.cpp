/**
 * Quantum Field Theory Implementation
 *
 * This file implements the quantum field theory models for radiation effects.
 */

#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>

// Performance optimizations
#include <algorithm>
#include <array>
#ifdef __AVX2__
#include <immintrin.h>
#endif

// Include Eigen properly
#ifdef __has_include
#if __has_include(<eigen3/Eigen/Dense>)
#include <eigen3/Eigen/Dense>
#elif __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#else
#error "Could not find Eigen/Dense"
#endif
#else
// Fallback for older compilers
#include <Eigen/Dense>
#endif

#include <rad_ml/physics/quantum_field_theory.hpp>

namespace rad_ml {
namespace physics {

// Implementation of QuantumField methods
template <int Dimensions>
QuantumField<Dimensions>::QuantumField(const std::vector<int>& grid_dimensions,
                                       double lattice_spacing, ParticleType particle_type)
    : particle_type_(particle_type), lattice_spacing_(lattice_spacing)
{
    // Validate dimensions
    if (grid_dimensions.size() != Dimensions) {
        throw std::invalid_argument("Dimension mismatch: Expected " + std::to_string(Dimensions) +
                                    " dimensions, got " + std::to_string(grid_dimensions.size()));
    }

    dimensions_ = grid_dimensions;

    // Calculate total size and initialize field_data_
    size_t total_size = 1;
    for (int dim : dimensions_) {
        // Check for overflow before multiplying
        if (total_size > std::numeric_limits<size_t>::max() / dim) {
            throw std::overflow_error("Field dimensions too large");
        }
        total_size *= dim;
    }

    // Initialize field data with zeros
    field_data_.resize(total_size, std::complex<double>(0.0, 0.0));

    std::cout << "Initialized quantum field with dimensions: ";
    for (size_t i = 0; i < dimensions_.size(); ++i) {
        std::cout << dimensions_[i];
        if (i < dimensions_.size() - 1) std::cout << "x";
    }
    std::cout << " (" << total_size << " points)" << std::endl;
}

template <int Dimensions>
void QuantumField<Dimensions>::initializeGaussian(double mean, double stddev)
{
    // Use class member random generator instead of creating a new one
    std::normal_distribution<double> real_dist(mean, stddev);
    std::normal_distribution<double> imag_dist(0.0, stddev);

    // Initialize each point in the field with a random value
    for (auto& value : field_data_) {
        value = std::complex<double>(real_dist(random_generator_), imag_dist(random_generator_));
    }

    std::cout << "Initialized quantum field with Gaussian distribution (mean=" << mean
              << ", stddev=" << stddev << ")" << std::endl;
}

template <int Dimensions>
void QuantumField<Dimensions>::initializeCoherentState(double amplitude, double phase)
{
    std::complex<double> base_value = amplitude * std::complex<double>(cos(phase), sin(phase));

    // Initialize a coherent state with the given amplitude and phase
    std::vector<int> position(dimensions_.size(), 0);

    std::function<void(int)> iterate = [&](int dim) {
        if (dim == dimensions_.size()) {
            // We've set all dimensions, initialize this point

            // Calculate distance from center of the grid
            double distance_squared = 0.0;
            for (size_t i = 0; i < dimensions_.size(); ++i) {
                double center = dimensions_[i] / 2.0;
                double dist = (position[i] - center) / center;
                distance_squared += dist * dist;
            }

            // Coherent state has Gaussian envelope
            double envelope = exp(-distance_squared);

            // Set the field value
            setFieldAt(position, base_value * envelope);
            return;
        }

        // Iterate through this dimension
        for (int i = 0; i < dimensions_[dim]; i++) {
            position[dim] = i;
            iterate(dim + 1);
        }
    };

    // Start the iteration from dimension 0
    iterate(0);

    std::cout << "Initialized quantum field with coherent state (amplitude=" << amplitude
              << ", phase=" << phase << ")" << std::endl;
}

template <int Dimensions>
typename QuantumField<Dimensions>::RealMatrix QuantumField<Dimensions>::calculateKineticTerm() const
{
    // Simple implementation to satisfy the compiler
    return RealMatrix(1, 1);
}

template <int Dimensions>
typename QuantumField<Dimensions>::RealMatrix QuantumField<Dimensions>::calculatePotentialTerm(
    const QFTParameters& params, std::optional<ParticleType> particle_type) const
{
    // Use provided particle type or fall back to the field's type
    const ParticleType type = particle_type.value_or(particle_type_);

    // Simple implementation to satisfy the compiler
    return RealMatrix(1, 1);
}

template <int Dimensions>
double QuantumField<Dimensions>::calculateTotalEnergy(
    const QFTParameters& params, std::optional<ParticleType> particle_type) const
{
    const ParticleType type = particle_type.value_or(particle_type_);
    const double mass = params.getMass(type);

    double gradientEnergy = 0.0;
    double massEnergy = 0.0;

    if (dimensions_.size() == 3 && dimensions_[0] > 2 && dimensions_[1] > 2 && dimensions_[2] > 2) {
        const double dx = lattice_spacing_;
        const int Nx = dimensions_[0], Ny = dimensions_[1], Nz = dimensions_[2];

        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    std::complex<double> center = getFieldAt({i, j, k});

                    // Gradient energy: 0.5 * sum_d |phi(x+d) - phi(x)|^2 / dx^2
                    int ip = (i + 1) % Nx, jp = (j + 1) % Ny, kp = (k + 1) % Nz;
                    std::complex<double> dx_diff = getFieldAt({ip, j, k}) - center;
                    std::complex<double> dy_diff = getFieldAt({i, jp, k}) - center;
                    std::complex<double> dz_diff = getFieldAt({i, j, kp}) - center;

                    gradientEnergy += 0.5 * (std::norm(dx_diff) + std::norm(dy_diff) + std::norm(dz_diff)) / (dx * dx);

                    // Mass term: 0.5 * m^2 * |phi|^2
                    massEnergy += 0.5 * mass * mass * std::norm(center);
                }
            }
        }
    }
    else {
        // Fallback for non-3D or small fields: mass term only
        std::vector<int> position(dimensions_.size(), 0);
        std::function<void(int)> iterate = [&](int dim) {
            if (dim == (int)dimensions_.size()) {
                massEnergy += 0.5 * mass * mass * std::norm(getFieldAt(position));
                return;
            }
            for (int i = 0; i < dimensions_[dim]; i++) {
                position[dim] = i;
                iterate(dim + 1);
            }
        };
        iterate(0);
    }

    double totalEnergy = gradientEnergy + massEnergy;

    std::cout << "Energy components - Gradient: " << gradientEnergy
              << ", Mass: " << massEnergy << std::endl;

    return totalEnergy;
}

template <int Dimensions>
void QuantumField<Dimensions>::evolve(const QFTParameters& params, int steps,
                                      std::optional<ParticleType> particle_type)
{
    // Use provided particle type or fall back to the field's type
    const ParticleType type = particle_type.value_or(particle_type_);

    // Get time step size from parameters
    double dt = params.time_step;

    // Simple time evolution loop
    for (int step = 0; step < steps; step++) {
        // Apply time evolution to each point in the field
        std::vector<int> position(dimensions_.size(), 0);

        // Recursive function to iterate through multi-dimensional field
        std::function<void(int)> iterate = [&](int dim) {
            if (dim == dimensions_.size()) {
                // We've set all dimensions, process this point
                std::complex<double> current_value = getFieldAt(position);

                // Apply simple harmonic oscillator evolution
                // Phase evolves with time
                double amplitude = std::abs(current_value);
                double phase = std::arg(current_value) + params.omega * dt;

                // Create new field value
                std::complex<double> new_value =
                    amplitude * std::complex<double>(cos(phase), sin(phase));

                // Apply small damping
                new_value *= (1.0 - 0.001 * dt);

                // Set the new field value
                setFieldAt(position, new_value);
                return;
            }

            // Iterate through this dimension
            for (int i = 0; i < dimensions_[dim]; i++) {
                position[dim] = i;
                iterate(dim + 1);
            }
        };

        // Start the iteration from dimension 0
        iterate(0);
    }

    // Debug output
    std::cout << "QuantumField: Evolved field for " << steps << " steps with dt = " << dt
              << std::endl;
}

template <int Dimensions>
typename QuantumField<Dimensions>::RealMatrix
QuantumField<Dimensions>::calculateCorrelationFunction(int max_distance) const
{
    // Simple implementation to satisfy the compiler
    RealMatrix result(max_distance + 1, 1);
    for (int i = 0; i <= max_distance; i++) {
        result(i, 0) = 1.0 / (i + 1.0);
    }
    return result;
}

// Calculate index helper method
template <int Dimensions>
std::size_t QuantumField<Dimensions>::calculateIndex(const std::vector<int>& position) const
{
    // Validate position dimensions
    if (position.size() != dimensions_.size()) {
        throw std::invalid_argument("Position vector dimension mismatch. Expected " +
                                    std::to_string(dimensions_.size()) + ", got " +
                                    std::to_string(position.size()));
    }

    // Handle empty dimensions case
    if (dimensions_.empty()) {
        return 0;
    }

    std::size_t index = 0;
    std::size_t stride = 1;

    // Use safe arithmetic: start from highest dimension and go down
    for (int i = static_cast<int>(dimensions_.size()) - 1; i >= 0; --i) {
        // Bounds checking
        if (position[i] < 0 || position[i] >= dimensions_[i]) {
            throw std::out_of_range("Position component " + std::to_string(i) + " out of range: " +
                                    std::to_string(position[i]) + " (valid range: 0 to " +
                                    std::to_string(dimensions_[i] - 1) + ")");
        }

        // Overflow check for stride calculation
        if (stride >
            std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimensions_[i])) {
            throw std::overflow_error("Index calculation would overflow");
        }

        index += static_cast<std::size_t>(position[i]) * stride;
        stride *= static_cast<std::size_t>(dimensions_[i]);
    }

    return index;
}

// Compute Laplacian for a field at a specific index
template <int Dimensions>
std::complex<double> QuantumField<Dimensions>::computeLaplacian(
    const std::vector<std::complex<double>>& field, size_t index) const
{
    // Simple Laplacian computation - sum of neighbors minus center * number of neighbors
    // This is a simplified version for performance
    std::complex<double> laplacian = -6.0 * field[index];  // 6 neighbors in 3D

    // Add contributions from neighbors (simplified - assuming cubic grid)
    // In a real implementation, this would iterate through actual neighbors
    if (index > 0) laplacian += field[index - 1];
    if (index < field.size() - 1) laplacian += field[index + 1];

    return laplacian;
}

// Get field value at position
template <int Dimensions>
std::complex<double> QuantumField<Dimensions>::getFieldAt(const std::vector<int>& position) const
{
    std::size_t index = calculateIndex(position);
    return field_data_[index];
}

// Set field value at position
template <int Dimensions>
void QuantumField<Dimensions>::setFieldAt(const std::vector<int>& position,
                                          const std::complex<double>& value)
{
    std::size_t index = calculateIndex(position);
    field_data_[index] = value;
}

// Implementation of KleinGordonEquation methods
KleinGordonEquation::KleinGordonEquation(const QFTParameters& params, ParticleType particle_type)
    : params_(params), particle_type_(particle_type)
{
    // Simple implementation to satisfy the compiler
}

void KleinGordonEquation::evolveField(QuantumField<3>& field)
{
    if (field.getParticleType() != particle_type_) {
        throw std::invalid_argument("Particle type mismatch in KleinGordonEquation::evolveField");
    }

    std::cout << "KleinGordon: Starting field evolution for particle type "
              << static_cast<int>(particle_type_) << "..." << std::endl;

    const auto& dims = field.getDimensions();
    const size_t Nx = dims[0], Ny = dims[1], Nz = dims[2];
    const size_t N = Nx * Ny * Nz;

    if (!initialized_) {
        pi_field_.assign(N, {0.0, 0.0});
        initialized_ = true;
    }

    const double dt = params_.time_step;
    const double dx = params_.lattice_spacing;
    const double dx2 = dx * dx;
    const double mass = params_.getMass(particle_type_);
    const double m2 = mass * mass;

    // Symplectic leapfrog (Stormer-Verlet):
    //   pi(t+dt/2) = pi(t) + (dt/2) * [Laplacian(phi) - m^2 * phi]
    //   phi(t+dt)  = phi(t) + dt * pi(t+dt/2)
    //   pi(t+dt)   = pi(t+dt/2) + (dt/2) * [Laplacian(phi_new) - m^2 * phi_new]

    auto idx = [&](int i, int j, int k) -> size_t {
        return static_cast<size_t>(i) * Ny * Nz + static_cast<size_t>(j) * Nz + static_cast<size_t>(k);
    };

    // Helper to compute Laplacian at (i,j,k) with periodic boundary conditions
    auto laplacian_at = [&](int i, int j, int k) -> std::complex<double> {
        int ip = (i + 1) % (int)Nx, im = (i - 1 + (int)Nx) % (int)Nx;
        int jp = (j + 1) % (int)Ny, jm = (j - 1 + (int)Ny) % (int)Ny;
        int kp = (k + 1) % (int)Nz, km = (k - 1 + (int)Nz) % (int)Nz;

        std::complex<double> center = field.getFieldAt({i, j, k});
        return (field.getFieldAt({ip, j, k}) + field.getFieldAt({im, j, k})
              + field.getFieldAt({i, jp, k}) + field.getFieldAt({i, jm, k})
              + field.getFieldAt({i, j, kp}) + field.getFieldAt({i, j, km})
              - 6.0 * center) / dx2;
    };

    // Step 1: half-kick pi
    for (int i = 0; i < (int)Nx; i++) {
        for (int j = 0; j < (int)Ny; j++) {
            for (int k = 0; k < (int)Nz; k++) {
                std::complex<double> phi = field.getFieldAt({i, j, k});
                std::complex<double> lap = laplacian_at(i, j, k);
                pi_field_[idx(i,j,k)] += 0.5 * dt * (lap - m2 * phi);
            }
        }
    }

    // Step 2: full drift phi
    for (int i = 0; i < (int)Nx; i++) {
        for (int j = 0; j < (int)Ny; j++) {
            for (int k = 0; k < (int)Nz; k++) {
                std::complex<double> phi = field.getFieldAt({i, j, k});
                phi += dt * pi_field_[idx(i,j,k)];
                field.setFieldAt({i, j, k}, phi);
            }
        }
    }

    // Step 3: half-kick pi (with updated phi)
    for (int i = 0; i < (int)Nx; i++) {
        for (int j = 0; j < (int)Ny; j++) {
            for (int k = 0; k < (int)Nz; k++) {
                std::complex<double> phi = field.getFieldAt({i, j, k});
                std::complex<double> lap = laplacian_at(i, j, k);
                pi_field_[idx(i,j,k)] += 0.5 * dt * (lap - m2 * phi);
            }
        }
    }

    std::cout << "KleinGordon: Field evolution step complete." << std::endl;
}

Eigen::MatrixXcd KleinGordonEquation::calculatePropagator(
    double momentum_squared, std::optional<ParticleType> particle_type) const
{
    // Use provided particle type or fall back to the equation's type
    const ParticleType type = particle_type.value_or(particle_type_);

    // Get the mass for this particle type
    const double mass = params_.getMass(type);

    // Simple implementation to satisfy the compiler
    Eigen::MatrixXcd result(1, 1);
    result(0, 0) = std::complex<double>(1.0, 0.0);
    return result;
}

double KleinGordonEquation::computeHamiltonian(const QuantumField<3>& field) const
{
    const auto& dims = field.getDimensions();
    const int Nx = dims[0], Ny = dims[1], Nz = dims[2];
    const double dx = params_.lattice_spacing;
    const double mass = params_.getMass(particle_type_);
    const double m2 = mass * mass;

    double kinetic = 0.0;
    double gradient = 0.0;
    double massTerm = 0.0;

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                size_t idx = static_cast<size_t>(i) * Ny * Nz
                           + static_cast<size_t>(j) * Nz
                           + static_cast<size_t>(k);

                // Kinetic: ½|π|²
                if (initialized_ && idx < pi_field_.size()) {
                    kinetic += 0.5 * std::norm(pi_field_[idx]);
                }

                std::complex<double> center = field.getFieldAt({i, j, k});

                // Gradient: ½|∇φ|² via forward differences (periodic BC)
                int ip = (i + 1) % Nx, jp = (j + 1) % Ny, kp = (k + 1) % Nz;
                gradient += 0.5 * (std::norm(field.getFieldAt({ip, j, k}) - center)
                                 + std::norm(field.getFieldAt({i, jp, k}) - center)
                                 + std::norm(field.getFieldAt({i, j, kp}) - center)) / (dx * dx);

                // Mass: ½m²|φ|²
                massTerm += 0.5 * m2 * std::norm(center);
            }
        }
    }

    return kinetic + gradient + massTerm;
}

// Implementation of DiracEquation methods
DiracEquation::DiracEquation(const QFTParameters& params, ParticleType particle_type)
    : params_(params), particle_type_(particle_type)
{
    // Simple implementation to satisfy the compiler
}

void DiracEquation::evolveField(QuantumField<3>& field) const
{
    // Check if field particle type matches equation particle type
    if (field.getParticleType() != particle_type_) {
        // Particle mismatch - either skip or throw an exception
        return;
    }

    // Simple implementation to satisfy the compiler
}

Eigen::MatrixXcd DiracEquation::calculatePropagator(const Eigen::Vector3d& momentum,
                                                    std::optional<ParticleType> particle_type) const
{
    // Use provided particle type or fall back to the equation's type
    const ParticleType type = particle_type.value_or(particle_type_);

    // Get the mass for this particle type
    const double mass = params_.getMass(type);

    // Simple implementation to satisfy the compiler
    Eigen::MatrixXcd result(1, 1);
    result(0, 0) = std::complex<double>(1.0, 0.0);
    return result;
}

// Implementation of MaxwellEquations methods
MaxwellEquations::MaxwellEquations(const QFTParameters& params) : params_(params)
{
    // Simple implementation to satisfy the compiler
}

void MaxwellEquations::evolveField(QuantumField<3>& electric_field,
                                   QuantumField<3>& magnetic_field)
{
    if (electric_field.getParticleType() != ParticleType::Photon ||
        magnetic_field.getParticleType() != ParticleType::Photon) {
        throw std::invalid_argument("Non-photon field provided to MaxwellEquations");
    }

    std::cout << "Maxwell: Starting electromagnetic field evolution..." << std::endl;

    const auto& dims = electric_field.getDimensions();
    const int Nx = dims[0], Ny = dims[1], Nz = dims[2];
    const size_t N = static_cast<size_t>(Nx) * Ny * Nz;

    if (!initialized_) {
        e_velocity_.assign(N, {0.0, 0.0});
        b_velocity_.assign(N, {0.0, 0.0});
        initialized_ = true;
    }

    const double dt = params_.time_step;
    const double dx = params_.lattice_spacing;

    // Symplectic leapfrog for the massless wave equation ∂²φ/∂t² = c²∇²φ (c=1):
    //   v += dt/2 * Lap(φ)
    //   φ += dt * v
    //   v += dt/2 * Lap(φ_new)

    auto idx = [&](int i, int j, int k) -> size_t {
        return static_cast<size_t>(i) * Ny * Nz + static_cast<size_t>(j) * Nz + static_cast<size_t>(k);
    };

    auto lap = [&](QuantumField<3>& f, int i, int j, int k) -> std::complex<double> {
        int ip = (i+1)%Nx, im = (i-1+Nx)%Nx;
        int jp = (j+1)%Ny, jm = (j-1+Ny)%Ny;
        int kp = (k+1)%Nz, km = (k-1+Nz)%Nz;
        std::complex<double> center = f.getFieldAt({i, j, k});
        return (f.getFieldAt({ip,j,k}) + f.getFieldAt({im,j,k})
              + f.getFieldAt({i,jp,k}) + f.getFieldAt({i,jm,k})
              + f.getFieldAt({i,j,kp}) + f.getFieldAt({i,j,km})
              - 6.0 * center) / (dx * dx);
    };

    // Evolve both fields with the same symplectic integrator
    auto evolve_one = [&](QuantumField<3>& field, std::vector<std::complex<double>>& vel) {
        // Half-kick velocity
        for (int i = 0; i < Nx; i++)
            for (int j = 0; j < Ny; j++)
                for (int k = 0; k < Nz; k++)
                    vel[idx(i,j,k)] += 0.5 * dt * lap(field, i, j, k);

        // Full drift field
        for (int i = 0; i < Nx; i++)
            for (int j = 0; j < Ny; j++)
                for (int k = 0; k < Nz; k++) {
                    auto phi = field.getFieldAt({i, j, k});
                    field.setFieldAt({i, j, k}, phi + dt * vel[idx(i,j,k)]);
                }

        // Half-kick velocity with updated field
        for (int i = 0; i < Nx; i++)
            for (int j = 0; j < Ny; j++)
                for (int k = 0; k < Nz; k++)
                    vel[idx(i,j,k)] += 0.5 * dt * lap(field, i, j, k);
    };

    evolve_one(electric_field, e_velocity_);
    evolve_one(magnetic_field, b_velocity_);

    std::cout << "Maxwell: Field evolution step complete." << std::endl;
}

double MaxwellEquations::computeHamiltonian(const QuantumField<3>& electric_field,
                                            const QuantumField<3>& magnetic_field) const
{
    const auto& dims = electric_field.getDimensions();
    const int Nx = dims[0], Ny = dims[1], Nz = dims[2];
    const double dx = params_.lattice_spacing;

    double energy = 0.0;

    auto field_energy = [&](const QuantumField<3>& f,
                            const std::vector<std::complex<double>>& vel) {
        double kinetic = 0.0, grad = 0.0;
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    size_t id = static_cast<size_t>(i) * Ny * Nz
                              + static_cast<size_t>(j) * Nz
                              + static_cast<size_t>(k);

                    kinetic += 0.5 * std::norm(vel[id]);

                    std::complex<double> center = f.getFieldAt({i, j, k});
                    int ip = (i+1)%Nx, jp = (j+1)%Ny, kp = (k+1)%Nz;
                    grad += 0.5 * (std::norm(f.getFieldAt({ip,j,k}) - center)
                                 + std::norm(f.getFieldAt({i,jp,k}) - center)
                                 + std::norm(f.getFieldAt({i,j,kp}) - center)) / (dx * dx);
                }
            }
        }
        return kinetic + grad;
    };

    if (initialized_) {
        energy = field_energy(electric_field, e_velocity_)
               + field_energy(magnetic_field, b_velocity_);
    }
    return energy;
}

// Implementation of utility functions
double calculateQuantumCorrectedDefectEnergy(double temperature, double defect_energy,
                                             const QFTParameters& params,
                                             ParticleType particle_type)
{
    // Get mass for the specific particle type
    const double mass = params.getMass(particle_type);

    // Calculate quantum correction
    double correction = calculateZeroPointEnergyContribution(params.hbar, mass,
                                                             params.lattice_spacing, temperature);

    // Apply correction to classical defect energy
    // Quantum effects generally lower the effective defect formation energy
    return defect_energy - correction;
}

double calculateQuantumTunnelingProbability(double barrier_height, double temperature,
                                            const QFTParameters& params, ParticleType particle_type)
{
    // Get mass for the specific particle type
    const double mass = params.getMass(particle_type);

    return calculateQuantumTunnelingProbability(barrier_height, mass, params.hbar, temperature);
}

double calculateQuantumTunnelingProbability(double barrier_height, double mass, double hbar,
                                            double temperature)
{
    // Implementation using WKB approximation for tunneling through a barrier
    const double kB = 8.617333262e-5;  // Boltzmann constant in eV/K
    double thermal_energy = kB * temperature;

    // Convert barrier height from eV to J
    double barrier_J = barrier_height * 1.602176634e-19;

    // Convert mass to kg
    double mass_kg = mass;

    // Convert hbar to J·s
    double hbar_J = hbar * 1.602176634e-19;

    // Calculate barrier width (simplified model)
    double width = 2.0e-10;  // 2 Angstroms as a typical atomic distance

    // Safety check for parameters to prevent numerical issues
    if (barrier_height <= 0.0 || mass <= 0.0 || hbar <= 0.0) {
        return 0.0;
    }

    // Calculate the WKB tunneling probability
    double exponent = -2.0 * width * std::sqrt(2.0 * mass_kg * barrier_J) / hbar_J;
    double P_tunnel = std::exp(exponent);

    // Factor in thermal activation (higher temperature reduces tunneling importance)
    double P_thermal = std::exp(-barrier_height / thermal_energy);

    // Total probability combines tunneling and thermal effects
    double total_prob = P_tunnel + P_thermal - P_tunnel * P_thermal;

    // Ensure the probability is within [0, 1]
    return std::clamp(total_prob, 0.0, 1.0);
}

double calculateZeroPointEnergyContribution(double hbar, double mass, double lattice_constant,
                                            double temperature)
{
    // ==== PHYSICALLY REALISTIC ZERO-POINT ENERGY CALCULATION ====
    // Based on lattice vibrations (phonons) in crystalline materials

    // Physical constants
    const double kB = 8.617333262e-5;  // Boltzmann constant in eV/K
    double thermal_energy = kB * temperature;

    // Typical Debye frequency for crystalline solids (~1e13 Hz)
    // This is much more realistic than the extreme frequencies calculated before
    double debye_frequency = 1.0e13;  // Hz

    // Convert to angular frequency
    double omega = 2.0 * M_PI * debye_frequency;  // rad/s

    // Calculate zero-point energy in SI units
    double hbar_SI = hbar * 1.602176634e-19;              // J·s
    double zero_point_energy_SI = 0.5 * hbar_SI * omega;  // J

    // Convert to eV
    double zero_point_energy_eV = zero_point_energy_SI / 1.602176634e-19;  // eV

    // ==== PHYSICALLY CORRECT TEMPERATURE SCALING ====
    // Zero-point energy contribution to displacement threshold
    // At low temperatures: quantum effects more significant
    // At high temperatures: classical thermal motion dominates

    double temperature_factor = 1.0;
    if (temperature > 0.0) {
        // Quantum-to-classical transition based on characteristic temperature
        double characteristic_temp = zero_point_energy_eV / kB;  // K

        // Temperature scaling based on quantum statistics
        if (temperature < characteristic_temp) {
            // Low temperature: quantum effects significant
            temperature_factor = characteristic_temp / temperature;
        }
        else {
            // High temperature: classical regime, quantum effects diminished
            temperature_factor = characteristic_temp / temperature;
        }
    }

    // Apply physical scaling - ZPE should be small correction (~0.01-0.1 eV)
    double zpe_contribution = zero_point_energy_eV * temperature_factor;

    // Ensure reasonable physical limits (ZPE corrections are typically small)
    return std::min(zpe_contribution, 0.5);  // Cap at 0.5 eV maximum
}

DefectDistribution applyQuantumFieldCorrections(const DefectDistribution& defects,
                                                const CrystalLattice& crystal,
                                                const QFTParameters& params, double temperature,
                                                const std::vector<ParticleType>& particle_types)
{
    // Create a copy of the input defect distribution
    DefectDistribution corrected = defects;

    // If no specific particle types requested, process all particles in the defects
    std::set<ParticleType> types_to_process;
    if (particle_types.empty()) {
        // Collect all particle types from the defects
        for (const auto& [type, _] : corrected.interstitials) {
            types_to_process.insert(type);
        }
        for (const auto& [type, _] : corrected.vacancies) {
            types_to_process.insert(type);
        }
        for (const auto& [type, _] : corrected.clusters) {
            types_to_process.insert(type);
        }
    }
    else {
        types_to_process.insert(particle_types.begin(), particle_types.end());
    }

    // Remove verbose debug output to prevent flooding during Monte Carlo tests
    // std::cout << "Applying quantum corrections to " << types_to_process.size()
    //           << " particle types..." << std::endl;

    // Process each particle type
    for (const auto& particle_type : types_to_process) {
        // Calculate quantum tunneling probability for this particle
        double tunneling_prob = calculateQuantumTunnelingProbability(
            crystal.barrier_height, temperature, params, particle_type);

        // Calculate zero-point energy contribution
        double zero_point = calculateZeroPointEnergyContribution(
            params.hbar, params.getMass(particle_type), crystal.lattice_constant, temperature);

        // Physical constants
        const double kB = 8.617333262e-5;  // Boltzmann constant in eV/K

        // Calculate enhancement factors based on quantum effects
        double interstitial_enhancement = 1.0 + 5.0 * tunneling_prob;
        double vacancy_enhancement = 1.0 + 3.0 * tunneling_prob;
        double cluster_enhancement = 1.0 + 2.0 * zero_point / crystal.barrier_height;

        // Apply physical limits based on energy conservation
        // Maximum enhancement limited by available thermal energy
        double max_thermal_enhancement = 1.0 + (kB * temperature) / crystal.barrier_height;
        interstitial_enhancement = std::min(interstitial_enhancement, max_thermal_enhancement);
        vacancy_enhancement = std::min(vacancy_enhancement, max_thermal_enhancement);
        cluster_enhancement = std::min(cluster_enhancement, max_thermal_enhancement);

        // Remove hardcoded minimums - let physics determine the actual values
        // If quantum effects are small, that's the real physics
        // No artificial floor values for "demonstration purposes"

        // Temperature-dependent scaling (quantum effects are stronger at lower temperatures)
        // Use configurable temperature threshold and scaling factor
        double temp_scale = 1.0;
        if (temperature < params.temperature_threshold) {
            temp_scale = 1.0 + (params.temperature_threshold - temperature) /
                                   params.temperature_scaling_factor;
        }

        // Remove per-particle debug output to prevent flooding during Monte Carlo tests
        // std::cout << "  Particle type " << static_cast<int>(particle_type)
        //           << ": interstitial enhancement = " << interstitial_enhancement
        //           << ", vacancy enhancement = " << vacancy_enhancement
        //           << ", cluster enhancement = " << cluster_enhancement << std::endl;

        // Apply enhancements to each region for this particle type
        auto it_interstitials = corrected.interstitials.find(particle_type);
        if (it_interstitials != corrected.interstitials.end()) {
            for (auto& value : it_interstitials->second) {
                value *= interstitial_enhancement * temp_scale;
            }
        }

        auto it_vacancies = corrected.vacancies.find(particle_type);
        if (it_vacancies != corrected.vacancies.end()) {
            for (auto& value : it_vacancies->second) {
                value *= vacancy_enhancement * temp_scale;
            }
        }

        auto it_clusters = corrected.clusters.find(particle_type);
        if (it_clusters != corrected.clusters.end()) {
            for (auto& value : it_clusters->second) {
                value *= cluster_enhancement * temp_scale;
            }
        }
    }

    return corrected;
}

// Optimized quantum field computation using SIMD
template <int Dimensions>
void QuantumField<Dimensions>::computeOptimized(
    const std::vector<std::complex<double>>& input_field,
    std::vector<std::complex<double>>& output_field) const
{
    // Use optimized computation when possible
    if (field_data_.size() >= 8 && input_field.size() == field_data_.size()) {
#ifdef __AVX2__
        this->computeSIMD(input_field, output_field);
#else
        this->computeStandard(input_field, output_field);
#endif
    }
    else {
        this->computeStandard(input_field, output_field);
    }
}

template <int Dimensions>
void QuantumField<Dimensions>::computeStandard(
    const std::vector<std::complex<double>>& input_field,
    std::vector<std::complex<double>>& output_field) const
{
    // Standard computation - fallback for small fields or non-SIMD systems
    const double dt = 0.01;  // Time step
    const double c = 1.0;    // Speed of light (normalized)

    for (size_t i = 0; i < field_data_.size(); ++i) {
        // Simple wave equation computation: d²φ/dt² = c²∇²φ
        std::complex<double> laplacian = computeLaplacian(input_field, i);
        output_field[i] = 2.0 * field_data_[i] - output_field[i] + c * c * dt * dt * laplacian;
    }
}

#ifdef __AVX2__
template <int Dimensions>
void QuantumField<Dimensions>::computeSIMD(const std::vector<std::complex<double>>& input_field,
                                           std::vector<std::complex<double>>& output_field) const
{
    // SIMD-optimized computation using AVX2
    const double dt = 0.01;
    const double c = 1.0;
    const double c2_dt2 = c * c * dt * dt;

    const size_t vector_size = 4;  // AVX2 complex double operations (8 doubles = 4 complex)
    const size_t n_vectors = field_data_.size() / vector_size;

    for (size_t v = 0; v < n_vectors; ++v) {
        // Load current field values
        __m256d current_real =
            _mm256_set_pd(field_data_[v * 4 + 3].real(), field_data_[v * 4 + 2].real(),
                          field_data_[v * 4 + 1].real(), field_data_[v * 4 + 0].real());
        __m256d current_imag =
            _mm256_set_pd(field_data_[v * 4 + 3].imag(), field_data_[v * 4 + 2].imag(),
                          field_data_[v * 4 + 1].imag(), field_data_[v * 4 + 0].imag());

        // Load previous field values (for second derivative)
        __m256d prev_real =
            _mm256_set_pd(output_field[v * 4 + 3].real(), output_field[v * 4 + 2].real(),
                          output_field[v * 4 + 1].real(), output_field[v * 4 + 0].real());
        __m256d prev_imag =
            _mm256_set_pd(output_field[v * 4 + 3].imag(), output_field[v * 4 + 2].imag(),
                          output_field[v * 4 + 1].imag(), output_field[v * 4 + 0].imag());

        // Compute Laplacian (simplified for performance)
        __m256d laplacian_real = _mm256_mul_pd(current_real, _mm256_set1_pd(-6.0));
        __m256d laplacian_imag = _mm256_mul_pd(current_imag, _mm256_set1_pd(-6.0));

        // Wave equation: φ_new = 2φ_current - φ_previous + c²dt²∇²φ_current
        __m256d two_current_real = _mm256_mul_pd(current_real, _mm256_set1_pd(2.0));
        __m256d two_current_imag = _mm256_mul_pd(current_imag, _mm256_set1_pd(2.0));
        __m256d c2_dt2_lap_real = _mm256_mul_pd(laplacian_real, _mm256_set1_pd(c2_dt2));
        __m256d c2_dt2_lap_imag = _mm256_mul_pd(laplacian_imag, _mm256_set1_pd(c2_dt2));

        __m256d new_real =
            _mm256_add_pd(_mm256_sub_pd(two_current_real, prev_real), c2_dt2_lap_real);
        __m256d new_imag =
            _mm256_add_pd(_mm256_sub_pd(two_current_imag, prev_imag), c2_dt2_lap_imag);

        // Store results
        double* real_ptr = reinterpret_cast<double*>(&new_real);
        double* imag_ptr = reinterpret_cast<double*>(&new_imag);

        for (size_t i = 0; i < 4; ++i) {
            output_field[v * 4 + i] = std::complex<double>(real_ptr[i], imag_ptr[i]);
        }
    }

    // Handle remaining elements with standard computation
    for (size_t i = n_vectors * vector_size; i < field_data_.size(); ++i) {
        output_field[i] = field_data_[i];  // Simplified for remaining elements
    }
}
#endif

// Explicit template instantiations
template class QuantumField<1>;
template class QuantumField<2>;
template class QuantumField<3>;

}  // namespace physics
}  // namespace rad_ml
