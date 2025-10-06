/**
 * Advanced Quantum Models Implementation
 *
 * Implementation of advanced theoretical physics models for radiation effects.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <numeric>
#include <rad_ml/physics/advanced_quantum_models.hpp>

namespace rad_ml {
namespace physics {

// ============================================================================
// Dirac Equation Solver Implementation
// ============================================================================

DiracEquationSolver::DiracMatrices::DiracMatrices()
{
    // Initialize 4x4 Dirac matrices in chiral representation
    gamma0.setZero(4, 4);
    gamma1.setZero(4, 4);
    gamma2.setZero(4, 4);
    gamma3.setZero(4, 4);
    gamma5.setZero(4, 4);

    // Standard chiral representation
    gamma0(0, 0) = 1.0;
    gamma0(1, 1) = 1.0;
    gamma0(2, 2) = -1.0;
    gamma0(3, 3) = -1.0;
    gamma1(0, 3) = 1.0;
    gamma1(1, 2) = 1.0;
    gamma1(2, 1) = -1.0;
    gamma1(3, 0) = -1.0;
    gamma2(0, 3) = std::complex<double>(0.0, -1.0);
    gamma2(1, 2) = std::complex<double>(0.0, 1.0);
    gamma2(2, 1) = std::complex<double>(0.0, 1.0);
    gamma2(3, 0) = std::complex<double>(0.0, -1.0);
    gamma3(0, 2) = 1.0;
    gamma3(1, 3) = -1.0;
    gamma3(2, 0) = -1.0;
    gamma3(3, 1) = 1.0;

    gamma5 = gamma0 * gamma1 * gamma2 * gamma3;
}

void DiracEquationSolver::initializeDiracMatrices()
{
    // Dirac matrices are initialized in constructor
}

DiracEquationSolver::ComplexVector DiracEquationSolver::solveDiracEquation(
    const Eigen::Vector3d& momentum, double energy, double mass)
{
    ComplexVector spinor(4);

    // Calculate relativistic gamma factor
    double p_squared = momentum.squaredNorm();
    double E_squared = energy * energy;
    double m_squared = mass * mass;

    // Check if solution exists (E² ≥ p²c² + m²c⁴)
    if (E_squared < p_squared + m_squared) {
        // No real solution - particle cannot exist
        spinor.setZero();
        return spinor;
    }

    // For positive energy solutions (particles)
    double gamma_factor = std::sqrt(E_squared - m_squared);

    // Construct Dirac spinor for positive energy state
    // ψ = (ϕ, χ) where ϕ and χ are 2-component spinors
    Eigen::Vector2cd phi, chi;

    // Assume spin up for simplicity (can be generalized)
    phi(0) = std::sqrt((energy + mass) / 2.0);  // Large component
    phi(1) = 0.0;                               // Spin down component

    chi(0) = (momentum(0) * phi(0)) / (energy + mass);  // Small component
    chi(1) = (momentum(1) * phi(0) + std::complex<double>(0.0, 1.0) * momentum(2) * phi(0)) /
             (energy + mass);

    // Combine into 4-component spinor
    spinor(0) = phi(0);  // ϕ₁
    spinor(1) = phi(1);  // ϕ₂
    spinor(2) = chi(0);  // χ₁
    spinor(3) = chi(1);  // χ₂

    return spinor;
}

double DiracEquationSolver::calculateRelativisticCrossSection(
    double incident_energy, double scattering_angle, const MaterialProperties& target_material)
{
    // Mott scattering formula for relativistic electrons
    // dσ/dΩ = (Zαℏc / (2E sin²(θ/2)) )² * (1 - β² sin²(θ/2))

    double Z = 14.0;               // Silicon atomic number (example)
    double alpha = 1.0 / 137.036;  // Fine structure constant
    double beta =
        std::sqrt(1.0 - ELECTRON_MASS_EV * ELECTRON_MASS_EV / (incident_energy + ELECTRON_MASS_EV) /
                            (incident_energy + ELECTRON_MASS_EV));

    double theta_half = scattering_angle / 2.0;
    double sin_theta_half = std::sin(theta_half);

    double prefactor =
        (Z * alpha * HBAR_C / (2.0 * incident_energy * sin_theta_half * sin_theta_half));
    double relativistic_factor = 1.0 - beta * beta * sin_theta_half * sin_theta_half;

    return prefactor * prefactor * relativistic_factor;
}

double DiracEquationSolver::calculateRelativisticDisplacementEnergy(double non_relativistic_energy,
                                                                    double electron_energy)
{
    // Relativistic correction factor
    double gamma = (electron_energy + ELECTRON_MASS_EV) / ELECTRON_MASS_EV;
    double beta_squared = 1.0 - 1.0 / (gamma * gamma);

    // Relativistic displacement energy correction
    // Based on relativistic kinematics for energy transfer
    double relativistic_factor = gamma * (1.0 - beta_squared / 3.0);

    return non_relativistic_energy * relativistic_factor;
}

// ============================================================================
// Bethe-Salpeter Equation Solver Implementation
// ============================================================================

BetheSalpeterSolver::BSEVector BetheSalpeterSolver::solveBetheSalpeter(
    const BSEMatrix& kernel, double energy_binding, const Eigen::Vector3d& total_momentum)
{
    // Solve eigenvalue problem: Γ = (1/(E_binding - K)) Γ
    // This is a simplified implementation - full BSE solution requires
    // sophisticated numerical methods (ladder approximation, etc.)

    BSEMatrix interaction_matrix =
        energy_binding * BSEMatrix::Identity(kernel.rows(), kernel.cols()) - kernel;

    // Use Eigen's eigenvalue solver (in practice, would use more sophisticated methods)
    Eigen::ComplexEigenSolver<BSEMatrix> solver(interaction_matrix);

    if (solver.info() != Eigen::Success) {
        BSEVector zero_vector(kernel.rows());
        zero_vector.setZero();
        return zero_vector;
    }

    // Find eigenvalue closest to zero (bound state condition)
    auto eigenvalues = solver.eigenvalues();
    int bound_state_index = 0;
    double min_eigenvalue_diff = std::abs(eigenvalues(0));

    for (int i = 1; i < eigenvalues.size(); ++i) {
        double diff = std::abs(eigenvalues(i));
        if (diff < min_eigenvalue_diff) {
            min_eigenvalue_diff = diff;
            bound_state_index = i;
        }
    }

    return solver.eigenvectors().col(bound_state_index);
}

double BetheSalpeterSolver::calculateClusterBindingEnergy(
    const std::vector<Eigen::Vector3d>& defect_positions,
    const std::vector<DefectType>& defect_types, const CrystalLattice& crystal_lattice,
    const QFTParameters& params)
{
    if (defect_positions.size() != defect_types.size()) {
        throw std::invalid_argument("Defect positions and types must have same size");
    }

    // Construct interaction kernel for this cluster
    BSEMatrix kernel = constructInteractionKernel(defect_positions, defect_types, crystal_lattice);

    // Solve BSE for bound state
    Eigen::Vector3d total_momentum = Eigen::Vector3d::Zero();  // At rest
    double binding_energy_guess = 0.1;                         // Initial guess in eV

    BSEVector bound_state_wavefunction =
        solveBetheSalpeter(kernel, binding_energy_guess, total_momentum);

    // Calculate actual binding energy from wavefunction normalization
    double binding_energy = 0.0;
    for (int i = 0; i < bound_state_wavefunction.size(); ++i) {
        binding_energy += std::norm(bound_state_wavefunction(i));
    }

    return binding_energy;
}

std::pair<double, BetheSalpeterSolver::BSEVector> BetheSalpeterSolver::calculateExcitonBinding(
    const Eigen::Vector3d& electron_position, const Eigen::Vector3d& hole_position,
    const CrystalLattice& crystal_lattice, double temperature)
{
    // Model electron-hole interaction as exciton using BSE
    std::vector<Eigen::Vector3d> positions = {electron_position, hole_position};
    std::vector<DefectType> types = {DefectType::ELECTRON_TRAP, DefectType::HOLE_TRAP};

    double binding_energy =
        calculateClusterBindingEnergy(positions, types, crystal_lattice, QFTParameters{});

    BSEMatrix kernel = constructInteractionKernel(positions, types, crystal_lattice);
    BSEVector exciton_wavefunction =
        solveBetheSalpeter(kernel, binding_energy, Eigen::Vector3d::Zero());

    return {binding_energy, exciton_wavefunction};
}

BetheSalpeterSolver::BSEMatrix BetheSalpeterSolver::constructInteractionKernel(
    const std::vector<Eigen::Vector3d>& positions, const std::vector<DefectType>& types,
    const CrystalLattice& lattice)
{
    int n_defects = positions.size();
    BSEMatrix kernel(n_defects, n_defects);

    for (int i = 0; i < n_defects; ++i) {
        for (int j = 0; j < n_defects; ++j) {
            double charge_i = (types[i] == DefectType::HOLE_TRAP) ? 1.0 : -1.0;
            double charge_j = (types[j] == DefectType::HOLE_TRAP) ? 1.0 : -1.0;

            kernel(i, j) = calculateCoulombInteraction(positions[i], positions[j], charge_i,
                                                       charge_j, lattice);
        }
    }

    return kernel;
}

double BetheSalpeterSolver::calculateCoulombInteraction(const Eigen::Vector3d& pos1,
                                                        const Eigen::Vector3d& pos2, double charge1,
                                                        double charge2,
                                                        const CrystalLattice& lattice)
{
    double distance = (pos1 - pos2).norm();
    double dielectric_screening =
        lattice.barrier_height;  // Use barrier height as proxy for screening

    // Screened Coulomb potential: V(r) = (q1*q2)/(4πεε_r r) * exp(-r/λ_D)
    double prefactor = (charge1 * charge2) / (4.0 * M_PI * 8.8541878128e-12 * dielectric_screening);
    double debye_length = 1.0;  // Debye screening length in nm (simplified)

    return prefactor * std::exp(-distance / debye_length) / distance;
}

// ============================================================================
// Green's Function Propagator Implementation
// ============================================================================

Eigen::VectorXcd GreensFunctionPropagator::solveHelmholtzEquation(
    const std::vector<Eigen::Vector3d>& source_positions, const Eigen::VectorXcd& source_amplitudes,
    double wave_number, const std::vector<Eigen::Vector3d>& field_points)
{
    int n_sources = source_positions.size();
    int n_points = field_points.size();

    if (n_sources != source_amplitudes.size()) {
        throw std::invalid_argument("Source positions and amplitudes must have same size");
    }

    Eigen::VectorXcd field_values(n_points);

    for (int i = 0; i < n_points; ++i) {
        std::complex<double> total_field = 0.0;

        for (int j = 0; j < n_sources; ++j) {
            std::complex<double> greens_function =
                freeSpaceGreensFunction(source_positions[j], field_points[i], wave_number);
            total_field += source_amplitudes(j) * greens_function;
        }

        field_values(i) = total_field;
    }

    return field_values;
}

Eigen::VectorXcd GreensFunctionPropagator::calculateRadiationField(
    const std::vector<Eigen::Vector3d>& particle_trajectory,
    const Eigen::Vector3d& particle_velocity,
    const std::vector<Eigen::Vector3d>& observation_points, double frequency)
{
    double wave_number = 2.0 * M_PI * frequency / 299792458.0;  // c = 3e8 m/s

    // Simplified radiation field calculation
    // In practice, would use Liénard-Wiechert potentials
    int n_points = observation_points.size();
    Eigen::VectorXcd radiation_field(n_points);

    for (int i = 0; i < n_points; ++i) {
        double retarded_time =
            calculateRetardedTime(particle_trajectory.back(), observation_points[i], 0.0);

        // Simplified radiation pattern (dipole approximation)
        Eigen::Vector3d direction =
            (observation_points[i] - particle_trajectory.back()).normalized();
        double cos_theta = direction.dot(particle_velocity.normalized());

        radiation_field(i) = std::complex<double>(cos_theta, 0.0) /
                             (observation_points[i] - particle_trajectory.back()).norm();
    }

    return radiation_field;
}

std::vector<DefectDistribution> GreensFunctionPropagator::propagateDefects(
    const DefectDistribution& initial_defect_distribution, const GreensMatrix& propagation_kernel,
    const std::vector<double>& time_points)
{
    std::vector<DefectDistribution> time_evolution;

    // For each time point, apply propagation kernel
    for (double time : time_points) {
        // Simplified propagation: exponential decay with kernel
        DefectDistribution propagated_defects;

        // Apply kernel to each defect type
        for (const auto& [particle_type, interstitials] :
             initial_defect_distribution.interstitials) {
            std::vector<double> propagated_interstitials;

            for (size_t i = 0; i < interstitials.size(); ++i) {
                double propagated_value = 0.0;

                for (size_t j = 0; j < interstitials.size(); ++j) {
                    propagated_value += propagation_kernel(i, j).real() * interstitials[j] *
                                        std::exp(-time / 1e-12);
                }

                propagated_interstitials.push_back(propagated_value);
            }

            propagated_defects.interstitials[particle_type] = propagated_interstitials;
        }

        time_evolution.push_back(propagated_defects);
    }

    return time_evolution;
}

GreensFunctionPropagator::GreensMatrix GreensFunctionPropagator::calculatePhononMediatedInteraction(
    const std::vector<Eigen::Vector3d>& defect_positions,
    const std::function<double(const Eigen::Vector3d&)>& phonon_dispersion, double temperature)
{
    int n_defects = defect_positions.size();
    GreensMatrix interaction_matrix(n_defects, n_defects);

    for (int i = 0; i < n_defects; ++i) {
        for (int j = 0; j < n_defects; ++j) {
            Eigen::Vector3d delta_r = defect_positions[i] - defect_positions[j];
            double distance = delta_r.norm();

            // Phonon-mediated interaction strength
            double phonon_frequency = phonon_dispersion(delta_r);
            double thermal_factor =
                1.0 / (1.0 - std::exp(-phonon_frequency / (8.617333262e-5 * temperature)));

            interaction_matrix(i, j) =
                thermal_factor / (distance * distance + 1e-10);  // Avoid singularity
        }
    }

    return interaction_matrix;
}

std::complex<double> GreensFunctionPropagator::freeSpaceGreensFunction(
    const Eigen::Vector3d& source_pos, const Eigen::Vector3d& field_pos, double wave_number)
{
    double distance = (field_pos - source_pos).norm();
    double phase = wave_number * distance;

    return std::exp(std::complex<double>(0.0, phase)) / (4.0 * M_PI * distance);
}

double GreensFunctionPropagator::calculateRetardedTime(const Eigen::Vector3d& source_pos,
                                                       const Eigen::Vector3d& field_pos,
                                                       double time)
{
    double distance = (field_pos - source_pos).norm();
    double speed_of_light = 299792458.0;  // m/s

    return time - distance / speed_of_light;
}

std::complex<double> GreensFunctionPropagator::integrateGreensFunction(
    const std::function<std::complex<double>(double)>& integrand, double a, double b,
    int num_points)
{
    // Simple trapezoidal integration
    double h = (b - a) / num_points;
    std::complex<double> integral = 0.0;

    for (int i = 0; i <= num_points; ++i) {
        double x = a + i * h;
        double weight = (i == 0 || i == num_points) ? 0.5 : 1.0;
        integral += weight * integrand(x);
    }

    return integral * h;
}

// ============================================================================
// Advanced Quantum Radiation Model Implementation
// ============================================================================

DefectDistribution AdvancedQuantumRadiationModel::calculateAdvancedRadiationEffects(
    const Particle& incident_particle, const MaterialProperties& target_material,
    const CrystalLattice& crystal_lattice, const RadiationEnvironment& radiation_environment)
{
    // Compose effects from Dirac cascade, BSE clusters, and Green's propagation
    DefectDistribution defects;

    // 1) Relativistic electron cascade
    std::vector<Particle> electron_cascade =
        simulateRelativisticElectronCascade(incident_particle, crystal_lattice, QFTParameters{});
    // Map cascade to an electron interstitial-like contribution (unit weights)
    std::vector<double> e_contrib(std::max<size_t>(1, electron_cascade.size()), 1.0);
    defects.interstitials[ParticleType::Electron] = e_contrib;

    // 2) Bound state formation (BSE)
    std::vector<DefectCluster> clusters =
        calculateDefectClusterFormation(defects, crystal_lattice, 300.0);
    defects.clusters[ParticleType::Electron] = std::vector<double>(clusters.size(), 1.0);

    // 3) Propagation (Green's functions)
    std::vector<DefectDistribution> evolution =
        propagateRadiationEffects(defects, crystal_lattice, {1e-15, 1e-14, 1e-13});

    return evolution.empty() ? defects : evolution.back();
}

std::vector<Particle> AdvancedQuantumRadiationModel::simulateRelativisticElectronCascade(
    const Particle& initial_electron, const CrystalLattice& lattice, const QFTParameters& params)
{
    std::vector<Particle> cascade;

    // Effective incident energy proxy (eV)
    double incident_energy_eV = 1.0e4;
    auto it_m = params.masses.find(ParticleType::Electron);
    if (it_m != params.masses.end() && std::isfinite(it_m->second) && it_m->second > 0.0) {
        double scale = 5.0e3 * (9.11e-31 / it_m->second);
        if (scale < 2.0e3) scale = 2.0e3;
        if (scale > 5.0e4) scale = 5.0e4;
        incident_energy_eV = scale;
    }

    // Synthesize material properties from lattice (proxy) for cross-sections
    MaterialProperties material{};
    material.band_gap = 1.12;
    material.electron_effective_mass = 0.26;
    material.hole_effective_mass = 0.37;
    material.dielectric_constant = 11.7;
    material.phonon_frequency = 15.3;
    material.deformation_potential = 9.0;
    material.lattice_thermal_conductivity = 148.0;
    if (std::isfinite(lattice.barrier_height) && lattice.barrier_height > 0.0) {
        double eps = lattice.barrier_height * 8.0;
        if (eps < 2.0) eps = 2.0;
        if (eps > 20.0) eps = 20.0;
        material.dielectric_constant = eps;
    }

    // Deterministic angular sampling
    std::vector<double> angles;
    angles.reserve(16);
    for (int i = 1; i <= 16; ++i) {
        double theta = 0.05 + (1.20 - 0.05) * (static_cast<double>(i) - 0.5) / 16.0;
        angles.push_back(theta);
    }

    // Cross-section weights via Dirac solver
    std::vector<double> weights;
    weights.reserve(angles.size());
    double sum_w = 0.0;
    for (double th : angles) {
        double w =
            dirac_solver_.calculateRelativisticCrossSection(incident_energy_eV, th, material);
        if (!std::isfinite(w) || w < 0.0) w = 0.0;
        weights.push_back(w);
        sum_w += w;
    }

    // Expected total yield based on energy and material scale (pair-creation heuristic)
    const double pair_e = std::max(3.0 * material.band_gap, 3.6);
    const double screening = 1.0 / (1.0 + material.dielectric_constant / 12.0);
    const double phonon_att = 1.0 - std::min(0.5, material.phonon_frequency / 60.0);
    const double scaling = 0.004;
    double expected_f = (incident_energy_eV / pair_e) * scaling * screening * phonon_att;
    size_t expected = static_cast<size_t>(std::floor(expected_f + 0.5));
    if (expected < 2) expected = 2;
    if (expected > 20) expected = 20;

    if (sum_w <= 0.0) {
        for (size_t i = 0; i < expected; ++i) {
            cascade.emplace_back(ParticleType::Electron, initial_electron.mass(), -1.0, 0.5);
        }
        return cascade;
    }

    // Proportional allocation by weights, summing to expected
    std::vector<size_t> counts(angles.size(), 0);
    size_t allocated = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        double w = weights[i] / sum_w;
        size_t n = static_cast<size_t>(std::floor(w * static_cast<double>(expected)));
        counts[i] = n;
        allocated += n;
    }
    // Greedy remainder by descending weight
    std::vector<size_t> idx(weights.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return weights[a] > weights[b]; });
    size_t remaining = (allocated >= expected) ? 0 : (expected - allocated);
    for (size_t k = 0; k < idx.size() && remaining > 0; ++k) {
        counts[idx[k]] += 1;
        --remaining;
    }

    for (size_t i = 0; i < counts.size(); ++i) {
        for (size_t n = 0; n < counts[i]; ++n) {
            cascade.emplace_back(ParticleType::Electron, initial_electron.mass(), -1.0, 0.5);
        }
    }

    return cascade;
}

std::vector<DefectCluster> AdvancedQuantumRadiationModel::calculateDefectClusterFormation(
    const DefectDistribution& initial_defects, const CrystalLattice& lattice, double temperature)
{
    std::vector<DefectCluster> clusters;

    // Example: form exciton-like bound states between electrons and holes
    Eigen::Vector3d electron_pos(1.0, 0.0, 0.0);
    Eigen::Vector3d hole_pos(0.0, 1.0, 0.0);

    auto [binding_energy, wavefunction] =
        bse_solver_.calculateExcitonBinding(electron_pos, hole_pos, lattice, temperature);

    if (binding_energy > 0.01) {  // Threshold for bound state formation
        DefectCluster exciton_cluster;
        exciton_cluster.defect_positions = {electron_pos, hole_pos};
        exciton_cluster.defect_types = {DefectType::ELECTRON_TRAP, DefectType::HOLE_TRAP};
        exciton_cluster.binding_energy = binding_energy;
        exciton_cluster.total_charge = 0.0;  // Neutral exciton

        clusters.push_back(exciton_cluster);
    }

    return clusters;
}

std::vector<DefectDistribution> AdvancedQuantumRadiationModel::propagateRadiationEffects(
    const DefectDistribution& initial_distribution, const CrystalLattice& lattice,
    const std::vector<double>& time_steps)
{
    // Use Green's function propagator for time evolution
    GreensFunctionPropagator::GreensMatrix propagation_kernel(10, 10);
    propagation_kernel.setIdentity();  // Simplified kernel

    return greens_propagator_.propagateDefects(initial_distribution, propagation_kernel,
                                               time_steps);
}

}  // namespace physics
}  // namespace rad_ml
