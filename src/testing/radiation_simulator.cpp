#include "../../include/rad_ml/testing/radiation_simulator.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace rad_ml {
namespace testing {

// Additional enums and structs for enhanced LET calculations
enum class ParticleType { Proton, Electron, HeavyIon, Photon };

enum class MaterialType { Silicon, GaAs, SiC };

enum class ErrorType { SINGLE_BIT, MULTI_BIT, BLOCK };

// Enhanced radiation event structure
struct RadiationEvent {
    double let;                // Linear Energy Transfer (MeV-cm²/mg)
    double energy;             // Particle energy (MeV)
    double angle;              // Impact angle (radians)
    size_t location;           // Memory location
    double error_probability;  // Probability of causing error
    ErrorType error_type;      // Type of error caused
    double error_magnitude;    // Magnitude of error
};

// Enhanced environment parameters
struct EnvironmentParams {
    double altitude_km = 400.0;
    double solar_activity = 1.0;
    bool inside_saa = false;
    double shielding_thickness_mm = 5.0;
    double temperature_K = 300.0;
    bool use_physics_based_let = true;  // Toggle between Bethe-Bloch and empirical
};

// Enhanced LET calculation constants
struct LETConstants {
    // Physical constants
    static constexpr double ELECTRON_REST_MASS_MEV = 0.511;         // MeV
    static constexpr double PROTON_REST_MASS_MEV = 938.3;           // MeV
    static constexpr double CLASSICAL_ELECTRON_RADIUS = 2.818e-13;  // cm
    static constexpr double AVOGADRO = 6.022e23;

    // Material properties (Silicon as default)
    static constexpr double SILICON_DENSITY = 2.33;  // g/cm³
    static constexpr double SILICON_Z = 14.0;        // Atomic number
    static constexpr double SILICON_A = 28.09;       // Atomic mass
    static constexpr double SILICON_I = 173.0e-6;    // Mean excitation energy (MeV)
};

// Enhanced RadiationSimulator class
class EnhancedRadiationSimulator {
   public:
    // Material properties structure
    struct MaterialProperties {
        double Z;        // Atomic number
        double A;        // Atomic mass
        double density;  // Density (g/cm³)
        double I;        // Mean excitation energy (MeV)
    };

   private:
    EnvironmentParams params_;
    std::mt19937 rng_;
    std::lognormal_distribution<double> let_dist_;
    std::gamma_distribution<double> flux_dist_;
    std::weibull_distribution<double> energy_dist_;
    std::uniform_real_distribution<double> angle_dist_;

   public:
    EnhancedRadiationSimulator(const EnvironmentParams& params);

    // Enhanced LET calculation methods
    double calculateLET(ParticleType type, double energy_MeV, MaterialType material) const;
    double calculateBetheBlochLET(ParticleType type, double energy_MeV,
                                  MaterialType material) const;
    double calculateEmpiricalLET(ParticleType type, double energy_MeV, MaterialType material) const;

    // Material and particle property methods
    MaterialProperties getMaterialProperties(MaterialType material) const;
    double getParticleRestMass(ParticleType type) const;
    double getParticleCharge(ParticleType type) const;

    // Physics correction methods
    double calculateDensityCorrection(double beta, const MaterialProperties& mat_props) const;
    double calculateShellCorrection(double energy_MeV, ParticleType type,
                                    const MaterialProperties& mat_props) const;

    // Simulation methods
    void initializeDistributions();
    double calculateBaseFlux() const;
    double calculateMeanLET() const;
    double calculateLETSigma() const;
    double calculateFluxShape() const;
    double calculateFluxScale(double base_flux) const;
    double calculateEnergyShape() const;
    double calculateEnergyScale() const;

    std::vector<RadiationEvent> simulateEffects(const uint8_t* memory, size_t size,
                                                std::chrono::milliseconds duration);

    ParticleType sampleParticleType() const;
    size_t calculateImpactLocation(size_t memory_size) const;
    double calculateErrorProbability(double let, double energy) const;
    ErrorType determineErrorType(double let) const;
    double calculateErrorMagnitude(double let, double energy) const;
};

EnhancedRadiationSimulator::EnhancedRadiationSimulator(const EnvironmentParams& params)
    : params_(params)
{
    // Initialize random number generators
    std::random_device rd;
    rng_ = std::mt19937(rd());

    // Initialize distributions based on environment parameters
    initializeDistributions();
}

void EnhancedRadiationSimulator::initializeDistributions()
{
    // Calculate base radiation flux based on altitude and solar activity
    double base_flux = calculateBaseFlux();

    // Initialize LET distribution (MeV-cm²/mg)
    let_dist_ =
        std::lognormal_distribution<double>(std::log(calculateMeanLET()), calculateLETSigma());

    // Initialize flux distribution (particles/cm²/s)
    flux_dist_ =
        std::gamma_distribution<double>(calculateFluxShape(), calculateFluxScale(base_flux));

    // Initialize energy distribution (MeV)
    energy_dist_ =
        std::weibull_distribution<double>(calculateEnergyShape(), calculateEnergyScale());

    // Initialize angle distribution (radians)
    angle_dist_ = std::uniform_real_distribution<double>(0, 2 * M_PI);
}

double EnhancedRadiationSimulator::calculateBaseFlux() const
{
    // Base flux calculation based on altitude and solar activity
    double altitude_factor = std::exp(-params_.altitude_km / 1000.0);
    double solar_factor = 1.0 + (params_.solar_activity - 1.0) * 0.2;

    // Adjust for South Atlantic Anomaly
    double saa_factor = params_.inside_saa ? 10.0 : 1.0;

    // Base flux in particles/cm²/s
    return 1.0e4 * altitude_factor * solar_factor * saa_factor;
}

// Enhanced LET calculation with multiple approaches
double EnhancedRadiationSimulator::calculateLET(ParticleType type, double energy_MeV,
                                                MaterialType material) const
{
    // Choose calculation method based on accuracy requirements
    if (params_.use_physics_based_let) {
        return calculateBetheBlochLET(type, energy_MeV, material);
    }
    else {
        return calculateEmpiricalLET(type, energy_MeV, material);
    }
}

double EnhancedRadiationSimulator::calculateBetheBlochLET(ParticleType type, double energy_MeV,
                                                          MaterialType material) const
{
    // Get material properties
    MaterialProperties mat_props = getMaterialProperties(material);

    // Calculate relativistic parameters
    double rest_mass = getParticleRestMass(type);
    double gamma = 1.0 + energy_MeV / rest_mass;
    double beta_squared = 1.0 - 1.0 / (gamma * gamma);
    double beta = std::sqrt(beta_squared);

    // Particle charge (in units of elementary charge)
    double z = getParticleCharge(type);

    // Bethe-Bloch formula components
    double K = 4.0 * M_PI * LETConstants::CLASSICAL_ELECTRON_RADIUS *
               LETConstants::CLASSICAL_ELECTRON_RADIUS * LETConstants::ELECTRON_REST_MASS_MEV *
               LETConstants::AVOGADRO;

    double density_correction = calculateDensityCorrection(beta, mat_props);
    double shell_correction = calculateShellCorrection(energy_MeV, type, mat_props);

    // Maximum energy transfer
    double T_max = (2.0 * LETConstants::ELECTRON_REST_MASS_MEV * beta_squared * gamma * gamma) /
                   (1.0 + 2.0 * gamma * LETConstants::ELECTRON_REST_MASS_MEV / rest_mass +
                    (LETConstants::ELECTRON_REST_MASS_MEV / rest_mass) *
                        (LETConstants::ELECTRON_REST_MASS_MEV / rest_mass));

    // Bethe-Bloch formula
    double dE_dx = K * (z * z) * (mat_props.Z / mat_props.A) * (mat_props.density / beta_squared) *
                   (0.5 * std::log(2.0 * LETConstants::ELECTRON_REST_MASS_MEV * beta_squared *
                                   gamma * gamma * T_max / (mat_props.I * mat_props.I)) -
                    beta_squared - density_correction - shell_correction);

    // Convert from MeV·cm²/g to MeV·cm²/mg
    return dE_dx / 1000.0;
}

double EnhancedRadiationSimulator::calculateEmpiricalLET(ParticleType type, double energy_MeV,
                                                         MaterialType material) const
{
    // Enhanced empirical model with material-specific coefficients
    MaterialProperties mat_props = getMaterialProperties(material);

    double base_let = 0.0;

    switch (type) {
        case ParticleType::Proton:
            // Improved proton LET with energy dependence
            if (energy_MeV < 10.0) {
                base_let = 15.0 * std::pow(energy_MeV, -0.8);  // Low energy enhancement
            }
            else {
                base_let = 3.0 * std::pow(energy_MeV, -0.5);  // Original scaling
            }
            break;

        case ParticleType::Electron:
            // Enhanced electron model with threshold
            if (energy_MeV > 0.5) {
                base_let = 0.2 * std::log(energy_MeV + 1.0);
            }
            else {
                base_let = 0.1 * energy_MeV;
            }
            break;

        case ParticleType::HeavyIon:
            // Z² dependence for heavy ions
            double effective_z = 26.0;  // Iron-56 as reference
            base_let = 0.5 * effective_z * effective_z * std::pow(energy_MeV, -0.3);
            break;

        default:
            base_let = 1.0 * energy_MeV * 0.1;
    }

    // Material correction factor
    double material_factor =
        std::sqrt(mat_props.Z / mat_props.A) * (mat_props.density / LETConstants::SILICON_DENSITY);

    return base_let * material_factor;
}

EnhancedRadiationSimulator::MaterialProperties EnhancedRadiationSimulator::getMaterialProperties(
    MaterialType material) const
{
    MaterialProperties props;

    switch (material) {
        case MaterialType::Silicon:
            props.Z = LETConstants::SILICON_Z;
            props.A = LETConstants::SILICON_A;
            props.density = LETConstants::SILICON_DENSITY;
            props.I = LETConstants::SILICON_I;
            break;

        case MaterialType::GaAs:
            props.Z = (31.0 + 33.0) / 2.0;  // Average Z
            props.A = (69.7 + 74.9) / 2.0;  // Average A
            props.density = 5.32;
            props.I = 384.0e-6;  // MeV
            break;

        case MaterialType::SiC:
            props.Z = (14.0 + 6.0) / 2.0;   // Average Z
            props.A = (28.1 + 12.0) / 2.0;  // Average A
            props.density = 3.21;
            props.I = 116.0e-6;  // MeV
            break;

        default:
            // Default to Silicon
            props.Z = LETConstants::SILICON_Z;
            props.A = LETConstants::SILICON_A;
            props.density = LETConstants::SILICON_DENSITY;
            props.I = LETConstants::SILICON_I;
    }

    return props;
}

double EnhancedRadiationSimulator::getParticleRestMass(ParticleType type) const
{
    switch (type) {
        case ParticleType::Proton:
            return LETConstants::PROTON_REST_MASS_MEV;
        case ParticleType::Electron:
            return LETConstants::ELECTRON_REST_MASS_MEV;
        case ParticleType::HeavyIon:
            return 52.0 * LETConstants::PROTON_REST_MASS_MEV;  // Iron-56 approximation
        default:
            return LETConstants::PROTON_REST_MASS_MEV;
    }
}

double EnhancedRadiationSimulator::getParticleCharge(ParticleType type) const
{
    switch (type) {
        case ParticleType::Proton:
            return 1.0;
        case ParticleType::Electron:
            return 1.0;  // Magnitude of charge
        case ParticleType::HeavyIon:
            return 26.0;  // Iron-56 charge
        default:
            return 1.0;
    }
}

double EnhancedRadiationSimulator::calculateDensityCorrection(
    double beta, const MaterialProperties& mat_props) const
{
    // Simplified density correction (Sternheimer parameterization)
    double plasma_energy =
        28.8e-6 * std::sqrt(mat_props.density * mat_props.Z / mat_props.A);  // MeV
    double gamma = 1.0 / std::sqrt(1.0 - beta * beta);

    if (beta < 0.1) {
        return 0.0;  // Negligible at low energies
    }

    double delta = std::log(plasma_energy / mat_props.I) + std::log(beta * gamma) - 0.5;
    return std::max(0.0, delta);
}

double EnhancedRadiationSimulator::calculateShellCorrection(
    double energy_MeV, ParticleType type, const MaterialProperties& mat_props) const
{
    // Simplified shell correction
    double velocity_ratio =
        std::sqrt(2.0 * energy_MeV / getParticleRestMass(type)) / 137.0;  // v/c in atomic units

    if (velocity_ratio > 1.0) {
        return 0.0;  // Negligible at high velocities
    }

    // Barkas correction approximation
    double z = getParticleCharge(type);
    return z * velocity_ratio * 0.1;  // Simplified correction
}

double EnhancedRadiationSimulator::calculateMeanLET() const
{
    // Enhanced mean LET calculation using physics-based approach
    double base_let = 10.0;  // Base LET in MeV-cm²/mg

    if (params_.use_physics_based_let) {
        // Calculate weighted average LET for typical space environment
        double proton_contribution =
            calculateLET(ParticleType::Proton, 30.0, MaterialType::Silicon) * 0.7;
        double electron_contribution =
            calculateLET(ParticleType::Electron, 2.0, MaterialType::Silicon) * 0.25;
        double heavy_ion_contribution =
            calculateLET(ParticleType::HeavyIon, 500.0, MaterialType::Silicon) * 0.05;

        base_let = proton_contribution + electron_contribution + heavy_ion_contribution;
    }

    // Adjust for altitude (higher altitude = higher LET)
    double altitude_factor = 1.0 + (params_.altitude_km / 1000.0) * 0.1;

    // Adjust for shielding
    double shielding_factor = std::exp(-params_.shielding_thickness_mm / 10.0);

    return base_let * altitude_factor * shielding_factor;
}

double EnhancedRadiationSimulator::calculateLETSigma() const
{
    // LET distribution width
    return 0.5;  // Log-normal distribution sigma
}

double EnhancedRadiationSimulator::calculateFluxShape() const
{
    // Shape parameter for gamma distribution
    return 2.0;
}

double EnhancedRadiationSimulator::calculateFluxScale(double base_flux) const
{
    // Scale parameter for gamma distribution
    return base_flux / calculateFluxShape();
}

double EnhancedRadiationSimulator::calculateEnergyShape() const
{
    // Shape parameter for Weibull distribution
    return 1.5;
}

double EnhancedRadiationSimulator::calculateEnergyScale() const
{
    // Scale parameter for Weibull distribution
    return 50.0;  // MeV
}

std::vector<RadiationEvent> EnhancedRadiationSimulator::simulateEffects(
    const uint8_t* memory, size_t size, std::chrono::milliseconds duration)
{
    std::vector<RadiationEvent> events;

    // Calculate number of particles based on flux and duration
    double flux = flux_dist_(rng_);
    double area = 1.0;                        // cm²
    double time = duration.count() / 1000.0;  // seconds
    int num_particles = static_cast<int>(flux * area * time);

    // Generate radiation events
    for (int i = 0; i < num_particles; ++i) {
        RadiationEvent event;

        // Generate particle properties
        event.energy = energy_dist_(rng_);
        event.angle = angle_dist_(rng_);

        // Determine particle type based on environment
        ParticleType particle_type = sampleParticleType();

        // Calculate LET using enhanced method
        event.let = calculateLET(particle_type, event.energy, MaterialType::Silicon);

        // Calculate impact location
        event.location = calculateImpactLocation(size);

        // Calculate error probability based on LET and energy
        event.error_probability = calculateErrorProbability(event.let, event.energy);

        // Determine if error occurs
        if (std::uniform_real_distribution<double>(0, 1)(rng_) < event.error_probability) {
            event.error_type = determineErrorType(event.let);
            event.error_magnitude = calculateErrorMagnitude(event.let, event.energy);
            events.push_back(event);
        }
    }

    return events;
}

ParticleType EnhancedRadiationSimulator::sampleParticleType() const
{
    // Sample particle type based on typical space environment
    std::uniform_real_distribution<double> type_dist(0, 1);
    double rand_val = type_dist(rng_);

    if (rand_val < 0.7) {
        return ParticleType::Proton;
    }
    else if (rand_val < 0.95) {
        return ParticleType::Electron;
    }
    else {
        return ParticleType::HeavyIon;
    }
}

size_t EnhancedRadiationSimulator::calculateImpactLocation(size_t memory_size) const
{
    // Calculate impact location in memory
    std::uniform_int_distribution<size_t> loc_dist(0, memory_size - 1);
    return loc_dist(rng_);
}

double EnhancedRadiationSimulator::calculateErrorProbability(double let, double energy) const
{
    // Enhanced error probability calculation
    double let_factor = 1.0 - std::exp(-let / 20.0);  // Sigmoid-like response
    double energy_factor = std::tanh(energy / 50.0);  // Saturation at high energies

    // Base probability with material sensitivity
    double base_prob = 0.05;

    // Adjust for shielding
    double shielding_factor = std::exp(-params_.shielding_thickness_mm / 15.0);

    // Temperature dependence (Arrhenius-like)
    double temp_factor = std::exp(-0.1 * (params_.temperature_K - 300.0) / 300.0);

    return base_prob * let_factor * energy_factor * shielding_factor * temp_factor;
}

ErrorType EnhancedRadiationSimulator::determineErrorType(double let) const
{
    // Enhanced error type determination with probabilistic approach
    std::uniform_real_distribution<double> error_dist(0, 1);
    double rand_val = error_dist(rng_);

    if (let < 10.0) {
        return ErrorType::SINGLE_BIT;
    }
    else if (let < 30.0) {
        // Mixed single/multi-bit errors
        return (rand_val < 0.7) ? ErrorType::SINGLE_BIT : ErrorType::MULTI_BIT;
    }
    else if (let < 80.0) {
        return ErrorType::MULTI_BIT;
    }
    else {
        // High LET can cause block errors
        return (rand_val < 0.6) ? ErrorType::MULTI_BIT : ErrorType::BLOCK;
    }
}

double EnhancedRadiationSimulator::calculateErrorMagnitude(double let, double energy) const
{
    // Enhanced error magnitude calculation
    double base_magnitude = 1.0;

    // LET contribution (logarithmic scaling)
    double let_factor = std::log10(let + 1.0) / 2.0;

    // Energy contribution (square root scaling)
    double energy_factor = std::sqrt(energy) / 10.0;

    // Charge collection efficiency
    double collection_efficiency = 1.0 - std::exp(-let / 50.0);

    return base_magnitude * let_factor * energy_factor * collection_efficiency;
}

}  // namespace testing
}  // namespace rad_ml
