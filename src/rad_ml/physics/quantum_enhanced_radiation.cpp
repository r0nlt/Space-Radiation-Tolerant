/**
 * @file quantum_enhanced_radiation.cpp
 * @brief Implementation of quantum-enhanced radiation physics
 */

#include "rad_ml/physics/quantum_enhanced_radiation.hpp"

#include <algorithm>
#include <cstring>

#include "rad_ml/core/logger.hpp"

namespace rad_ml {
namespace physics {

QuantumEnhancedRadiation::QuantumEnhancedRadiation(const SemiconductorProperties& material)
    : material_(material), rng_(std::random_device{}())
{
    // Initialize QFT parameters based on semiconductor properties
    qft_params_.hbar = HBAR_EV_S;
    qft_params_.lattice_spacing = material_.lattice_constant_nm * 1e-9;  // Convert to meters
    qft_params_.temperature_threshold = material_.temperature_k;
    qft_params_.temperature_scaling_factor = 100.0;  // Kelvin

    // Set particle masses using the correct API (masses are already initialized in QFTParameters)
    // The masses map is already populated with correct values in the constructor
    // Add missing HeavyIon mass for our calculations
    qft_params_.masses[ParticleType::HeavyIon] =
        12000.0 * 1.782662e-36;  // Approximate heavy ion mass

    core::Logger::info("QuantumEnhancedRadiation initialized with " +
                       std::to_string(material_.bandgap_ev) + " eV bandgap");
}

double QuantumEnhancedRadiation::calculateQuantumChargeDeposition(double particle_energy,
                                                                  double let,
                                                                  ParticleType particle_type)
{
    // Step 1: Improved classical charge deposition with proper physics
    // Use Bethe-Bloch formula approximation for charge deposition
    double charge_per_energy = 0.278;  // fC per MeV (empirical for silicon)

    // Account for particle type differences
    double particle_factor = 1.0;
    switch (particle_type) {
        case ParticleType::Proton:
            particle_factor = 1.0;
            break;
        case ParticleType::Electron:
            particle_factor = 0.5;  // Electrons deposit less charge per unit energy
            break;
        case ParticleType::HeavyIon:
            particle_factor = 2.0;  // Heavy ions deposit more charge
            break;
        case ParticleType::Neutron:
            particle_factor = 0.3;  // Neutrons deposit charge via secondary reactions
            break;
        default:
            particle_factor = 1.0;
    }

    // Calculate classical charge deposition with proper LET scaling
    double classical_charge_fc = let * charge_per_energy * particle_factor;

    // Step 2: Apply conservative quantum corrections (1-5% maximum)
    CrystalLattice crystal;
    crystal.lattice_constant = material_.lattice_constant_nm * 1e-9;  // Convert to meters
    crystal.barrier_height = material_.bandgap_ev;

    // Use proper quantum field theory parameters
    QFTParameters qft_params;
    qft_params.hbar = 6.582119569e-16;  // eV·s
    qft_params.masses[particle_type] = getPhysicalMass(particle_type);
    qft_params.lattice_spacing = crystal.lattice_constant;

    // Calculate quantum tunneling probability with proper WKB approximation
    double barrier_width = 2.0e-10;  // 2 Angstroms in meters
    double mass = qft_params.masses[particle_type];
    double action = 2.0 * std::sqrt(2.0 * mass * crystal.barrier_height * 1.602176634e-19) *
                    barrier_width / (qft_params.hbar * 1.602176634e-19);

    double tunneling_prob = std::exp(-action);

    // Temperature-dependent quantum enhancement (Arrhenius behavior)
    double activation_energy = 0.1;  // eV
    double kB = 8.617333262e-5;      // eV/K
    double temp_factor = std::exp(-activation_energy / (kB * material_.temperature_k));

    // Conservative quantum enhancement (maximum 5%)
    double quantum_enhancement = std::min(0.05, tunneling_prob * temp_factor);

    double quantum_charge = classical_charge_fc * (1.0 + quantum_enhancement);

    // Step 3: Apply realistic charge collection efficiency
    double collection_efficiency =
        calculateQuantumCollectionEfficiency(quantum_charge, MemoryDeviceType::SRAM_6T);

    double final_charge = quantum_charge * collection_efficiency;

    core::Logger::debug("Improved quantum charge: " + std::to_string(classical_charge_fc) +
                        " fC (classical) -> " + std::to_string(final_charge) + " fC (quantum)");

    return final_charge;
}

// Helper function to get physical particle masses
double QuantumEnhancedRadiation::getPhysicalMass(ParticleType particle_type) const
{
    switch (particle_type) {
        case ParticleType::Proton:
            return 1.6726219e-27;  // kg
        case ParticleType::Electron:
            return 9.1093837e-31;  // kg
        case ParticleType::Neutron:
            return 1.6749275e-27;  // kg
        case ParticleType::HeavyIon:
            return 12.0 * 1.6726219e-27;  // Approximate for Carbon-12
        case ParticleType::Photon:
            return 1.0e-35;  // Effective mass for calculations
        default:
            return 1.6726219e-27;  // Default to proton mass
    }
}

double QuantumEnhancedRadiation::calculateTemperatureCriticalCharge(double base_critical_charge,
                                                                    double temperature)
{
    // Improved physics-based temperature dependence

    // 1. Classical temperature dependence (thermal activation)
    double activation_energy = 0.1;  // eV (typical for silicon defects)
    double kB = 8.617333262e-5;      // Boltzmann constant in eV/K
    double reference_temp = 300.0;   // Reference temperature (K)

    // Arrhenius temperature dependence
    double classical_factor =
        std::exp(activation_energy / kB * (1.0 / reference_temp - 1.0 / temperature));

    // 2. Quantum tunneling correction (small effect)
    double barrier_height = material_.bandgap_ev;
    double mass = getPhysicalMass(ParticleType::Electron);  // Use electron mass for tunneling

    // WKB tunneling probability
    double barrier_width = 2.0e-10;  // 2 Angstroms
    double hbar = 1.054571817e-34;   // J·s
    double action =
        2.0 * std::sqrt(2.0 * mass * barrier_height * 1.602176634e-19) * barrier_width / hbar;

    double tunneling_prob = std::exp(-action);

    // Quantum correction is small (< 1%)
    double quantum_correction = 1.0 - 0.01 * tunneling_prob * std::exp(-temperature / 100.0);

    // 3. Combine effects
    double corrected_charge = base_critical_charge * classical_factor * quantum_correction;

    // Physical bounds: don't go below 30% or above 300% of base value
    corrected_charge = std::max(corrected_charge, base_critical_charge * 0.3);
    corrected_charge = std::min(corrected_charge, base_critical_charge * 3.0);

    return corrected_charge;
}

double QuantumEnhancedRadiation::calculateDeviceSensitivity(MemoryDeviceType device_type,
                                                            double feature_size_nm)
{
    // Base sensitivity factors for different device types
    double base_sensitivity = 1.0;
    switch (device_type) {
        case MemoryDeviceType::SRAM_6T:
            base_sensitivity = 1.0;  // Baseline
            break;
        case MemoryDeviceType::SRAM_8T:
            base_sensitivity = 0.7;  // More robust
            break;
        case MemoryDeviceType::DRAM:
            base_sensitivity = 1.5;  // More sensitive due to capacitive storage
            break;
        case MemoryDeviceType::FLASH_SLC:
            base_sensitivity = 0.3;  // Less sensitive, non-volatile
            break;
        case MemoryDeviceType::FLASH_MLC:
            base_sensitivity = 0.8;  // More sensitive than SLC
            break;
        case MemoryDeviceType::MRAM:
            base_sensitivity = 0.1;  // Very robust, magnetic storage
            break;
        case MemoryDeviceType::FRAM:
            base_sensitivity = 0.2;  // Robust, ferroelectric storage
            break;
    }

    // Apply quantum size effects - smaller features are more sensitive
    // Use quantum confinement effects
    double quantum_size_factor = 1.0;
    if (feature_size_nm < 100.0) {
        // Calculate quantum confinement energy
        double confinement_energy = (HBAR_EV_S * HBAR_EV_S * M_PI * M_PI) /
                                    (2.0 * material_.effective_mass_ratio * 9.109e-31 *
                                     (feature_size_nm * 1e-9) * (feature_size_nm * 1e-9));

        // Convert to eV and normalize
        confinement_energy /= ELECTRON_CHARGE;

        // Higher confinement energy means more sensitivity
        quantum_size_factor = 1.0 + confinement_energy / material_.bandgap_ev;
    }

    return base_sensitivity * quantum_size_factor;
}

double QuantumEnhancedRadiation::calculateEnhancedBitFlipProbability(double deposited_charge,
                                                                     MemoryDeviceType device_type,
                                                                     double temperature)
{
    // Calculate temperature-corrected critical charge
    double device_critical_charge = material_.critical_charge_fc;

    // Adjust for device type
    switch (device_type) {
        case MemoryDeviceType::SRAM_6T:
            device_critical_charge *= 1.0;
            break;
        case MemoryDeviceType::SRAM_8T:
            device_critical_charge *= 1.4;  // Higher critical charge
            break;
        case MemoryDeviceType::DRAM:
            device_critical_charge *= 0.8;  // Lower critical charge
            break;
        case MemoryDeviceType::FLASH_SLC:
            device_critical_charge *= 5.0;  // Much higher critical charge
            break;
        case MemoryDeviceType::FLASH_MLC:
            device_critical_charge *= 2.0;
            break;
        case MemoryDeviceType::MRAM:
            device_critical_charge *= 10.0;  // Very high critical charge
            break;
        case MemoryDeviceType::FRAM:
            device_critical_charge *= 8.0;
            break;
    }

    double critical_charge =
        calculateTemperatureCriticalCharge(device_critical_charge, temperature);

    // Calculate probability using Weibull distribution (industry standard)
    if (deposited_charge <= 0.0 || critical_charge <= 0.0) {
        return 0.0;
    }

    double charge_ratio = deposited_charge / critical_charge;

    // Weibull parameters (typical for SEU cross-section curves)
    double shape_parameter = 2.0;  // Weibull shape
    double scale_parameter = 1.0;  // Normalized to critical charge

    // Weibull CDF: 1 - exp(-(x/λ)^k)
    double probability = 1.0 - std::exp(-std::pow(charge_ratio / scale_parameter, shape_parameter));

    // Apply quantum corrections for very small charges
    if (charge_ratio < 0.1) {
        // Use quantum tunneling probability for sub-threshold events
        double tunneling_prob = calculateQuantumTunnelingProbability(
            material_.bandgap_ev, temperature, qft_params_, ParticleType::Proton);
        probability += tunneling_prob * 0.1;  // Small quantum contribution
    }

    return std::clamp(probability, 0.0, 1.0);
}

uint32_t QuantumEnhancedRadiation::applyQuantumEnhancedRadiation(void* memory, size_t size_bytes,
                                                                 double particle_energy, double let,
                                                                 ParticleType particle_type,
                                                                 MemoryDeviceType device_type,
                                                                 uint32_t duration_ms)
{
    if (!memory || size_bytes == 0) return 0;

    // Calculate quantum-corrected charge deposition
    double deposited_charge = calculateQuantumChargeDeposition(particle_energy, let, particle_type);

    // Calculate bit flip probability
    double flip_probability =
        calculateEnhancedBitFlipProbability(deposited_charge, device_type, material_.temperature_k);

    // Determine number of bits to affect
    uint32_t mbu_size = calculateQuantumMBUSize(deposited_charge, particle_type);

    uint32_t total_flips = 0;
    uint8_t* byte_memory = static_cast<uint8_t*>(memory);

    // Apply radiation effects
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_int_distribution<size_t> byte_dist(0, size_bytes - 1);
    std::uniform_int_distribution<uint8_t> bit_dist(0, 7);

    // Number of potential impact sites scales with duration and flux
    uint32_t num_impacts = std::max(
        1u, static_cast<uint32_t>(flip_probability * size_bytes * (duration_ms / 1000.0) * 0.01));

    for (uint32_t impact = 0; impact < num_impacts; ++impact) {
        if (prob_dist(rng_) < flip_probability) {
            // Select impact location
            size_t byte_idx = byte_dist(rng_);
            uint8_t start_bit = bit_dist(rng_);

            // Apply MBU if calculated
            uint32_t bits_to_flip = std::min(mbu_size, static_cast<uint32_t>(8 - start_bit));

            for (uint32_t bit_offset = 0; bit_offset < bits_to_flip; ++bit_offset) {
                uint8_t bit_pos = start_bit + bit_offset;
                byte_memory[byte_idx] ^= (1u << bit_pos);
                total_flips++;
            }
        }
    }

    core::Logger::debug("Applied " + std::to_string(total_flips) +
                        " quantum-enhanced bit flips to " + std::to_string(size_bytes) + " bytes");

    return total_flips;
}

uint32_t QuantumEnhancedRadiation::calculateQuantumMBUSize(double deposited_charge,
                                                           ParticleType particle_type)
{
    // Use QFT defect clustering to determine MBU size
    // Higher charge deposition creates larger defect clusters

    double base_mbu_threshold = material_.critical_charge_fc * 2.0;  // 2x critical charge for MBU

    if (deposited_charge < base_mbu_threshold) {
        return 1;  // Single bit upset
    }

    // Calculate cluster size using quantum field correlation
    double cluster_factor = deposited_charge / base_mbu_threshold;

    // Apply particle-specific clustering
    double particle_clustering = 1.0;
    switch (particle_type) {
        case ParticleType::Proton:
            particle_clustering = 1.0;
            break;
        case ParticleType::Neutron:
            particle_clustering = 1.2;  // Slightly more clustering
            break;
        case ParticleType::HeavyIon:
            particle_clustering = 3.0;  // Highest clustering
            break;
        default:
            particle_clustering = 1.5;  // Default for other particles
            break;
    }

    // Calculate final MBU size
    uint32_t mbu_size = static_cast<uint32_t>(std::ceil(cluster_factor * particle_clustering));

    // Limit to reasonable range (1-8 bits)
    return std::clamp(mbu_size, 1u, 8u);
}

double QuantumEnhancedRadiation::calculateDefectFormationEnergy(double particle_energy,
                                                                ParticleType particle_type)
{
    // Use existing QFT framework to calculate defect formation
    double quantum_correction = calculateQuantumCorrectedDefectEnergy(
        material_.temperature_k, material_.bandgap_ev, qft_params_, particle_type);

    // Scale with particle energy
    double energy_factor = std::log10(particle_energy + 1.0) / 3.0;  // Logarithmic scaling

    return quantum_correction * energy_factor;
}

double QuantumEnhancedRadiation::calculateQuantumCollectionEfficiency(double deposited_charge,
                                                                      MemoryDeviceType device_type)
{
    // Base collection efficiency depends on device type
    double base_efficiency = 0.8;  // 80% baseline

    switch (device_type) {
        case MemoryDeviceType::SRAM_6T:
        case MemoryDeviceType::SRAM_8T:
            base_efficiency = 0.9;  // Good collection in SRAM
            break;
        case MemoryDeviceType::DRAM:
            base_efficiency = 0.7;  // Lower due to capacitive storage
            break;
        case MemoryDeviceType::FLASH_SLC:
        case MemoryDeviceType::FLASH_MLC:
            base_efficiency = 0.6;  // Lower due to floating gate
            break;
        case MemoryDeviceType::MRAM:
        case MemoryDeviceType::FRAM:
            base_efficiency = 0.95;  // Very good collection
            break;
    }

    // Apply quantum corrections for very small charges
    if (deposited_charge < material_.critical_charge_fc * 0.1) {
        // Quantum tunneling can enhance collection of small charges
        double tunneling_enhancement = calculateQuantumTunnelingProbability(
            material_.bandgap_ev * 0.5, material_.temperature_k, qft_params_, ParticleType::Proton);
        base_efficiency += tunneling_enhancement * 0.1;
    }

    return std::clamp(base_efficiency, 0.1, 1.0);
}

}  // namespace physics
}  // namespace rad_ml
