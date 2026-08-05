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

QuantumEnhancedRadiation::QuantumEnhancedRadiation(const SemiconductorProperties& material,
                                                   const CircuitProperties& circuit)
    : material_(material), circuit_(circuit), rng_(std::random_device{}())
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

    core::Logger::debug("QuantumEnhancedRadiation initialized with " +
                        std::to_string(material_.bandgap_ev) + " eV bandgap");
}

double QuantumEnhancedRadiation::calculateCircuitCriticalCharge(MemoryDeviceType device_type,
                                                                double temperature_k) const
{
    if (!std::isfinite(temperature_k) || temperature_k <= 0.0 ||
        !std::isfinite(circuit_.feature_size_nm) || circuit_.feature_size_nm <= 0.0 ||
        !std::isfinite(circuit_.reference_temperature_k) ||
        circuit_.reference_temperature_k <= 0.0 ||
        !std::isfinite(circuit_.qcrit_temperature_coefficient_per_k)) {
        throw std::invalid_argument("Circuit critical-charge inputs must be finite and positive");
    }

    using Model = CriticalChargePowerLaw<double>;
    static const Model sram_model =
        Model::fit({{180.0, 6.70}, {130.0, 3.30}});

    double critical_charge_fc = material_.critical_charge_fc;
    switch (device_type) {
        case MemoryDeviceType::SRAM_6T:
            critical_charge_fc = sram_model.predict(circuit_.feature_size_nm);
            break;
        case MemoryDeviceType::SRAM_8T:
            critical_charge_fc = 1.4 * sram_model.predict(circuit_.feature_size_nm);
            break;
        case MemoryDeviceType::DRAM:
            critical_charge_fc *= 0.8;
            break;
        case MemoryDeviceType::FLASH_SLC:
            critical_charge_fc *= 5.0;
            break;
        case MemoryDeviceType::FLASH_MLC:
            critical_charge_fc *= 2.0;
            break;
        case MemoryDeviceType::MRAM:
            critical_charge_fc *= 10.0;
            break;
        case MemoryDeviceType::FRAM:
            critical_charge_fc *= 8.0;
            break;
    }

    const double temperature_factor =
        1.0 + circuit_.qcrit_temperature_coefficient_per_k *
                  (temperature_k - circuit_.reference_temperature_k);
    if (!std::isfinite(temperature_factor) || temperature_factor <= 0.0) {
        throw std::invalid_argument("Circuit Qcrit temperature correction must stay positive");
    }
    return critical_charge_fc * temperature_factor;
}

double QuantumEnhancedRadiation::calculateClassicalChargeDeposition(double particle_energy,
                                                                    double let) const
{
    if (!std::isfinite(particle_energy) || !std::isfinite(let) || particle_energy <= 0.0 ||
        let <= 0.0 || material_.density_g_cm3 <= 0.0 ||
        material_.sensitive_depth_um <= 0.0 || material_.pair_creation_energy_ev <= 0.0) {
        return 0.0;
    }

    // LET [MeV cm²/mg] × density [mg/cm³] × path [cm] = deposited energy [MeV].
    // The deposited energy cannot exceed the incident particle energy. Dividing
    // by the mean pair-creation energy gives electron-hole pairs; multiplying
    // by elementary charge converts them to generated charge.
    constexpr double micrometers_to_cm = 1.0e-4;
    constexpr double grams_to_milligrams = 1.0e3;
    constexpr double mev_to_ev = 1.0e6;
    constexpr double coulombs_to_femtocoulombs = 1.0e15;

    const double density_mg_cm3 = material_.density_g_cm3 * grams_to_milligrams;
    const double path_cm = material_.sensitive_depth_um * micrometers_to_cm;
    const double let_deposited_energy_mev = let * density_mg_cm3 * path_cm;
    const double deposited_energy_mev = std::min(let_deposited_energy_mev, particle_energy);
    const double electron_hole_pairs =
        deposited_energy_mev * mev_to_ev / material_.pair_creation_energy_ev;
    return electron_hole_pairs * ELECTRON_CHARGE * coulombs_to_femtocoulombs;
}

double QuantumEnhancedRadiation::calculateQuantumChargeDeposition(double particle_energy,
                                                                  double let,
                                                                  ParticleType particle_type)
{
    // Step 1: Establish the dimensionally consistent classical baseline.
    const double classical_charge_fc = calculateClassicalChargeDeposition(particle_energy, let);
    if (classical_charge_fc == 0.0) {
        return 0.0;
    }

    // Step 2: Apply quantum corrections using existing QFT framework
    CrystalLattice crystal;
    crystal.lattice_constant = material_.lattice_constant_nm * 1e-9;  // Convert nm to meters
    crystal.barrier_height = material_.bandgap_ev;  // Use bandgap as barrier height

    // Create defect distribution from particle impact
    DefectDistribution defects;
    defects.interstitials[particle_type] = {classical_charge_fc *
                                            0.3};                    // 30% creates interstitials
    defects.vacancies[particle_type] = {classical_charge_fc * 0.5};  // 50% creates vacancies
    defects.clusters[particle_type] = {classical_charge_fc * 0.2};   // 20% creates clusters

    // Apply quantum field corrections
    DefectDistribution quantum_corrected = applyQuantumFieldCorrections(
        defects, crystal, qft_params_, material_.temperature_k, {particle_type});

    // Calculate total quantum-corrected charge
    double quantum_charge = 0.0;
    if (quantum_corrected.interstitials.count(particle_type)) {
        for (double charge : quantum_corrected.interstitials[particle_type]) {
            quantum_charge += charge;
        }
    }
    if (quantum_corrected.vacancies.count(particle_type)) {
        for (double charge : quantum_corrected.vacancies[particle_type]) {
            quantum_charge += charge;
        }
    }
    if (quantum_corrected.clusters.count(particle_type)) {
        for (double charge : quantum_corrected.clusters[particle_type]) {
            quantum_charge += charge;
        }
    }

    // Step 3: Apply charge collection efficiency
    double collection_efficiency = calculateQuantumCollectionEfficiency(
        quantum_charge, MemoryDeviceType::SRAM_6T);  // Default to SRAM

    double final_charge = quantum_charge * collection_efficiency;

    core::Logger::debug("Quantum charge deposition: " + std::to_string(classical_charge_fc) +
                        " fC (classical) -> " + std::to_string(final_charge) + " fC (quantum)");

    return final_charge;
}

double QuantumEnhancedRadiation::calculateTemperatureCriticalCharge(double base_critical_charge,
                                                                    double temperature)
{
    // Use quantum tunneling probability to adjust critical charge
    double barrier_height = material_.bandgap_ev;
    double tunneling_prob = calculateQuantumTunnelingProbability(barrier_height, temperature,
                                                                 qft_params_, ParticleType::Proton);

    // At lower temperatures, quantum tunneling makes bit flips easier
    // So critical charge decreases
    double temperature_factor = 1.0 - 0.3 * tunneling_prob;

    // Also apply classical temperature dependence
    double classical_factor = 1.0 + (temperature - 300.0) / 1000.0;  // Weak temperature dependence

    double corrected_charge = base_critical_charge * temperature_factor * classical_factor;

    return std::max(corrected_charge, base_critical_charge * 0.5);  // Don't go below 50% of base
}

double QuantumEnhancedRadiation::calculateDeviceSensitivity(MemoryDeviceType device_type,
                                                            double feature_size_nm)
{
    if (!std::isfinite(feature_size_nm) || feature_size_nm <= 0.0) {
        return 0.0;
    }

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
        // Calculate particle-in-a-box confinement energy in SI units, then
        // convert joules to eV. HBAR_EV_S cannot be mixed directly with kg/m.
        constexpr double hbar_joule_seconds = HBAR_EV_S * ELECTRON_CHARGE;
        constexpr double electron_mass_kg = 9.1093837015e-31;
        const double feature_size_m = feature_size_nm * 1e-9;
        const double confinement_energy_j =
            (hbar_joule_seconds * hbar_joule_seconds * M_PI * M_PI) /
            (2.0 * material_.effective_mass_ratio * electron_mass_kg * feature_size_m *
             feature_size_m);
        const double confinement_energy = confinement_energy_j / ELECTRON_CHARGE;

        // Higher confinement energy means more sensitivity
        quantum_size_factor = 1.0 + confinement_energy / material_.bandgap_ev;
    }

    return base_sensitivity * quantum_size_factor;
}

double QuantumEnhancedRadiation::calculateEnhancedBitFlipProbability(double deposited_charge,
                                                                     MemoryDeviceType device_type,
                                                                     double temperature)
{
    // Qcrit is determined only by circuit/process inputs. Particle energy and
    // LET have already acted through deposited_charge and do not modify Qcrit.
    const double critical_charge = calculateCircuitCriticalCharge(device_type, temperature);

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

    const double base_mbu_threshold =
        calculateCircuitCriticalCharge(MemoryDeviceType::SRAM_6T, material_.temperature_k) * 2.0;

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
    const double critical_charge =
        calculateCircuitCriticalCharge(device_type, material_.temperature_k);
    if (deposited_charge < critical_charge * 0.1) {
        // Quantum tunneling can enhance collection of small charges
        double tunneling_enhancement = calculateQuantumTunnelingProbability(
            material_.bandgap_ev * 0.5, material_.temperature_k, qft_params_, ParticleType::Proton);
        base_efficiency += tunneling_enhancement * 0.1;
    }

    return std::clamp(base_efficiency, 0.1, 1.0);
}

}  // namespace physics
}  // namespace rad_ml
