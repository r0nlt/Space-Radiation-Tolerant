/**
 * @file quantum_enhanced_radiation.hpp
 * @brief Bridge between QFT framework and realistic semiconductor radiation physics
 *
 * This connects the existing QFT implementation to actual physical processes
 * that occur when radiation hits semiconductor devices.
 */

#pragma once

#include <cmath>
#include <random>

#include "rad_ml/physics/quantum_field_theory.hpp"
#include "rad_ml/radiation/environment.hpp"
#include "rad_ml/utils/bit_manipulation.hpp"

namespace rad_ml {
namespace physics {

/**
 * @brief Memory device types with different quantum properties
 */
enum class MemoryDeviceType {
    SRAM_6T,    // 6-transistor SRAM cell
    SRAM_8T,    // 8-transistor SRAM cell (more robust)
    DRAM,       // Dynamic RAM
    FLASH_SLC,  // Single-level cell Flash
    FLASH_MLC,  // Multi-level cell Flash
    MRAM,       // Magnetoresistive RAM
    FRAM        // Ferroelectric RAM
};

/**
 * @brief Semiconductor material properties for quantum calculations
 */
struct SemiconductorProperties {
    double bandgap_ev = 1.12;            // Silicon bandgap at 300K
    double effective_mass_ratio = 0.26;  // Electron effective mass / free electron mass
    double dielectric_constant = 11.7;   // Silicon relative permittivity
    double lattice_constant_nm = 0.543;  // Silicon lattice constant
    double critical_charge_fc = 15.0;    // Critical charge for bit flip (femtocoulombs)
    double temperature_k = 300.0;        // Operating temperature
};

/**
 * @brief Quantum-enhanced radiation effect calculator
 *
 * Uses QFT to calculate realistic charge deposition and defect formation
 * when cosmic rays interact with semiconductor devices.
 */
class QuantumEnhancedRadiation {
   private:
    SemiconductorProperties material_;
    QFTParameters qft_params_;
    std::mt19937 rng_;

    // Physical constants
    static constexpr double ELECTRON_CHARGE = 1.602176634e-19;  // Coulombs
    static constexpr double BOLTZMANN_K = 8.617333262e-5;       // eV/K
    static constexpr double HBAR_EV_S = 6.582119569e-16;        // eV⋅s

   public:
    QuantumEnhancedRadiation(const SemiconductorProperties& material = {});

    /**
     * @brief Calculate quantum-corrected charge deposition
     *
     * Uses QFT defect calculations to determine how much charge is actually
     * deposited when a particle hits the semiconductor.
     *
     * @param particle_energy Energy in MeV
     * @param let Linear Energy Transfer in MeV⋅cm²/mg
     * @param particle_type Type of incident particle
     * @return Deposited charge in femtocoulombs
     */
    double calculateQuantumChargeDeposition(double particle_energy, double let,
                                            ParticleType particle_type);

    /**
     * @brief Calculate temperature-dependent critical charge
     *
     * Uses quantum tunneling probability to adjust critical charge based on temperature.
     * At lower temperatures, quantum effects make bit flips easier.
     *
     * @param base_critical_charge Base critical charge at room temperature (fC)
     * @param temperature Temperature in Kelvin
     * @return Temperature-corrected critical charge (fC)
     */
    double calculateTemperatureCriticalCharge(double base_critical_charge, double temperature);

    /**
     * @brief Calculate device-specific sensitivity using QFT
     *
     * Different memory types (SRAM, DRAM, Flash) have different quantum properties
     * that affect their radiation sensitivity.
     *
     * @param device_type Type of memory device
     * @param feature_size_nm Technology node size in nanometers
     * @return Sensitivity factor (1.0 = baseline, >1.0 = more sensitive)
     */
    double calculateDeviceSensitivity(MemoryDeviceType device_type, double feature_size_nm);

    /**
     * @brief Enhanced bit flip probability calculation
     *
     * Combines classical radiation physics with quantum corrections from QFT.
     *
     * @param deposited_charge Charge deposited by particle (fC)
     * @param device_type Type of memory device
     * @param temperature Operating temperature (K)
     * @return Probability of bit flip (0.0 to 1.0)
     */
    double calculateEnhancedBitFlipProbability(double deposited_charge,
                                               MemoryDeviceType device_type, double temperature);

    /**
     * @brief Apply quantum-enhanced radiation effects to memory
     *
     * This replaces the simple bit flip functions with physics-based calculations.
     *
     * @param memory Pointer to memory to corrupt
     * @param size_bytes Size of memory region
     * @param particle_energy Incident particle energy (MeV)
     * @param let Linear Energy Transfer (MeV⋅cm²/mg)
     * @param particle_type Type of incident particle
     * @param device_type Type of memory device
     * @param duration_ms Exposure duration
     * @return Number of bit flips applied
     */
    uint32_t applyQuantumEnhancedRadiation(void* memory, size_t size_bytes, double particle_energy,
                                           double let, ParticleType particle_type,
                                           MemoryDeviceType device_type, uint32_t duration_ms);

    /**
     * @brief Calculate multi-bit upset probability using quantum clustering
     *
     * Uses QFT defect clustering to determine if multiple adjacent bits
     * will be affected by a single particle.
     *
     * @param deposited_charge Total charge deposited (fC)
     * @param particle_type Type of incident particle
     * @return Number of bits likely to be affected (1-8)
     */
    uint32_t calculateQuantumMBUSize(double deposited_charge, ParticleType particle_type);

   private:
    /**
     * @brief Calculate quantum defect formation energy
     *
     * Uses the existing QFT framework to calculate how particle impacts
     * create defects in the crystal lattice.
     */
    double calculateDefectFormationEnergy(double particle_energy, ParticleType particle_type);

    /**
     * @brief Calculate charge collection efficiency with quantum corrections
     *
     * Not all deposited charge is collected - quantum effects determine
     * how much actually reaches the sensitive node.
     */
    double calculateQuantumCollectionEfficiency(double deposited_charge,
                                                MemoryDeviceType device_type);
};

}  // namespace physics
}  // namespace rad_ml
