/**
 * Enhanced Physics Radiation Simulator
 *
 * This file integrates advanced quantum models into the existing radiation framework:
 * 1. Dirac equation for relativistic electron effects in semiconductors
 * 2. Bethe-Salpeter equation for bound state formation in defect clusters
 * 3. Green's function methods for sophisticated propagation modeling
 *
 * Enhanced Features:
 * - Relativistic electron cascade modeling using Dirac equation
 * - Defect cluster formation using Bethe-Salpeter equation
 * - Advanced propagation using Green's functions
 * - Integration with existing neural network protection system
 */

#pragma once

#include <rad_ml/neural/protected_neural_network.hpp>
#include <rad_ml/physics/advanced_quantum_models.hpp>
#include <rad_ml/physics/quantum_field_theory.hpp>
#include <rad_ml/sim/mission_environment.hpp>
#include <rad_ml/sim/physics_radiation_simulator.hpp>

// Forward declarations for types used in interface
namespace rad_ml {
namespace physics {
class Particle;
enum class ParticleType;
struct DefectCluster;
enum class DefectType;
struct DefectDistribution;
struct MaterialProperties;
struct CrystalLattice;
class AdvancedQuantumRadiationModel;
class DiracEquationSolver;
class BetheSalpeterSolver;
class GreensFunctionPropagator;
}  // namespace physics

namespace neural {
template <typename T>
class ProtectedNeuralNetwork;
}  // namespace neural

namespace sim {
struct EnvironmentParams;
}  // namespace sim
}  // namespace rad_ml

namespace rad_ml {
namespace sim {

/**
 * @brief Enhanced radiation simulator using advanced quantum models
 */
class EnhancedPhysicsRadiationSimulator : public PhysicsRadiationSimulator {
   public:
    /**
     * @brief Initialize enhanced simulator with advanced quantum models
     */
    EnhancedPhysicsRadiationSimulator();

    /**
     * @brief Initialize enhanced simulator with specific environment parameters
     *
     * @param params Environment parameters for radiation simulation
     */
    explicit EnhancedPhysicsRadiationSimulator(const EnvironmentParams& params);

    /**
     * @brief Calculate relativistic electron cascade using Dirac equation
     *
     * @param incident_particle Initial particle causing radiation damage
     * @param target_material Semiconductor material properties
     * @param crystal_lattice Crystal structure information
     * @return Vector of secondary electrons generated
     */
    std::vector<rad_ml::physics::Particle> calculateRelativisticElectronCascade(
        const rad_ml::physics::Particle& incident_particle,
        const rad_ml::physics::MaterialProperties& target_material,
        const rad_ml::physics::CrystalLattice& crystal_lattice);

    /**
     * @brief Calculate relativistic electron cascade with explicit incident energy
     *
     * @param incident_particle Initial particle causing radiation damage
     * @param target_material Semiconductor material properties
     * @param crystal_lattice Crystal structure information
     * @param incident_energy_eV Incident particle energy in eV
     * @return Vector of secondary electrons generated
     */
    std::vector<rad_ml::physics::Particle> calculateRelativisticElectronCascade(
        const rad_ml::physics::Particle& incident_particle,
        const rad_ml::physics::MaterialProperties& target_material,
        const rad_ml::physics::CrystalLattice& crystal_lattice, double incident_energy_eV);

    /**
     * @brief Calculate defect cluster formation using Bethe-Salpeter equation
     *
     * @param initial_defects Initial defect distribution
     * @param crystal_lattice Crystal structure
     * @param temperature Operating temperature in Kelvin
     * @return Vector of formed defect clusters with binding energies
     */
    std::vector<rad_ml::physics::DefectCluster> calculateDefectClusterFormation(
        const rad_ml::physics::DefectDistribution& initial_defects,
        const rad_ml::physics::CrystalLattice& crystal_lattice, double temperature);

    /**
     * @brief Propagate radiation effects using Green's function methods
     *
     * @param initial_distribution Initial defect distribution
     * @param crystal_lattice Crystal structure
     * @param time_steps Vector of time points for evolution
     * @return Time-evolved defect distributions
     */
    std::vector<rad_ml::physics::DefectDistribution> propagateRadiationEffects(
        const rad_ml::physics::DefectDistribution& initial_distribution,
        const rad_ml::physics::CrystalLattice& crystal_lattice,
        const std::vector<double>& time_steps);

    /**
     * @brief Enhanced radiation effect calculation using all theoretical methods
     *
     * @param incident_particle Incident radiation particle
     * @param target_material Semiconductor material properties
     * @param crystal_lattice Crystal structure
     * @param radiation_environment Radiation environment conditions
     * @return Complete defect distribution with advanced quantum corrections
     */
    rad_ml::physics::DefectDistribution calculateEnhancedRadiationEffects(
        const rad_ml::physics::Particle& incident_particle,
        const rad_ml::physics::MaterialProperties& target_material,
        const rad_ml::physics::CrystalLattice& crystal_lattice,
        const rad_ml::sim::RadiationEnvironment& radiation_environment);

    /**
     * @brief Integrate advanced quantum effects into neural network protection
     *
     * @param network Neural network to protect
     * @param radiation_level Current radiation level
     * @param material_properties Semiconductor material properties
     * @param crystal_lattice Crystal structure
     * @return Enhanced protection recommendations
     */
    template <typename T>
    std::vector<std::string> enhanceNeuralNetworkProtection(
        rad_ml::neural::ProtectedNeuralNetwork<T>& network, double radiation_level,
        const rad_ml::physics::MaterialProperties& material_properties,
        const rad_ml::physics::CrystalLattice& crystal_lattice);

    /**
     * @brief Calculate quantum-enhanced displacement energy
     *
     * @param non_relativistic_energy Standard displacement energy
     * @param particle_energy Incident particle energy
     * @param particle_type Type of incident particle
     * @return Relativistic displacement energy
     */
    double calculateQuantumEnhancedDisplacementEnergy(double non_relativistic_energy,
                                                      double particle_energy,
                                                      rad_ml::physics::ParticleType particle_type);

    /**
     * @brief Advanced phonon-mediated defect interaction modeling
     *
     * @param defect_positions Positions of defects
     * @param crystal_lattice Crystal structure
     * @param temperature Temperature for thermal effects
     * @return Phonon-mediated interaction matrix
     */
    Eigen::MatrixXd calculatePhononMediatedInteractions(
        const std::vector<Eigen::Vector3d>& defect_positions,
        const rad_ml::physics::CrystalLattice& crystal_lattice, double temperature);

    /**
     * @brief Get access to advanced quantum models for direct manipulation
     */
    rad_ml::physics::AdvancedQuantumRadiationModel& getAdvancedModel() { return advanced_model_; }

    /**
     * @brief Create material properties for specific semiconductor (public for testing)
     */
    rad_ml::physics::MaterialProperties createMaterialProperties(
        const std::string& material_name) const;

    /**
     * @brief Holistic framework integration - combine enhanced physics with neural protection
     *
     * @param network Neural network to protect with enhanced physics models
     * @param radiation_environment Current radiation environment
     * @param material_properties Semiconductor material properties
     * @param crystal_lattice Crystal structure information
     * @return Comprehensive protection recommendations integrating all framework components
     */
    template <typename T>
    std::vector<std::string> performHolisticFrameworkIntegration(
        rad_ml::neural::ProtectedNeuralNetwork<T>& network,
        const rad_ml::sim::RadiationEnvironment& radiation_environment,
        const rad_ml::physics::MaterialProperties& material_properties,
        const rad_ml::physics::CrystalLattice& crystal_lattice);

    /**
     * @brief Apply enhanced physics corrections to neural network weights
     *
     * @param weights Neural network weights to correct
     * @param radiation_level Current radiation intensity
     * @param material_properties Semiconductor material properties
     * @return Physics-corrected weights with enhanced reliability
     */
    template <typename T>
    std::vector<std::vector<T>> applyEnhancedPhysicsCorrections(
        const std::vector<std::vector<T>>& weights, double radiation_level,
        const rad_ml::physics::MaterialProperties& material_properties);

    /**
     * @brief Generate comprehensive radiation-aware training recommendations
     *
     * @param mission_duration Expected mission duration in days
     * @param radiation_environments Vector of expected radiation environments
     * @return Training recommendations optimized for radiation environments
     */
    std::vector<std::string> generateRadiationAwareTrainingRecommendations(
        double mission_duration,
        const std::vector<rad_ml::sim::RadiationEnvironment>& radiation_environments) const;

    /**
     * @brief Calculate total radiation intensity from environment parameters
     *
     * @param env Radiation environment to analyze
     * @return Total radiation intensity combining all particle types
     */
    double calculateRadiationIntensity(const rad_ml::sim::RadiationEnvironment& env) const;

   private:
    rad_ml::physics::AdvancedQuantumRadiationModel advanced_model_;
    rad_ml::physics::DiracEquationSolver dirac_solver_;
    rad_ml::physics::BetheSalpeterSolver bse_solver_;
    rad_ml::physics::GreensFunctionPropagator greens_propagator_;
    rad_ml::physics::Particle
        incident_particle_;  // Store the current incident particle for calculations

    /**
     * @brief Convert framework particle to advanced quantum particle
     */
    rad_ml::physics::Particle convertToAdvancedParticle(
        const rad_ml::physics::Particle& framework_particle);

    /**
     * @brief Convert advanced quantum defects to framework defects
     */
    rad_ml::physics::DefectDistribution convertToFrameworkDefects(
        const rad_ml::physics::DefectDistribution& advanced_defects);

    /**
     * @brief Calculate relativistic scattering cross-section for specific particle-material
     * combination
     */
    double calculateRelativisticScatteringCrossSection(
        const rad_ml::physics::Particle& incident_particle,
        const rad_ml::physics::MaterialProperties& material, double scattering_angle);
};

/**
 * @brief Enhanced neural network protection using advanced quantum models
 */
class QuantumEnhancedNeuralProtection {
   public:
    /**
     * @brief Apply advanced quantum protection to neural network layer
     *
     * @param layer_weights Layer weights to protect
     * @param radiation_level Current radiation intensity
     * @param material_properties Semiconductor material properties
     * @param crystal_lattice Crystal structure
     * @return Protection-enhanced weights with quantum corrections
     */
    template <typename T>
    std::vector<std::vector<T>> applyQuantumEnhancedProtection(
        const std::vector<std::vector<T>>& layer_weights, double radiation_level,
        const rad_ml::physics::MaterialProperties& material_properties,
        const rad_ml::physics::CrystalLattice& crystal_lattice);

    /**
     * @brief Calculate quantum-enhanced error correction for neural network
     *
     * @param corrupted_weights Weights affected by radiation
     * @param original_weights Original uncorrupted weights
     * @param radiation_environment Radiation environment conditions
     * @return Error-corrected weights using advanced quantum methods
     */
    template <typename T>
    std::vector<std::vector<T>> calculateQuantumErrorCorrection(
        const std::vector<std::vector<T>>& corrupted_weights,
        const std::vector<std::vector<T>>& original_weights,
        const rad_ml::sim::RadiationEnvironment& radiation_environment);

    /**
     * @brief Predict neural network degradation using advanced quantum models
     *
     * @param network Neural network to analyze
     * @param mission_duration Mission duration in days
     * @param radiation_environment Target radiation environment
     * @param material_properties Semiconductor material properties
     * @return Predicted performance degradation over mission lifetime
     */
    template <typename T>
    std::vector<double> predictNeuralDegradation(
        const rad_ml::neural::ProtectedNeuralNetwork<T>& network, double mission_duration,
        const rad_ml::sim::RadiationEnvironment& radiation_environment,
        const rad_ml::physics::MaterialProperties& material_properties);

   private:
    EnhancedPhysicsRadiationSimulator enhanced_simulator_;
};

}  // namespace sim
}  // namespace rad_ml
