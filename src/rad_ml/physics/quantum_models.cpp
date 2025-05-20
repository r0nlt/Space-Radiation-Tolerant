/**
 * Implementation of Quantum Models
 *
 * This file implements functions declared in quantum_models.hpp
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <rad_ml/physics/quantum_field_theory.hpp>
#include <rad_ml/physics/quantum_models.hpp>
#include <random>

namespace rad_ml {
namespace physics {

double calculateQuantumDecoherence(const DefectDistribution& defects, double temperature,
                                   const ExtendedQFTParameters& params, ParticleType particle_type)
{
    // Simple decoherence model based on temperature and defect concentration
    double total_defects = 0.0;

    // Get defect distribution for this particle type
    auto it_interstitials = defects.interstitials.find(particle_type);
    if (it_interstitials != defects.interstitials.end()) {
        total_defects +=
            std::accumulate(it_interstitials->second.begin(), it_interstitials->second.end(), 0.0);
    }

    auto it_vacancies = defects.vacancies.find(particle_type);
    if (it_vacancies != defects.vacancies.end()) {
        total_defects +=
            std::accumulate(it_vacancies->second.begin(), it_vacancies->second.end(), 0.0);
    }

    auto it_clusters = defects.clusters.find(particle_type);
    if (it_clusters != defects.clusters.end()) {
        total_defects +=
            std::accumulate(it_clusters->second.begin(), it_clusters->second.end(), 0.0);
    }

    // Get particle-specific decoherence rate and dissipation coefficient
    double decoherence_rate = params.getDecoherenceRate(particle_type);
    double dissipation_coefficient = params.getDissipationCoefficient(particle_type);

    // Decoherence rate increases with temperature and defect concentration
    return decoherence_rate * (1.0 + temperature / 300.0) *
           (1.0 + total_defects * dissipation_coefficient);
}

double calculateQuantumTransitionProbability(double incident_energy, double temperature,
                                             const QFTParameters& params,
                                             ParticleType particle_type)
{
    // Simplified model for quantum transition probability
    // Higher probabilities at low temperatures and high incident energies
    const double kB = 8.617333262e-5;  // Boltzmann constant in eV/K
    double thermal_energy = kB * temperature;

    // Get particle-specific mass
    double mass = params.getMass(particle_type);

    // Calculate transition probability using quantum mechanics principles
    double transition_prob =
        1.0 - std::exp(-incident_energy / (thermal_energy + params.hbar * 1e15));

    // Particle-specific adjustments
    if (particle_type == ParticleType::Photon) {
        // Photons have different transition probabilities due to their zero rest mass
        transition_prob = std::max(0.0, transition_prob * 1.5);
    }
    else if (particle_type == ParticleType::Electron) {
        // Electrons have higher transition probabilities due to their small mass
        transition_prob = std::min(1.0, transition_prob * 1.2);
    }

    // Bound the result to [0, 1]
    return std::clamp(transition_prob, 0.0, 1.0);
}

double calculateDisplacementEnergy(const CrystalLattice& crystal, const QFTParameters& params,
                                   ParticleType particle_type)
{
    // Base displacement energy depends on lattice type
    double base_energy = 0.0;
    switch (crystal.type) {
        case CrystalLattice::Type::FCC:
            base_energy = 15.0 + 2.5 * crystal.lattice_constant;  // eV
            break;
        case CrystalLattice::Type::BCC:
            base_energy = 10.0 + 3.0 * crystal.lattice_constant;  // eV
            break;
        case CrystalLattice::Type::DIAMOND:
            base_energy = 20.0 + 4.0 * crystal.lattice_constant;  // eV
            break;
        default:
            base_energy = 25.0;  // Default value
    }

    // Get mass for the specific particle type
    double mass = params.getMass(particle_type);

    // Adjust displacement energy based on particle type
    if (particle_type == ParticleType::Electron) {
        // Electrons have lower displacement energies due to their small mass
        base_energy *= 0.5;
    }
    else if (particle_type == ParticleType::Proton || particle_type == ParticleType::HeavyIon) {
        // Protons and heavy ions have higher displacement energies
        base_energy *= 1.5;
    }

    // Apply quantum correction
    double quantum_correction =
        calculateZeroPointEnergyContribution(params.hbar, mass, crystal.lattice_constant, 300.0);

    // Apply scaling correction to fix energy magnitude (1.0e6 converts from MeV to eV range)
    return (base_energy + quantum_correction) * 1.0e6;
}

DefectDistribution simulateDisplacementCascade(const CrystalLattice& crystal, double pka_energy,
                                               const QFTParameters& params,
                                               double displacement_energy,
                                               ParticleType particle_type)
{
    // Initialize defect distribution
    DefectDistribution defects;

    // Simple model for defect production:
    // Number of defects scales with PKA energy and inversely with displacement energy
    if (pka_energy > displacement_energy) {
        double defect_count = std::floor(0.8 * pka_energy / displacement_energy);

        // Distribute defects among different types
        // Spatial distribution follows cascade morphology
        double vacancy_fraction = 0.6;
        double interstitial_fraction = 0.3;
        double cluster_fraction = 0.1;

        // Adjust fractions based on particle type
        if (particle_type == ParticleType::Electron) {
            // Electrons produce more point defects and fewer clusters
            vacancy_fraction = 0.7;
            interstitial_fraction = 0.25;
            cluster_fraction = 0.05;
        }
        else if (particle_type == ParticleType::HeavyIon) {
            // Heavy ions produce more clusters
            vacancy_fraction = 0.5;
            interstitial_fraction = 0.2;
            cluster_fraction = 0.3;
        }

        // Clear any existing data for this particle type
        defects.interstitials[particle_type].clear();
        defects.vacancies[particle_type].clear();
        defects.clusters[particle_type].clear();

        // Region 1 (core)
        defects.vacancies[particle_type].push_back(defect_count * vacancy_fraction * 0.6);
        defects.interstitials[particle_type].push_back(defect_count * interstitial_fraction * 0.4);
        defects.clusters[particle_type].push_back(defect_count * cluster_fraction * 0.7);

        // Region 2 (intermediate)
        defects.vacancies[particle_type].push_back(defect_count * vacancy_fraction * 0.3);
        defects.interstitials[particle_type].push_back(defect_count * interstitial_fraction * 0.4);
        defects.clusters[particle_type].push_back(defect_count * cluster_fraction * 0.2);

        // Region 3 (periphery)
        defects.vacancies[particle_type].push_back(defect_count * vacancy_fraction * 0.1);
        defects.interstitials[particle_type].push_back(defect_count * interstitial_fraction * 0.2);
        defects.clusters[particle_type].push_back(defect_count * cluster_fraction * 0.1);
    }

    return defects;
}

std::unique_ptr<QuantumField<3>> createParticleField(const std::vector<int>& grid_dimensions,
                                                     double lattice_spacing,
                                                     ParticleType particle_type,
                                                     const QFTParameters& params)
{
    // Create the appropriate field for the particle type
    auto field = std::make_unique<QuantumField<3>>(grid_dimensions, lattice_spacing, particle_type);

    // Initialize field appropriately based on particle type
    switch (particle_type) {
        case ParticleType::Photon:
            // Photons typically have wave-like characteristics
            field->initializeCoherentState(1.0, 0.0);
            break;
        case ParticleType::Electron:
        case ParticleType::Proton:
            // Charged particles often have Gaussian distributions
            field->initializeGaussian(0.0, 0.5);
            break;
        default:
            // Default initialization for other particles
            field->initializeGaussian(0.0, 1.0);
            break;
    }

    return field;
}

std::vector<double> simulateMultiParticleInteraction(
    std::vector<std::reference_wrapper<QuantumField<3>>> fields, const QFTParameters& params,
    int steps)
{
    std::vector<double> energy_changes(fields.size(), 0.0);

    // Calculate initial energies
    std::vector<double> initial_energies;
    for (const auto& field_ref : fields) {
        QuantumField<3>& field = field_ref.get();
        initial_energies.push_back(field.calculateTotalEnergy(params));
    }

    // Evolve each field separately
    for (size_t i = 0; i < fields.size(); ++i) {
        QuantumField<3>& field = fields[i].get();
        ParticleType type = field.getParticleType();

        // Create appropriate equation object based on particle type
        if (type == ParticleType::Photon) {
            // For photons, we need both electric and magnetic fields
            // This is a simplification - in reality we'd need to couple them
            MaxwellEquations maxwell(params);

            // Check if next field exists and is also a photon field (for electric/magnetic pairing)
            if (i + 1 < fields.size() &&
                fields[i + 1].get().getParticleType() == ParticleType::Photon) {
                // Evolve both fields together as electromagnetic field
                maxwell.evolveField(field, fields[i + 1].get());
                // Skip the next field since we've already processed it
                i++;
            }
            else {
                // If no paired field available, just evolve this one separately
                // This would require a custom implementation not shown here
                // For now, we'll just log a message
                std::cout << "Warning: Unpaired photon field detected, proper evolution requires "
                             "paired E/B fields"
                          << std::endl;
            }
        }
        else if (type == ParticleType::Electron || type == ParticleType::Proton ||
                 type == ParticleType::Positron || type == ParticleType::Muon) {
            // For fermions use Dirac equation
            DiracEquation dirac(params, type);
            dirac.evolveField(field);
        }
        else {
            // For bosons use Klein-Gordon
            KleinGordonEquation kg(params, type);
            kg.evolveField(field);
        }
    }

    // Calculate final energies and compute changes
    for (size_t i = 0; i < fields.size(); ++i) {
        QuantumField<3>& field = fields[i].get();
        double final_energy = field.calculateTotalEnergy(params);
        energy_changes[i] = final_energy - initial_energies[i];
    }

    return energy_changes;
}

}  // namespace physics
}  // namespace rad_ml
