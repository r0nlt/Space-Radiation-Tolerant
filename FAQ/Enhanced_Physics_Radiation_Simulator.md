## Enhanced Physics Radiation Simulator — FAQ and Guide

**Files covered**:
- `include/rad_ml/physics/enhanced_physics_radiation_simulator.hpp`
- `src/rad_ml/physics/enhanced_physics_radiation_simulator.cpp`

### Overview
The Enhanced Physics Radiation Simulator integrates advanced quantum models into the existing radiation framework to provide physics-informed recommendations and corrections for ML systems operating in radiation environments. It combines:

- Dirac equation (relativistic electron effects)
- Bethe–Salpeter equation (defect cluster formation)
- Green's functions (propagation/phonon interactions)

It also includes holistic integration with neural protection, memory resilience, and redundancy strategies.

**Impact on the framework**
- **Physics fidelity**: Introduces Dirac, Bethe–Salpeter, and Green’s-function methods to move beyond non-relativistic and heuristic models for cascades, clustering, and propagation.
- **Neural protection**: Provides physics-informed recommendations, quantum-enhanced displacement energy scaling, and propagation-aware error corrections for `ProtectedNeuralNetwork`.
- **Memory and redundancy**: Adds phonon/propagation-aware insights for ECC and health-weighted TMR, improving resilience under correlated radiation events.
- **Training operations**: Generates mission/environment-aware training guidance that matches expected radiation profiles and material properties.
- **System adaptivity**: Supplies radiation-intensity driven protection levels and holistic recommendations spanning compute, memory, and storage layers.
- **Extensibility**: Template-based APIs with explicit instantiations allow controlled type expansion while keeping link-time safe.
- **Performance considerations**: Encourages gating diagnostics and precomputations for scalable Green’s/phonon workloads.

### Key Classes
- `rad_ml::sim::EnhancedPhysicsRadiationSimulator`
  - Wraps advanced physics models and exposes higher-level APIs for radiation effects, quantum-enhanced energy calculations, phonon interactions, training guidance, and holistic integration.

- `rad_ml::sim::QuantumEnhancedNeuralProtection`
  - Applies quantum-informed corrections/protection to network weights and predicts degradation over mission timelines using the enhanced simulator.

### Important Public APIs (high level)
- `calculateRelativisticElectronCascade(incident_particle, material, lattice)`
- `calculateDefectClusterFormation(initial_defects, lattice, temperature)`
- `propagateRadiationEffects(initial_distribution, lattice, time_steps)`
- `calculateEnhancedRadiationEffects(particle, material, lattice, environment)`
- `enhanceNeuralNetworkProtection(network, radiation_level, material, lattice)` [template]
- `applyEnhancedPhysicsCorrections(weights, radiation_level, material)` [template]
- `performHolisticFrameworkIntegration(network, environment, material, lattice)` [template]
- `calculateQuantumEnhancedDisplacementEnergy(non_rel_e, particle_e, type)`
- `calculatePhononMediatedInteractions(defect_positions, lattice, temperature)`
- `generateRadiationAwareTrainingRecommendations(mission_duration, environments)`
- `calculateRadiationIntensity(environment)`
- `createMaterialProperties(material_name)`

### Template and Linkage Notes
Some API functions are templates and defined in the `.cpp` with explicit instantiations near the bottom of `src/.../enhanced_physics_radiation_simulator.cpp`.

If you use a new template type that is not explicitly instantiated, you will get link-time errors. You have two options:

1) Add explicit instantiations in the `.cpp` for the new type(s). Example pattern:
```cpp
// Add for your new type MyType
template std::vector<std::string>
EnhancedPhysicsRadiationSimulator::enhanceNeuralNetworkProtection<MyType>(
    rad_ml::neural::ProtectedNeuralNetwork<MyType>&, double,
    const rad_ml::physics::MaterialProperties&,
    const rad_ml::physics::CrystalLattice&);
```

2) Move the full template definitions into the header to avoid explicit instantiation. This increases compile times but removes linkage risks. Keep style consistent with the rest of the project.

Currently instantiated types include `float` and `double` for the most common template methods.

### Usage Examples

Basic physics usage:
```cpp
#include <rad_ml/physics/enhanced_physics_radiation_simulator.hpp>
#include <rad_ml/physics/crystal_lattice_properties.hpp>

using namespace rad_ml;

sim::EnhancedPhysicsRadiationSimulator sim;
auto material = sim.createMaterialProperties("silicon");
auto lattice = physics::CrystalLatticeFactory::Diamond(0.543, 1.0);

auto cascade = sim.calculateRelativisticElectronCascade(
    physics::Particle::createProton(), material, lattice);

auto interactions = sim.calculatePhononMediatedInteractions(
    /* defect_positions */ {Eigen::Vector3d(0,0,0)}, lattice, /*T*/ 300.0);
```

Holistic integration with a protected network (template):
```cpp
#include <rad_ml/neural/protected_neural_network.hpp>

rad_ml::neural::ProtectedNeuralNetwork<float> net;
rad_ml::sim::RadiationEnvironment env; // Populate with mission-specific values

auto recs = sim.performHolisticFrameworkIntegration<float>(net, env, material, lattice);
```

Applying physics-based corrections to weights (template):
```cpp
std::vector<std::vector<float>> weights = /* ... */;
double radiation_level = 0.6;
auto corrected = sim.applyEnhancedPhysicsCorrections(weights, radiation_level, material);
```

Generating radiation-aware training guidance:
```cpp
std::vector<sim::RadiationEnvironment> envs = {env};
auto tips = sim.generateRadiationAwareTrainingRecommendations(120.0, envs);
```

### Behavior/Implementation Notes
- Some calculations are intentionally simplified placeholders for integration readiness (e.g., cascade and radiation intensity aggregation). Replace or extend with domain-accurate models as needed.
- `convertToAdvancedParticle(...)` currently normalizes to a proton for a robust default.
- `calculatePhononMediatedInteractions(...)` uses Green's functions and returns an `Eigen::MatrixXd` built from the real part of the internal complex matrix.

### Performance Considerations
- Several methods log to `std::cout` for traceability. Consider gating behind a compile-time or runtime flag for high-throughput scenarios.
- Phonon/Green's-function steps scale with the number of defects; prefer batching and precomputation when possible.

### Common Questions
- Why do I see explicit template instantiations at the bottom of the `.cpp`?
  - To avoid link errors and control compile times. Add new instantiations if you introduce new types.

- The relativistic cascade seems minimal—where do I plug in detailed physics?
  - Extend usage of `dirac_solver_` and `advanced_model_` in `calculateRelativisticElectronCascade(...)` with your detailed transport/scattering chain.

- How is total radiation intensity computed?
  - `calculateRadiationIntensity(...)` currently uses a simplified aggregation. Wire it to real `RadiationEnvironment` flux fields for mission-accurate values.

- How do I add more materials?
  - Update `createMaterialProperties(...)` with new branches or replace with a material database/factory.

### Related Files/Tests
- Look for verification tests under `test/verification/` and example runs in `simple_enhanced_test` folders to understand typical integrations.

### Extending Safely
- Preserve explicit instantiation coverage when adding new template usages.
- Avoid accessing private members of framework types—prefer factory/creation methods already in use here.
- Keep API changes synchronized between header and implementation.

### Mathematical formulation

The following equations clarify the theory referenced by the simulator. They serve as guidance for extending the placeholder implementations toward higher-fidelity physics.

- Dirac equation (with minimal coupling):
\[\bigl(i\,\gamma^\mu(\partial_\mu + i q A_\mu) - m\bigr)\,\psi(x) = 0.\]

  Conventions and numerical notes:
  - Units: unless otherwise noted, we assume natural units (\(\hbar = c = 1\)). In SI, the operator is \(i\hbar\,\gamma^\mu(\partial_\mu + i q A_\mu)\).
  - Metric signature: adopt \((+,-,-,-)\) so \(\gamma^0\) is Hermitian and \(\gamma^i\) anti-Hermitian; adjust signs if using \((- ,+,+,+)\).
  - Discretization: numerical implementations should specify spatial/temporal grids, boundary conditions (e.g., absorbing/Dirichlet), and spinor component handling to avoid unphysical reflections.

- Bethe–Salpeter equation (two-body bound state):
\[\bigl(G_0^{-1} - K\bigr)\,\Phi = 0,\qquad G = G_0 + G_0 K G,\]
where \(G_0\) is the non-interacting two-particle Green's function and \(K\) the interaction kernel.

- Dyson equation (single-particle Green’s function):
\[G(\omega,\mathbf{k}) = G_0(\omega,\mathbf{k}) + G_0(\omega,\mathbf{k})\,\Sigma(\omega,\mathbf{k})\,G(\omega,\mathbf{k}).\]

- Phonon dispersion (acoustic branch, long-wavelength limit):
\[\omega(\mathbf{k}) \approx v_s\,\lVert\mathbf{k}\rVert,\]
where \(v_s\) is the sound velocity determined by lattice and material parameters.

- Relativistic displacement energy scaling (conceptual):
\[E_{d}^{(\mathrm{rel})}(T) = \gamma(T)\,E_{d}^{(\mathrm{NR})},\qquad \gamma(T) = 1 + \frac{T}{m c^2},\]
with kinetic energy \(T\) and particle rest mass \(m\).

- Phonon-mediated interaction matrix via propagator:
\[\mathcal{M}_{ij}(T) = \Re\!\left\{G_{ij}(\omega;T)\right\},\]
matching the implementation that takes the real part of the Green's function-based interaction matrix.

- Radiation intensity aggregation (mission-level):
\[I_{\mathrm{tot}} = \sum_{s \in \text{species}} \int_{0}^{\infty} \Phi_s(E)\,w_s(E)\,\mathrm{d}E,\]
where \(\Phi_s\) is species flux and \(w_s\) a weighting (e.g., cross section or dose conversion). The current code uses a simplified surrogate; wire to environment flux fields for fidelity.

### Equation-to-API mapping

- Dirac equation \( (i\,\gamma^\mu D_\mu - m)\,\psi = 0 \)
  - `calculateRelativisticElectronCascade(...)` (header `.../enhanced_physics_radiation_simulator.hpp`, impl in `.../enhanced_physics_radiation_simulator.cpp`): governs relativistic electron behavior in cascades.
  - `calculateQuantumEnhancedDisplacementEnergy(...)`: uses a relativistic scaling factor consistent with \(\gamma(T)\).

- Bethe–Salpeter \( (G_0^{-1} - K)\,\Phi = 0 \)
  - `calculateDefectClusterFormation(...)`: models bound-state formation of defect clusters via the advanced model’s BSE solver.

- Dyson/Green’s function \( G = G_0 + G_0\,\Sigma\,G \)
  - `propagateRadiationEffects(...)`: time-evolves defect distributions using Green’s function methods.
  - `calculatePhononMediatedInteractions(...)`: builds an interaction matrix from a propagator and dispersion, returning the real part as `Eigen::MatrixXd`.
  - `QuantumEnhancedNeuralProtection::calculateQuantumErrorCorrection(...)`: estimates error propagation with Green’s function evolution to guide corrections.

- Phonon dispersion \( \omega(\mathbf{k}) \approx v_s\lVert\mathbf{k}\rVert \)
  - `calculatePhononMediatedInteractions(...)`: supplies a dispersion lambda consistent with the linear long-wavelength approximation used by the propagator.

- Relativistic displacement scaling \( E_d^{(\mathrm{rel})}(T) = \gamma(T) E_d^{(\mathrm{NR})} \)
  - `calculateQuantumEnhancedDisplacementEnergy(...)`: returns \(E_d^{(\mathrm{rel})}\) for incident particle energy.
  - `applyEnhancedPhysicsCorrections<T>(...)`: incorporates the enhanced displacement energy into total correction factors applied to weights.

- Radiation intensity aggregation \( I_{\mathrm{tot}} = \sum_s \int \Phi_s(E) w_s(E)\,dE \)
  - `calculateRadiationIntensity(...)`: currently a surrogate; replace with an implementation that integrates mission/environment flux fields and physics-based weightings.

### Updated relativistic cascade model

- Implementation highlights:
  - 24-angle deterministic sampling from ~0.05 to ~1.5 rad
  - Cross-section evaluation via Dirac solver; proportional allocation across angles
  - Energy/material-dependent expected yield using:
    - Pair-creation scale ~ max(3×band_gap, 3.6 eV)
    - Screening factor from dielectric constant
    - Deformation-potential scaling and mild phonon attenuation
  - Minimum and cap guards (3 to 24 secondaries) to ensure stability

- New overload with explicit energy:
```92:95:include/rad_ml/physics/enhanced_physics_radiation_simulator.hpp
std::vector<rad_ml::physics::Particle> calculateRelativisticElectronCascade(
    const rad_ml::physics::Particle& incident_particle,
    const rad_ml::physics::MaterialProperties& target_material,
    const rad_ml::physics::CrystalLattice& crystal_lattice, double incident_energy_eV);
```

- Selected implementation references:
```84:101:src/rad_ml/physics/enhanced_physics_radiation_simulator.cpp
// Angle sampling (24 bins) and cross-section evaluation
std::vector<double> scattering_angles;
// ... fill 24 bins ...
double sigma = dirac_solver_.calculateRelativisticCrossSection(incident_energy_eV, theta, target_material);
```

```114:166:src/rad_ml/physics/enhanced_physics_radiation_simulator.cpp
// Proportional allocation with minimum and cap; logs E in eV
const size_t max_secondaries = 24;
const size_t min_secondaries = 3;
// ... proportional distribution and emission ...
std::cout << "Calculated relativistic electron cascade with " << cascade.size()
          << " secondary electrons (E=" << incident_energy_eV << " eV)\n";
```

### Advanced model updates (Dirac/BSE/Green’s integration)

- Advanced cascade now mirrors simulator behavior with energy/material sensitivity and proportional angle weighting.
- Combined effects pipeline composes Dirac cascade, BSE cluster summary, and Green’s propagation into a concrete `DefectDistribution`.

- Code references:
```416:434:src/rad_ml/physics/advanced_quantum_models.cpp
DefectDistribution AdvancedQuantumRadiationModel::calculateAdvancedRadiationEffects(
    const Particle& incident_particle, const MaterialProperties& target_material,
    const CrystalLattice& crystal_lattice, const RadiationEnvironment& radiation_environment)
{
    // Compose effects from Dirac cascade, BSE clusters, and Green's propagation
    DefectDistribution defects;
    // ...
}
```

```442:497:src/rad_ml/physics/advanced_quantum_models.cpp
std::vector<Particle> AdvancedQuantumRadiationModel::simulateRelativisticElectronCascade(
    const Particle& initial_electron, const CrystalLattice& lattice, const QFTParameters& params)
{
    // 16-angle sampling, Dirac cross-sections, expected yield from energy/material
    // and proportional allocation to produce electron secondaries deterministically.
}
```

### Verification test: Relativistic cascade

- Location: `test/verification/relativistic_cascade_test.cpp`
- Validates:
  - Only electron secondaries
  - Deterministic count within [3, 24]
  - Material sensitivity (higher dielectric constant does not exceed cap)
  - Energy overload calls accept different energies and remain within bounds

- Build and run:
```bash
cmake --build /Users/rishabnuguru/space/build-radiation --target relativistic_cascade_test -j 8
ctest --test-dir /Users/rishabnuguru/space/build-radiation -R relativistic_cascade_test --output-on-failure
```

### Source line references

- Header declarations (anchors)
```56:82:include/rad_ml/physics/enhanced_physics_radiation_simulator.hpp
class EnhancedPhysicsRadiationSimulator : public PhysicsRadiationSimulator {
   public:
    // ...
    std::vector<rad_ml::physics::Particle> calculateRelativisticElectronCascade(
        const rad_ml::physics::Particle& incident_particle,
        const rad_ml::physics::MaterialProperties& target_material,
        const rad_ml::physics::CrystalLattice& crystal_lattice);
```

```91:107:include/rad_ml/physics/enhanced_physics_radiation_simulator.hpp
std::vector<rad_ml::physics::DefectCluster> calculateDefectClusterFormation(
    const rad_ml::physics::DefectDistribution& initial_defects,
    const rad_ml::physics::CrystalLattice& crystal_lattice, double temperature);

std::vector<rad_ml::physics::DefectDistribution> propagateRadiationEffects(
    const rad_ml::physics::DefectDistribution& initial_distribution,
    const rad_ml::physics::CrystalLattice& crystal_lattice,
    const std::vector<double>& time_steps);
```

```117:121:include/rad_ml/physics/enhanced_physics_radiation_simulator.hpp
rad_ml::physics::DefectDistribution calculateEnhancedRadiationEffects(
    const rad_ml::physics::Particle& incident_particle,
    const rad_ml::physics::MaterialProperties& target_material,
    const rad_ml::physics::CrystalLattice& crystal_lattice,
    const rad_ml::sim::RadiationEnvironment& radiation_environment);
```

```146:149:include/rad_ml/physics/enhanced_physics_radiation_simulator.hpp
double calculateQuantumEnhancedDisplacementEnergy(double non_relativistic_energy,
                                                  double particle_energy,
                                                  rad_ml::physics::ParticleType particle_type);
```

```158:160:include/rad_ml/physics/enhanced_physics_radiation_simulator.hpp
Eigen::MatrixXd calculatePhononMediatedInteractions(
    const std::vector<Eigen::Vector3d>& defect_positions,
    const rad_ml::physics::CrystalLattice& crystal_lattice, double temperature);
```

```219:219:include/rad_ml/physics/enhanced_physics_radiation_simulator.hpp
double calculateRadiationIntensity(const rad_ml::sim::RadiationEnvironment& env) const;
```

### Verification test: Radiation intensity aggregation

- Location: `test/verification/radiation_intensity_aggregation_test.cpp`
- Validates:
  - Monotonic increases with trapped flux and SPE scaling (solar activity, 1/r²)
  - SAA multiplier (~2.5×)
  - Atmosphere attenuation ratio 1/(1 + depth/50)
  - Magnetic attenuation ratio 1/(1 + 0.3·(B−1))
  - Negative flux clamping to 0
  - Combined-factor exact composition match
  - Zero-distance guard (fallback to 1.0)
  - Noise robustness: Gaussian perturbations; multi-seed mean stability (<5% deviation)

- Code references (selected snippets):
```44:56:test/verification/radiation_intensity_aggregation_test.cpp
// Baseline and trapped flux monotonicity
auto env = makeEnv();
double I0 = sim.calculateRadiationIntensity(env);
env.trapped_proton_flux *= 2.0;
env.trapped_electron_flux *= 2.0;
double I_flux = sim.calculateRadiationIntensity(env);
assert(I_flux > I0);
```

```99:106:test/verification/radiation_intensity_aggregation_test.cpp
// Isolated SPE scaling: solar activity and 1/r^2
env.solar_activity = 1.0;
env.distance_from_sun = 2.0;  // 1/4 factor
double I_spe_2 = sim.calculateRadiationIntensity(env);
double expected_spe_2 = 1.0 * 5.0e4 * (1.0 / 4.0);
assert(approxEqual(I_spe_2, expected_spe_2, 1e-3));
```

```123:133:test/verification/radiation_intensity_aggregation_test.cpp
// Atmosphere attenuation ratio
double I_atm0 = sim.calculateRadiationIntensity(env);
env.atmosphere_depth = 100.0;
double I_atm100 = sim.calculateRadiationIntensity(env);
double expected_atm_ratio = 1.0 / (1.0 + 100.0 / 50.0);  // 1/3
assert(approxEqual(I_atm100 / I_atm0, expected_atm_ratio, 1e-2));
```

```185:206:test/verification/radiation_intensity_aggregation_test.cpp
// Combined factors exact composition check
double I_combined = sim.calculateRadiationIntensity(env);
// ... compute expected_total from components, SAA, atmosphere, magnetic factors ...
assert(approxEqual(I_combined, expected_total, 1e-6));
```

- Build and run:
```bash
cmake --build /Users/rishabnuguru/space/build-radiation --target radiation_intensity_aggregation_test -j 8
ctest --test-dir /Users/rishabnuguru/space/build-radiation -R radiation_intensity_aggregation_test --output-on-failure
```

- `calculateRelativisticElectronCascade` (implementation)
```60:79:src/rad_ml/physics/enhanced_physics_radiation_simulator.cpp
std::vector<rad_ml::physics::Particle>
EnhancedPhysicsRadiationSimulator::calculateRelativisticElectronCascade(
    const rad_ml::physics::Particle& incident_particle,
    const rad_ml::physics::MaterialProperties& target_material,
    const rad_ml::physics::CrystalLattice& crystal_lattice)
{
    // Use Dirac equation for relativistic electron behavior
    std::vector<rad_ml::physics::Particle> cascade;
    // ... more code ...
    return cascade;
}
```

- `calculateQuantumEnhancedDisplacementEnergy` (implementation)
```170:179:src/rad_ml/physics/enhanced_physics_radiation_simulator.cpp
double EnhancedPhysicsRadiationSimulator::calculateQuantumEnhancedDisplacementEnergy(
    double non_relativistic_energy, double particle_energy,
    rad_ml::physics::ParticleType particle_type)
{
    double relativistic_factor = dirac_solver_.calculateRelativisticDisplacementEnergy(
        non_relativistic_energy, particle_energy);
    return non_relativistic_energy * relativistic_factor;
}
```

- `calculateDefectClusterFormation` (implementation)
```81:94:src/rad_ml/physics/enhanced_physics_radiation_simulator.cpp
std::vector<rad_ml::physics::DefectCluster>
EnhancedPhysicsRadiationSimulator::calculateDefectClusterFormation(
    const rad_ml::physics::DefectDistribution& initial_defects,
    const rad_ml::physics::CrystalLattice& crystal_lattice, double temperature)
{
    std::vector<rad_ml::physics::DefectCluster> clusters =
        advanced_model_.calculateDefectClusterFormation(initial_defects, crystal_lattice,
                                                        temperature);
    return clusters;
}
```

- `propagateRadiationEffects` (implementation)
```96:109:src/rad_ml/physics/enhanced_physics_radiation_simulator.cpp
std::vector<rad_ml::physics::DefectDistribution>
EnhancedPhysicsRadiationSimulator::propagateRadiationEffects(
    const rad_ml::physics::DefectDistribution& initial_distribution,
    const rad_ml::physics::CrystalLattice& crystal_lattice, const std::vector<double>& time_steps)
{
    std::vector<rad_ml::physics::DefectDistribution> evolution =
        advanced_model_.propagateRadiationEffects(initial_distribution, crystal_lattice,
                                                  time_steps);
    return evolution;
}
```

- `calculatePhononMediatedInteractions` (implementation)
```181:199:src/rad_ml/physics/enhanced_physics_radiation_simulator.cpp
Eigen::MatrixXd EnhancedPhysicsRadiationSimulator::calculatePhononMediatedInteractions(
    const std::vector<Eigen::Vector3d>& defect_positions,
    const rad_ml::physics::CrystalLattice& crystal_lattice, double temperature)
{
    auto phonon_dispersion = [&](const Eigen::Vector3d& k_vector) -> double {
        double k_magnitude = k_vector.norm();
        double sound_velocity = 8000.0;       // m/s for silicon
        return sound_velocity * k_magnitude;  // Linear dispersion approximation
    };
    // ... more code ...
}
```

- `QuantumEnhancedNeuralProtection::calculateQuantumErrorCorrection` (implementation)
```526:563:src/rad_ml/physics/enhanced_physics_radiation_simulator.cpp
template <typename T>
std::vector<std::vector<T>> QuantumEnhancedNeuralProtection::calculateQuantumErrorCorrection(
    const std::vector<std::vector<T>>& corrupted_weights,
    const std::vector<std::vector<T>>& original_weights,
    const rad_ml::sim::RadiationEnvironment& radiation_environment)
{
    std::vector<std::vector<T>> corrected_weights = corrupted_weights;
    // ... more code ...
    return corrected_weights;
}
```

- `calculateRadiationIntensity` (implementation)
```404:412:src/rad_ml/physics/enhanced_physics_radiation_simulator.cpp
double EnhancedPhysicsRadiationSimulator::calculateRadiationIntensity(
    const rad_ml::sim::RadiationEnvironment& env) const
{
    double total_flux = 1.0e7;               // Default baseline flux
    return total_flux * (1.0 + 0.1 * 10.0);  // Scale factor approximation
}
```
