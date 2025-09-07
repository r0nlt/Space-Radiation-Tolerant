# GEO Mission Validation: Deep Dive

## Overview

In short, this test is a digital stress test for a virtual satellite computer, designed to ensure it can survive the harsh radiation of a 15-year mission in Geostationary Earth Orbit (GEO).

Think of it like a "crash test" for software and memory, but instead of physical impacts, you're simulating destructive particles from the sun and deep space.

The test simulates the entire process of a radiation strike and the system's attempt to recover, millions of times over, to build statistical confidence.

### What the Test Does

1. **Simulate the GEO Environment**: Creates a virtual space environment, modeling different conditions a satellite will face - from a calm "normal" day, flying through the intense Van Allen radiation belts, to a full-blown solar storm.

2. **Inject Realistic Errors**: Based on the environment, deliberately corrupts data in memory. It doesn't just flip random bits; it mimics the specific damage patterns caused by different types of radiation—from a single bit flip to a catastrophic "word error" that scrambles an entire piece of data.

3. **Test the Defenses**: Runs a suite of self-healing software (advanced TMR and voting algorithms) on the corrupted data. It checks if these defenses can correctly identify the error and perfectly reconstruct the original, uncorrupted information.

4. **Measure and Report**: Acts as a meticulous bookkeeper, recording which defenses succeeded under which conditions and generates a detailed "report card" (geo_mission_verification_report.txt). This report shows the exact success rate of each protection method, effectively proving how robust the system is.

By simulating everything from minor glitches to worst-case scenarios, this test provides the critical proof needed to trust that a multi-million dollar satellite will operate reliably for its entire 15-year lifespan.

---

## Imports and Libraries

### C++ Standard Library
- `<algorithm>`, `<bitset>`, `<cassert>`, `<chrono>`, `<cmath>`, `<cstring>`, `<fstream>`, `<functional>`, `<iomanip>`, `<iostream>`, `<map>`, `<numeric>`, `<random>`, `<string>`, `<vector>`
- **Purpose**: algorithms/utilities, time measurement, math, I/O, containers, RNG for trials

### Eigen (header-only)
- `#include <Eigen/Core>`
- **Purpose**: foundational linear algebra types/ops leveraged by physics/neural modules
- **External dependency**: Eigen 3.x (found via CMake, e.g., `/usr/local/include/eigen3`)

### Rad-ML Framework Headers (project modules)

#### Core Memory
- `include/rad_ml/core/memory/aligned_memory.hpp`: aligned, redundant value storage
- `include/rad_ml/core/memory/memory_scrubber.hpp`: registration- and callback-based scrubbing
- `include/rad_ml/core/memory/protected_value.hpp`: TMR-like protected value container

#### Core Redundancy
- `include/rad_ml/core/redundancy/enhanced_voting.hpp`: standard/bit/burst/word/weighted/adaptive voting, fast-bit correction, pattern detection
- `include/rad_ml/core/redundancy/tmr.hpp`: Triple Modular Redundancy primitives
- `include/rad_ml/tmr/enhanced_tmr.hpp`: enhanced TMR with repair/regeneration

#### Mission/Scenario
- `include/rad_ml/mission/mission_profile.hpp`: mission type (e.g., GEOSTATIONARY) config
- `include/rad_ml/radiation/space_mission.hpp`: radiation-related mission interfaces

#### Neural
- `include/rad_ml/neural/protected_neural_network.hpp`: protected NN for robustness tests

#### Physics
- `include/rad_ml/physics/quantum_enhanced_radiation.hpp`: physics-based radiation effects (charge deposition, MBU size, temperature-critical charge, enhanced bit flip probability)

### Optional Advanced Features
Guarded by `RAD_ML_ADVANCED_FEATURES`:
- `core/adaptive/adaptive_framework.hpp`
- `testing/benchmark_framework.hpp`
- `testing/fault_injector.hpp`
- `tmr/health_weighted_tmr.hpp`
- `tmr/physics_driven_protection.hpp`
- `tmr/temporal_redundancy.hpp`

**Purpose**: extended benchmarking, fault injection, and advanced redundancy policies (not required for the base GEO validation)

---

## Namespaces

The test uses shorthand namespace imports for readability:

- `using namespace rad_ml::core::redundancy;` — voting and redundancy primitives (standard/bit/burst/word/weighted/adaptive, TMR/enhanced TMR, pattern detection, fast corrections)
- `using namespace rad_ml::physics;` — physics models (e.g., `QuantumEnhancedRadiation`)
- `using namespace rad_ml::neural;` — protected neural network utilities
- `using namespace rad_ml::mission;` — mission profiling (e.g., GEOSTATIONARY)
- `using namespace rad_ml::radiation;` — radiation-domain types and helpers

---

## Environment Setup

The test initializes core environment elements before trials:

### Mission Profile
```cpp
MissionProfile geo_profile(MissionProfile::MissionType::GEOSTATIONARY);
```
Establishes GEO mission context for downstream logic/reporting.

### Physics Simulator
```cpp
SemiconductorProperties silicon_props;
QuantumEnhancedRadiation quantum_sim(silicon_props);
```
Provides charge deposition, MBU size, temperature‑critical charge, and bit‑flip probability APIs.

### Neural Protection Component
```cpp
std::vector<size_t> nn_architecture = {8, 64, 32, 8};
ProtectedNeuralNetwork<float> protected_nn(nn_architecture, ProtectionLevel::ADAPTIVE_TMR);
```
Optional robustness checks alongside voting.

### Random Number Generation
```cpp
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> val_dist(-1000.0, 1000.0);
```

### Scenario Parameters
`GEO_SCENARIOS[6]` table defines: particle flux, upset probabilities (single/multi/burst/word), temperature, shielding thickness (unused here), dominant particle, average energy, LET, van Allen factor, eclipse flag, solar storm probability.

Each scenario drives error injection choices and physics accumulation.

---

## Parameter Setup

### Trials and Seeding
- **Trials per test**: `NUM_TRIALS_PER_TEST = 50000` (configurable constant)
- **RNG**: `std::mt19937` seeded from `std::random_device` for non-deterministic runs
  - For reproducibility, you can fix the seed (e.g., via a compile-time define or CLI/env in your harness) and log it
- **Value distributions**: uniform over representative ranges (e.g., `[-1000, 1000]`) per data type

### Scenario Table (GEO_SCENARIOS)
Typical fields used in selection and physics:
- `flux_scale` (relative particle flux), `van_allen_factor`, `storm_probability`, `eclipse` (bool)
- Upset probabilities: `p_single`, `p_multi`, `p_burst`, `p_word`
- Environment: `temperature_C`, `dominant_particle`, `avg_energy_MeV`, `avg_let_MeV_cm2_mg`, `shielding_mm`

The active scenario is iterated in the outer loop; its fields parameterize both error selection and physics.

### Error Model Selection
For each trial, an error model is chosen using scenario-weighted probabilities (single, multi, burst, word).
GEO-specific compound tests (e.g., solar storm, van Allen) may inject multiple/clustered upsets or sustained disturbances.

### Protection Methods Under Test
Evaluated per trial on three logical copies (TMR-like):
- Standard vote, Bit-level vote, Burst vote, Word vote, Adaptive vote, Weighted vote
- Fast-bit correction; Pattern detection success path
- **Advanced (optional)**: Enhanced TMR with repair, health-weighted TMR, temporal redundancy

### Physics Parameters
`SemiconductorProperties` + `QuantumEnhancedRadiation` provide:
- Charge deposition vs energy/LET/particle, temperature-dependent critical charge `Qcrit(T)`
- Quantum MBU size distribution, enhancement to flip probability
- Physics metrics are accumulated selectively (e.g., Van Allen, Solar storm scenarios)

### Progress and Diagnostics Cadence
- **Progress prints**: roughly every 1%: `progress_step = max(1, NUM_TRIALS_PER_TEST / 100)` with rate and ETA
- **Silent data corruption (SDC) probes**: at a fixed interval (e.g., every 1000 trials) to estimate SDC rate
- **Per-test timing**: uses high-resolution clocks; durations are logged alongside throughput

### Build/Runtime Toggles
For reproducible performance:
- **Release flags**: `-O3 -march=native -DNDEBUG -flto`; disable OpenMP in this configuration
- **Eigen**: header-only; ensure include path set; optionally disable Eigen debug checks in Release
- **Optional compile-time feature gates**: `RAD_ML_ADVANCED_FEATURES` to include extended policies/benchmarks

---

## Scenarios Tested

### Nominal GEO
- **Purpose**: Baseline operating conditions in GEO
- **Effects**: Moderate `flux_scale`, baseline `p_single`; very low `p_multi`/`p_burst`/`p_word`; typical `temperature_C`, nominal `avg_energy_MeV` and `avg_let_MeV_cm2_mg`

### Van Allen Peak
- **Purpose**: Increased charged particle prevalence near the belts
- **Effects**: Elevated `van_allen_factor` and effective flux; higher clustering likelihood (raised `p_multi`/`p_burst`); possibly higher `avg_let_MeV_cm2_mg`; physics exposure metrics accumulated here

### Solar Storm
- **Purpose**: Transient extreme environment during solar events
- **Effects**: Non‑zero `storm_probability` driving compound upsets; increased `p_burst`/`p_word` and multi‑error injections within a trial; charge deposition spikes in physics model

### Eclipse
- **Purpose**: Thermal changes during Earth shadowing
- **Effects**: `eclipse = true` with lower `temperature_C`; temperature shift affects critical charge `Qcrit(T)`, typically reducing flip probability compared to hot conditions

### End‑of‑Life (EOL)
- **Purpose**: Cumulative dose and aging toward mission completion
- **Effects**: Long‑term exposure considered in reliability projection; parameters may reflect slightly higher susceptibility versus nominal in projections

### Solar Maximum
- **Purpose**: Elevated activity phase in the solar cycle
- **Effects**: Broad increase in ambient flux and/or LET over extended periods; higher baseline upset probabilities than nominal

**Note**: The shielding variant of the test applies scenario‑specific attenuation via `shielding_mm` (not used in this baseline), reducing effective flux/LET before probability mapping.

---

## How We Test Each Scenario

### Loop Structure
For each data type → for each GEO scenario → for each error/test type, run `NUM_TRIALS_PER_TEST` trials and time the section.

### Per‑Trial Flow (within a scenario)
1. Draw an input value from the configured distribution for the active data type
2. Create three logical copies (TMR‑like) of the value
3. Select an error model using the scenario's probabilities (`p_single`, `p_multi`, `p_burst`, `p_word`) plus GEO‑specific tests (Van Allen, Solar storm, etc.)
4. Apply the chosen corruption to the copies (single/multi/adjacent burst/word or compound injections for GEO tests)
5. Evaluate protection methods: standard/bit/burst/word/adaptive/weighted voting, fast‑bit correction, pattern detection success; advanced TMR variants where enabled
6. If the scenario is physics‑relevant (e.g., Van Allen, Solar storm), query `QuantumEnhancedRadiation` to accumulate charge, MBU size, enhancement, and exposure
7. Update counters: method successes, fault pattern histogram, SDC probe outcome (at fixed cadence), and progress output

### Scenario‑Level Aggregation
After all trials: compute per‑method success rates, mean Hamming distances when votes differ, SDC rate, physics aggregates, and average time per trial.

Emit a concise per‑scenario summary and append results to the verification report.

### Reproducibility and Controls
- Trials count is fixed (`NUM_TRIALS_PER_TEST`); progress cadence ~1% of trials
- RNG defaults to nondeterministic seed; fix seed to reproduce identical runs
- Build in Release with `-O3 -march=native -DNDEBUG -flto`; OpenMP disabled here to avoid confounding

---

## Method‑by‑Method Details

### Standard Voting (baseline TMR)
- **Purpose**: Simple majority across three replicas on whole‑value equality (or numeric median where applicable)
- **Strengths**: Masks any single‑replica corruption when two replicas remain correct
- **Failure modes**: Two replicas share same wrong value (e.g., word‑wide failure); three distinct values with no majority; sub‑word/burst faults that create consistent wrong majorities
- **Tunables**: None in core rule; optionally enable median for numeric types
- **Performance**: O(1), branch‑light, minimal overhead

### Bit‑Level Voting
- **Purpose**: Majority per bit across three replicas to reconstruct the correct value
- **Strengths**: Handles scattered multi‑bit upsets that do not align across replicas
- **Failure modes**: Correlated errors where two replicas share flips at many positions; semantic constraints beyond bit agreement (e.g., parity/CRC) not enforced
- **Tunables**: Word width for vectorized ops; optional mask of immutable bits
- **Performance**: Use word‑wide XOR/AND and popcount; avoid heap and branches

### Burst‑Aware Voting
- **Purpose**: Specialize for contiguous runs of flipped bits (typical of charge sharing/MBUs)
- **Strengths**: Tolerant when a single replica has localized contiguous damage
- **Failure modes**: Aligned bursts across two replicas; repeated bursts producing multi‑region damage
- **Tunables**: Min/max run length; adjacency tolerance to merge near‑adjacent flips
- **Performance**: Detect runs via XOR + run‑length or bit scans; keep constant time

### Word‑Level Voting
- **Purpose**: Vote over word‑sized chunks/lanes to tolerate whole‑word corruptions in one replica
- **Strengths**: Robust against full‑lane/register failures in a single copy
- **Failure modes**: Two replicas lose the same lane(s); sub‑word symmetric patterns
- **Tunables**: Lane size (8/16/32/64‑bit); tie‑break rules per lane
- **Performance**: Chunked comparisons with masks; vectorize where available

### Adaptive Voting (pattern‑guided)
- **Purpose**: Classify fault pattern (single/multi/burst/word) and route to the best voter dynamically
- **Strengths**: Near‑best‑of‑breed behavior across diverse patterns when classification is accurate
- **Failure modes**: Misclassification (e.g., multi vs burst) or inconclusive patterns; sub‑optimal routing
- **Tunables**: Hamming thresholds; burst run‑length bounds; routing table; confidence cutoffs
- **Performance**: Precompute Hamming/run metrics once, reuse across voters

### Weighted Voting (health/confidence)
- **Purpose**: Down‑weight replicas inferred unhealthy by recent inconsistencies
- **Strengths**: Maintains correctness when one replica is repeatedly damaged
- **Failure modes**: Over/under‑reaction of weights; trusting a currently damaged replica
- **Tunables**: Initial weights; decay/recovery rates; hysteresis band; cap/floor; feature fusion (bit disagreement, history, physics cues)
- **Performance**: Keep weight updates branch‑reduced; cache features

### Fast‑Bit Correction
- **Purpose**: Constant‑time repair for obvious single‑bit divergence prior to/alongside voting
- **Strengths**: Perfect for single‑bit upsets; improves all downstream voters
- **Failure modes**: Not applicable to multi‑bit/burst/word faults
- **Tunables**: Gating threshold (strict single‑bit via popcount==1); early‑exit ordering
- **Performance**: XOR + popcount; branchless where possible

### Pattern Detection
- **Purpose**: Label injected fault type; feeds adaptive/weighted strategies
- **Strengths**: Enables specialized voters to be selected automatically
- **Failure modes**: Confusions between multi and burst; ambiguous patterns
- **Tunables**: Hamming/run‑length thresholds; lane‑parity checks; confidence scoring
- **Performance**: Compute once per trial; avoid repeated scans

### Enhanced TMR with Repair (advanced)
- **Purpose**: After a correct vote, repair/regenerate the damaged replica and verify re‑alignment
- **Strengths**: Restores redundancy, not just masks errors
- **Failure modes**: Repair trigger too lax/strict; insufficient validation; repeated immediate failures
- **Tunables**: Trigger conditions; retry count; cool‑down; post‑repair validation rule
- **Performance**: Keep repair paths deterministic; log outcomes rather than verbose traces

### Temporal Redundancy (advanced)
- **Purpose**: Vote across time samples to reject transient faults
- **Strengths**: Removes short‑lived upsets that single‑shot voters can't distinguish
- **Failure modes**: Persistent or long bursts across the sampling window
- **Tunables**: Number of samples; inter‑sample delay; majority vs unanimity; debouncing after repair
- **Performance**: Choose intervals from the transient duration distribution to minimize overhead

---

## Decision Boundaries and Failure Modes

### Standard Voting
- **Masks**: any single‑replica corruption when the other two match the truth
- **Fails**: when two replicas suffer the same error (e.g., identical word corruption) or diverge in different ways such that no majority equals the original

### Bit‑Level Voting
- **Handles**: scattered multi‑bit upsets if bitwise majorities remain truthful
- **Fails**: when two replicas share the same bit flips at many positions (correlated errors) or when semantic correctness depends on more than per‑bit agreement (e.g., parity/CRC not recomputed)

### Burst‑Aware Voting
- **Excels**: when damage is localized and contiguous in one replica
- **Fails**: on multi‑replica bursts aligned to the same region or repeated bursts creating multi‑region damage

### Word‑Level Voting
- **Robust**: against full‑word corruption in a single replica
- **Fails**: when two replicas lose the same word(s) or when corruption is sub‑word but symmetrically misleading

### Adaptive/Weighted
- **Dependent**: on fault pattern classification and health inference; misclassification or delayed weight decay can misroute decisions
- **Improve**: by logging confusion matrices and adding hysteresis to health updates

### Temporal Redundancy
- **Removes**: transients shorter than the sampling window; persistent faults across the window still pass through
- **Tune**: sampling interval/window length to the expected transient duration distribution

---

## Tunables and Configuration Knobs

### Bitwise/Pattern Detection
- Hamming distance thresholds for single vs multi‑bit classification
- Burst minimum/maximum run length; adjacency tolerance
- Word size/chunking strategy (e.g., 8/16/32‑bit lanes)

### Weighted/Health
- Initial weights, decay rate, recovery rate, cap/floor, hysteresis band
- Confidence fusion: combine per‑bit disagreement rates, recent error history, physics signals

### Temporal Redundancy
- Number of samples, inter‑sample delay, vote rule (majority vs unanimity)
- Optional debouncing to require stability after repair

### Enhanced TMR/Repair
- Repair trigger conditions, retry count, cool‑down period, validation rule post‑repair

### Performance and Reproducibility
- Fixed seed for reproducible runs; `NUM_TRIALS_PER_TEST` for tighter confidence
- Ensure Release flags and disable OpenMP to avoid variability here

---

## Performance Considerations

All voters are O(1) per trial with small constant factors; bit‑level voters should be implemented with word‑wide bit operations and no heap allocations.

Prefer branch‑reduced logic and precomputed masks; keep data in registers/L1.

Log only at coarse cadence (~1%) to avoid I/O bottlenecks; aggregate counters in registers then flush.

---

## Worked Example (Why Bit‑Level Can Beat Standard)

- **Original**: `0b0110 1110` (110)
- **Replica A (truth)**: `0b0110 1110`
- **Replica B (two scattered flips)**: `0b0111 1010`
- **Replica C (other scattered flips)**: `0b0010 1111`

**Standard voting** compares whole values → three different values, no majority equals original.

**Bit‑level voting** per position recovers the original at most bit positions because at each bit there is still a 2‑of‑3 majority matching A.

---

## Measurement and Reporting

Per‑method success counters, SDC probes, mean Hamming distance on mismatches, and average time per trial are recorded.

Scenario summaries are emitted to console and appended to `geo_mission_verification_report.txt` for traceability.

### Success Criteria and Metrics
- **Success recorded**: per method when the output equals the original uncorrupted value
- **Additional metrics**: mean Hamming distance on disagreements, SDC rate (periodic known‑truth probes), per‑method success rates, and average time per trial

---

## Physics Model Assumptions and Validation

### Semiconductor Device Model (charge deposition)
- **Sensitive volume**: rectangular parallelepiped (RPP) per cell with effective depth `t_sv` and lateral pitch from `SemiconductorProperties`
- **LET‑to‑charge**: deposited charge computed from LET, silicon density, and path length through the SV; converted using the silicon ionization energy (~3.6 eV per e‑h pair). A simple straight‑track path with tilt is assumed; collection efficiency scales charge to account for transport/geometry
- **Charge collection**: first‑order funnel collection factor broadens the effective sensitive region under high‑LET strikes; parameterized, not fully transport‑simulated

### Temperature‑Dependent Critical Charge Qcrit(T)
- **Model**: `Qcrit(T) = Qcrit_25C * f(T)` where `f(T)` is a monotonic scaling reflecting mobility/lifetime changes with temperature (calibration parameter in `SemiconductorProperties`)
- **Validation**: fit `f(T)` to temperature sweep data (vendor/public SRAM/FF characterization where available). The simulator uses this fitted slope/curve; if no data, a conservative default trend is used and called out in logs

### Particle Interaction Cross‑Sections
- **Heavy‑ion SEU**: Weibull parameterization `σ(LET) = σ_sat * (1 − exp(−((LET − L0)/W)^s))` for `LET ≥ L0`, else 0. Parameters `(σ_sat, L0, W, s)` are part of the physics config
- **Proton indirect ionization**: optional equivalence mapping to an effective LET or folded reaction model; disabled by default unless scenario explicitly enables it
- **Angular dependence**: effective path length scales with incidence angle; simple secant‑based correction within geometric limits

### MBU Size Distribution
- **Distribution**: discrete clustered‑bit model (geometric/negative‑binomial‑like) whose mean/shape grow with LET and shrink with larger cell pitch/spacing; truncated by word/line width
- **Validation**: matched against published MBU histograms for advanced nodes (e.g., 65/45/28 nm) where available; goodness assessed by histogram overlap/KS‑style checks in offline calibration. Defaults chosen to reproduce mid‑LET clustering while not overpredicting at low LET

### Calibration Workflow and Parameters
- **Exposed knobs** (via `SemiconductorProperties` / simulator config): `t_sv`, collection efficiency, temperature scaling `f(T)`, cell pitch/word width, Weibull `(σ_sat, L0, W, s)`, MBU shape/scale
- **Scenario linkage**: scenario provides dominant particle, average energy/LET, temperature, and flux scaling; shielding (in the shielded test) attenuates LET/flux before mapping to upset probabilities

### Limitations (scope of the current model)
- No full GEANT4/CREME transport; straight‑track energy deposition with parameterized collection
- Layout‑specific effects (well proximity, guard rings) are abstracted via pitch/efficiency parameters
- Proton nuclear secondaries are approximated or disabled unless explicitly enabled with calibrated parameters

### Reproducibility and Extension
All physics parameters are centralized; include them in reports for traceability. Replace the LET/σ/MBU parameter blocks with device‑specific fits when characterization data is available. The simulator interfaces allow swapping in a higher‑fidelity transport backend if needed.

---

## Algorithmic Details

### Bit‑Level Majority Reconstruction
**Inputs**: three replicas A, B, C of equal bit width W.

**Process**:
1. Compute disagreement masks: `dAB = A ^ B`, `dAC = A ^ C`, `dBC = B ^ C`
2. Bits where A is majority: `mA = ~dAB & ~dAC` (A equals B and C at that bit)
3. Bits where B is majority: `mB = ~dAB & ~dBC`
4. Bits where C is majority: `mC = ~dAC & ~dBC`
5. Reconstruct: `R = (A & mA) | (B & mB) | (C & mC)`

**Ties** (no clear 2‑of‑3 at a bit) are rare; policy options: prefer A, or fall back to standard vote/median.

**Implementation note**: operate on machine words (e.g., 64‑bit chunks) for speed; no loops per bit.

### Implemented Form in Code
The production voter performs an explicit per‑bit majority using the identity `(ab) | (ac) | (bc)`:
- For each bit i: `majority_i = (bit_a & bit_b) | (bit_a & bit_c) | (bit_b & bit_c)` and set it in the result
- This is equivalent to the mask method above. The path is generic across arithmetic types via `memcpy` to unsigned words
- **Floating‑point note**: this voter is raw bitwise on IEEE‑754 representations (no exponent/mantissa special handling in this path)

### Adaptive Vote Routing (as implemented)
**Input**: `FaultPattern` enum from `detectFaultPattern(a,b,c)`

**Exact mapping**:
- `SINGLE_BIT` → `bitLevelVote`
- `ADJACENT_BITS` → `bitLevelVote`
- `BYTE_ERROR` → `burstErrorVote` (segment‑based)
- `WORD_ERROR` → `wordErrorVote` (closest‑pair + reconstruction)
- `BURST_ERROR` → `burstErrorVote`
- `UNKNOWN` → compute `bitLevelVote`, `wordErrorVote`, `burstErrorVote`; return the result that equals any input if possible, else prefer bit‑level as conservative

**Fast path**: if any two inputs are equal, return that value before pattern mapping.

**Optional API**: `detectFaultPatternWithConfidence` exists, but the adaptive router consumes only the enum in the current tests.

### Fault‑Pattern Classification (as implemented)
**Inputs**: `diff_ab = a^b`, `diff_ac = a^c`, `diff_bc = b^c`; bit counts via `countBits(...)`

**Process**:
1. **Perfect match**: if no differences, return UNKNOWN (nothing to correct)
2. **Choose `diff_pattern`**: if any pair matches exactly, take the XOR to the outlier; else take the pair with the fewest differing bits
3. **Classification tests** on `diff_pattern`:
   - `SINGLE_BIT`: `countBits(diff_pattern) == 1`
   - `ADJACENT_BITS`: contiguous 1‑run heuristic (`areAdjacentBits`) indicating neighboring flips
   - `BYTE_ERROR`: all differing bits lie within one 8‑bit byte (`areByteBoundary`)
   - `WORD_ERROR`: for 32‑bit or smaller, "large" corruption (more than half the bits differ); for 64‑bit, all differing bits confined to either the lower or upper 32‑bit half (`areWordBoundary`)
   - `BURST_ERROR`: longest run of 1s ≥ 3 and at least 60% of all set bits are in that run (`areBurstPattern`)
   - **Otherwise**: `UNKNOWN`

A separate `detectFaultPatternWithConfidence` variant computes a confidence score; the main router does not use it.

---

## Mathematical Formulation

### Notation
Replicas are bit vectors of width W: a,b,c ∈ {0,1}^W. XOR is ⊕, AND is ∧, OR is ∨. Hamming weight is ||x||₁. Index bits as aᵢ.

### Bit‑Level Majority (framework implementation)
```
r = (a ∧ b) ∨ (a ∧ c) ∨ (b ∧ c), i.e., rᵢ = maj(aᵢ,bᵢ,cᵢ)
```

### Weighted Bit‑Level Vote
Let weights wₐ,wb,wc ∈ [0,1]. For each bit i:
```
rᵢ = 1[wₐaᵢ + wbbᵢ + wccᵢ ≥ ½(wₐ+wb+wc)]
```

### Word‑Level Vote (as used by EnhancedVoting)
Define d(x,y) = ||x ⊕ y||₁. Choose closest pair (x,y) ∈ {(a,b),(a,c),(b,c)} minimizing d; let z be the outlier. Then:

```
r = {
  x                           if d(x,y) = 0
  x or y                      if d(x,y) = 1 and chosen value's differing bit equals zᵢ
  bitMaj(x,y,z)              otherwise (fallback to per‑bit majority on x,y,z)
}
```

**Note**: An alternate agree‑mask reconstruction exists in ETMR, but the GEO test harness uses the EnhancedVoting path above.

### Burst‑Aware (segment) Vote (8‑bit segmentation as in code)
Let total width be W = 8M bits and k ∈ {0,...,M-1} index segments. Define shift sₖ = 8k and mask Mₖ = ((2⁸-1) << sₖ). Extract 8‑bit segments:

```
xₖ = ((x ∧ Mₖ) >> sₖ) ∈ {0,1}⁸, yₖ, zₖ analogously
```

For each segment k, select:
```
rₖ = {
  xₖ                          if xₖ = yₖ
  xₖ                          if xₖ = zₖ
  yₖ                          if yₖ = zₖ
  (maj(xₖ,ⱼ,yₖ,ⱼ,zₖ,ⱼ))ⱼ₌₀⁷   otherwise (per‑bit majority within the 8‑bit segment)
}
```

Reconstruct the value by placing segments back:
```
r = Σₖ₌₀^(M-1) (rₖ << sₖ)
```

This matches the implementation with `SEGMENT_SIZE = 8` and the per‑segment short‑circuit on equal pairs.

### Adaptive Routing (enum mapping)
Let π = detectFaultPattern(a,b,c). The router selects:

```
f(π) = {
  bit      if π ∈ {SINGLE_BIT, ADJACENT_BITS}
  burst    if π ∈ {BURST_ERROR, BYTE_ERROR}
  word     if π = WORD_ERROR
  unknown  if π = UNKNOWN
}
```

and returns the corresponding voter result. For UNKNOWN, compute rbit, rword, rburst and return any r equal to a or b or c; otherwise prefer rbit.

### Fault‑Pattern Classification
Pairwise diffs: Δab = a ⊕ b, Δac = a ⊕ c, Δbc = b ⊕ c. Bit counts: hab = ||Δab||₁, etc. Choose a representative pattern Δ: if any pair matches exactly, take the XOR to the outlier; else take the XOR of the closest pair.

Define predicates:
```
Single(Δ): ||Δ||₁ = 1
Adj(Δ): bits of Δ form one contiguous run
Byte(Δ): ∃j: Δ ⊆ byte j
Word(Δ): {||Δ||₁ > W/2 if W ≤ 32; Δ ⊆ lower32 ∨ Δ ⊆ upper32 if W = 64}
Burst(Δ): L(Δ) ≥ 3 ∧ L(Δ)/||Δ||₁ ≥ 0.6, where L(Δ) is longest 1‑run
```

Then:
```
detect(a,b,c) = {
  SINGLE_BIT      if Single(Δ)
  ADJACENT_BITS   if Adj(Δ)
  BYTE_ERROR      if Byte(Δ)
  WORD_ERROR      if Word(Δ)
  BURST_ERROR     if Burst(Δ)
  UNKNOWN         otherwise
}
```

### Floating‑Point Caveat
In the current path, voting is applied to the raw IEEE‑754 bit patterns of floats/doubles; no special exponent/mantissa handling is performed in this voter.

---

## Statistical Convergence and Significance

### Convergence of Success Rates (per method)
Each method's success is a Bernoulli process with rate p. After N trials, the estimator is p̂ = x/N.

**95% confidence interval** (Wald for intuition; Wilson recommended):
```
p̂ ± 1.96√(p̂(1-p̂)/N)  (Wilson has better coverage for small/edge p̂)
```

**Worst‑case half‑width** at p̂ = 0.5: 1.96√(0.25/N). With N = 50,000, half‑width ≈ 0.00438 (±0.44 pp). Thus visible differences ≥ 0.9 pp are significant at 95% in the worst case.

### Sequential Convergence Check (optional)
Monitor running CI width every K trials (e.g., K = 1% of N). Stop when half‑width < target ε or when N reaches max. This yields bounded runtime with guaranteed precision.

### Comparing Methods: Paired vs Unpaired

#### Unpaired (rough, when comparing across independent runs)
Difference of proportions Δ = p̂₁ - p̂₂, 95% CI via pooled SE:
```
SE_pooled = √(p̂(1-p̂)/N₁ + p̂(1-p̂)/N₂), p̂ = (x₁+x₂)/(N₁+N₂)
```

#### Paired (recommended; both methods evaluated on the same trials)
Use McNemar's test on discordant counts b (method A correct, B wrong) and c (A wrong, B correct). With continuity correction:
```
χ² = (|b-c|-1)²/(b+c) ~ χ²₁  (p < 0.05 ⇒ significant)
```

A 95% CI for the paired difference uses b and c (e.g., exact binomial on Pr(b>c) or Newcombe's method for paired data).

### Multiple Scenarios and Methods
Many comparisons inflate Type‑I error. Control false discoveries per family (e.g., all methods within a scenario) with Holm–Bonferroni, or control FDR across all tests with Benjamini–Hochberg.

Report effect sizes alongside p‑values: absolute difference (pp), relative risk p̂₁/p̂₂, or odds ratio with CIs.

### Power and Sample Sizing
For detecting a difference |Δ| between two proportions at α = 0.05 and power 1-β = 0.8 under worst‑case variance, a quick bound per group is:
```
N ≈ 2(z₀.₉₇₅ + z₀.₈)² · 0.25 / Δ² ≈ 3.92/Δ²
```

**Examples**:
- Δ = 1% ⇒ N ≈ 39,200 per method
- Δ = 0.5% ⇒ N ≈ 156,800

### Non‑Binary Metrics
For mean Hamming distance or runtime, use bootstrap CIs (percentile or BCa) over trials, or normal CI if justified by CLT.

### Practical Reporting in This Framework
With N = 50,000 per test, report per‑method Wilson 95% CIs, paired McNemar p‑values vs a baseline (e.g., Standard Voting), and adjust across methods in each scenario (Holm).

Optionally expose a flag to enable sequential stopping at a target precision (e.g., ±0.3 pp).
