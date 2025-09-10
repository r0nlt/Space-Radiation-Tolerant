# GEO Mission Shield Validation — Lab Write‑Up

## Purpose

The purpose of this test is to validate a payload's memory protection and fault tolerance in a 15‑year Geostationary Earth Orbit (GEO) environment, including periodic GEO events (nominal, Van Allen peak, solar storm, eclipse, end‑of‑life, solar maximum). Additionally measuring post shielding error rates and aggregates mission-level reliability

## How the orbit test works (detailed)

High‑level flow
- Scenarios: A fixed set of GEO scenarios is exercised (NOMINAL, VAN_ALLEN_PEAK, SOLAR_STORM, ECLIPSE, END_OF_LIFE, SOLAR_MAXIMUM). Each scenario has environment parameters (particle flux, temperature), upset probabilities per error type, optional storm probability, and a time_fraction (share of mission time).
- Trials: For each scenario and data type (float, double, int32, int64), the harness runs NUM_TRIALS_PER_TEST random trials to estimate a post‑shielding average failure rate λ_avg for that scenario.
- Aggregation: Mission failure rate is λ_total = Σ_s (λ_avg[s] × time_fraction[s]). The 15‑year reliability is R = exp(−λ_total × hours_in_15_years).

What happens inside a trial
- Value selection: Draw a random input value for the active data type.
- Replication: Create three logical replicas (TMR‑like) of the value.
- Event context: Use the active scenario’s environment (flux, LET, temperature, storm flag) to parameterize corruption selection and physics cues.
- Corruption selection: Choose an error model by scenario‑specific probabilities:
  - SINGLE_BIT (isolated SEU)
  - MULTI_BIT (scattered bits)
  - BURST (adjacent run; typical MBU)
  - WORD (whole‑word corruption)
  - GEO‑specific compound tests (e.g., SOLAR_STORM, VAN_ALLEN_EXPOSURE) that amplify clustering/burst likelihood and rate.
- Shielding: Compute a reduction factor from the configured shield (single material or multilayer stack). This attenuates effective upset likelihoods before application.
- Injection: Apply the selected corruption pattern to replicas (one or more replicas, depending on the test’s mapping for the scenario/type).
- Protection: Run protection methods (standard/bit/burst/word/weighted/adaptive voting; optional ETMR/temporal redundancy). Record whether reconstruction equals the original value.
- Diagnostics: Update per‑type injected/detected/corrected, mean Hamming distance on disagreements, and physics counters where enabled.

How events map to “realistic” behavior
- Nominal GEO: Baseline flux; higher proportion of SINGLE_BIT; low BURST/WORD.
- Van Allen peak: Elevated flux and clustering tendency; increased MULTI_BIT/BURST rates.
- Solar storm: Rare but intense; activates higher BURST/WORD rates and compound patterns; gated by solar_storm_probability.
- Eclipse: Lower temperature; temperature‑dependent critical charge tends to reduce flips versus hot conditions; probabilities adjusted via physics hooks.
- End‑of‑life: Represents aging/dose accumulation; small adjustment to susceptibility in projections.
- Solar maximum: Persistent elevation of background flux and LET; higher baseline upset probabilities than nominal.

Timeline vs weighting (important)
- The harness does not simulate a minute‑by‑minute 15‑year timeline. Instead, each scenario is exercised with many trials to estimate λ_avg, and mission time_fraction weights those estimates in λ_total. This preserves long‑mission realism while keeping runs tractable and repeatable.

## Why this sim approximates a 15‑year GEO mission
- Scenario realism: Distinct GEO regimes (nominal, belts, storms, eclipse, EOL, solar maximum) have explicit time_fraction. Rare/intense events (storms) are gated by solar_storm_probability.
- Post‑shielding physics: Error probabilities are attenuated by material stacks using areal density (g/cm²) and material properties (including LET‑related terms and temperature influence), so rates reflect actual shielding choices.
- Rate aggregation: For each scenario a post‑shielding average failure rate λ_avg is estimated via large Monte Carlo (50k trials per test). Mission risk is λ_total = Σ_s (λ_avg[s] × time_fraction[s]). Reliability is then R = exp(−λ_total × hours_in_15_years), the standard Poisson no‑event probability over 15 years.
- Coverage of error modes: Single, multi, burst, and word‑level corruptions plus GEO‑specific compound tests are injected, then detection and correction are measured across several protection methods to capture realistic failure/repair pathways.
- Cross‑type, cross‑data validation: Repeated across multiple data types (float, double, int32, int64) to bound type‑specific sensitivities.
- Diagnostics that map to mission knobs: The limiting table decomposes λ_total by scenario, identifying where incremental shielding (or stack changes) yields maximum reliability gain.
- Conservative, repeatable method: With fixed seeds and Release builds estimates are reproducible; raising trial counts tightens confidence intervals. Where the model is simplified (e.g., not a full transport), assumptions are conservative and documented.

## What the test exercises
- GEO scenario profiles with event probabilities (nominal, storms, belts, eclipse, EOL)
- Systematic fault injection (single, multi, burst, word) plus GEO‑specific compound tests
- Protection methods (standard/bit/burst/word/weighted/adaptive voting, fast‑bit correction; optional ETMR/temporal redundancy)
- Physics hooks (temperature effects, LET‑driven phenomena via `QuantumEnhancedRadiation`)
- Shielding attenuation using the material database

## Key outputs to read
- Mission reliability (Poisson, aggregated): PASS/FAIL for ≥95% over 15 years
- λ_total (per hour): mission failure rate; target ≤ ~3.9e‑7/h for ≥95%/15y
- Limiting scenarios (with `--limiting`): scenario contributions ranked by λ_avg × time_fraction
- Per‑method success, SDC rate, mean Hamming distance (diagnostic quality of correction)

## Running the test
Binary path may vary; examples assume a Release build in `build-release`.

### Single‑material thickness
```
./build-release/test/verification/geo_mission_shield_validation \
  --material=Aluminum --shield-mm=1000
```

### Multilayer (graded‑Z) stacks
- Global stack (applies to all scenarios):
```
./build-release/test/verification/geo_mission_shield_validation \
  --shield-stack="Polyethylene:50,Tungsten:5,Polyethylene:50" --limiting
```
- Per‑scenario stack:
```
./build-release/test/verification/geo_mission_shield_validation \
  --scenario-stack="GEO_NOMINAL:Polyethylene:50;Tungsten:5;Polyethylene:50" --limiting
```

Interpretation: The effective reduction factor equals the product of each layer’s attenuation (computed from the material database). The console prints base material/thickness and the effective reduction factor; when extremely small, it may print as 0.000 due to precision.

## Single‑layer vs multilayer shielding

Why single‑layer fell short in this model
- Diminishing returns: A single material (e.g., Aluminum) shows steep early gains then small improvements per added mm; meter‑scale thickness still leaves λ_total too high.
- Secondary production trade‑offs: High‑Z layers alone (e.g., Tungsten) can increase secondaries (bremsstrahlung, neutrons) unless paired with low‑Z moderation.
- Mass efficiency: Meeting ≥95%/15y with one material often demands impractical mass compared to layered designs.

Why a graded‑Z stack helps
- Low‑Z, H‑rich outer (e.g., Polyethylene) reduces proton dose and moderates secondaries.
- Mid high‑Z (e.g., Tungsten) attenuates electrons/γ; kept thin to limit secondary generation.
- Low‑Z inner (e.g., Polyethylene) captures/thermalizes remaining secondaries.
- Ceramic option (e.g., Boron Carbide) offers favorable per‑mass attenuation and lower neutron yield than many high‑Z metals.

Tuning checklist when FAIL persists
- Increase low‑Z moderation, reduce high‑Z thickness.
- Try ceramics (Boron Carbide) in place of or in addition to high‑Z metal.
- Target only limiting scenarios with `--scenario-stack` to avoid over‑thickening everywhere.
- Use `--limiting` to verify λ contribution shifts after changes, then refine.

### Auto‑search minimal thickness
Binary search the minimum thickness for a material to reach ≥95%/15y:
```
./build-release/test/verification/geo_mission_shield_validation \
  --auto-thickness --material=Aluminum --auto-min=400 --auto-max=1000 --auto-tol=0.5 --limiting
```

## Example: 5000 mm Aluminum (prototype)
- Command:
```
--material=Aluminum --shield-mm=5000 --limiting
```
- Observed outcome in this model: mission reliability remains FAIL with λ_total ≈ 2.75e‑3/h (≫ 3.9e‑7/h), and limiting scenarios dominated by nominal GEO contributions. This indicates Aluminum alone is insufficient in this configuration.

## Recommendations when FAIL
- Use `--limiting` to identify bottlenecks, then:
  - Increase thickness only for the limiting scenario with `--scenario-shield=SCENARIO:MM`, or
  - Switch to hydrogen‑rich or ceramic materials (e.g., Polyethylene, Boron Carbide)
  - Adopt graded‑Z stacks (low‑Z / high‑Z / low‑Z) with the new stack flags
- Use `--auto-thickness` to bracket and converge on minimal viable thickness per material.

## Material database notes
The material database provides density and attenuation parameters used to compute per‑layer factors. Available entries include Aluminum, Polyethylene, Water, Lead, Boron Carbide, and Tungsten (as used in scenarios). Names are case‑sensitive; match exactly when specifying layers.

## Limitations
- Prototype attenuation; not a full transport code. Extremely small factors may print as 0.000 due to 3‑decimal formatting.
- Mission reliability uses average per‑scenario post‑shielding λ weighted by time_fraction; ensure scenario fractions reflect your mission time profile.

- Scenario probabilities are representative, not mission‑specific by default. Calibrate single/multi/burst/word rates and storm probabilities to your payload and epoch.
- Secondary particle production (neutrons, bremsstrahlung) is approximated; detailed transport is out of scope. Keep high‑Z layers thin and validate empirically when possible.
- Temperature effects are first‑order via Qcrit(T); extreme thermal cycles and device‑specific behaviors may require tuning.
- Per‑data‑type behavior (float/double/int) is covered statistically; device/layout specifics (e.g., bit cell topology) are abstracted.
- Confidence grows with trials; increase NUM_TRIALS_PER_TEST when comparing close designs.

## Reproducibility
- Default RNG is nondeterministic; set a fixed seed in your harness for repeatability.
- Build in Release (`-O3 -DNDEBUG`) for consistent runtimes.


EO mission validation completed in 109 seconds.

================================================================================
                    GEO MISSION VALIDATION SUMMARY
================================================================================

Average Success Rates Across All GEO Tests:
------------------------------------------------------------

## How to read the console summary (first‑time guide)

Use this checklist top‑to‑bottom; it tells you at a glance whether the mission passes and why.

1) Mission reliability (Poisson, aggregated)
- What it is: Your overall 15‑year mission reliability computed from a single failure‑rate number λ_total.
- PASS/FAIL rule: PASS if ≥ 95.00% over 15 years; otherwise FAIL.
- How it’s computed: Average each scenario’s post‑shielding error rate (λ_avg), weight by time_fraction, sum to λ_total, then compute R = exp(−λ_total × hours_in_15_years).

2) λ_total (per hour)
- What it is: The single mission failure‑rate number (events per hour) that drives the reliability above.
- Target: ≤ ~3.9e‑7 1/h to reach ≥95% over 15 years.
- If this is large, no amount of voting will change PASS/FAIL; you need better/layered shielding or to thicken the bottleneck scenarios.

3) Top limiting scenarios (with --limiting)
- Scenario: The GEO condition (e.g., NOMINAL, VAN_ALLEN_PEAK) or a sub‑test variant.
- λ_avg (1/h): Average post‑shielding error rate for that scenario.
- λ_contrib: λ_avg × time_fraction; how much this scenario adds to λ_total.
- % of total: Contribution share. The top rows are your bottlenecks to optimize first.

4) Average Success Rates Across All GEO Tests
- Shows how often each protection method reconstructed the correct value across all trials.
- Close to 100% means the protection is strong; it does not guarantee mission PASS if λ_total is still high.

5) Memory Management Features / Advanced TMR Methods
- On/off or success indicators for system‑level protections (scrubbing, enhanced TMR, etc.). High values help quality but cannot compensate for high λ_total alone.

6) Advanced Error Analysis
- Mean Hamming Distance: Average number of differing bits when a vote disagrees with the original; lower is better.
- Silent Data Corruption: Fraction of undetected errors; should be ~0%.
- MTBF: Mean time between failures from observed rates (hours).
- 30‑Day / 1‑Year Reliability: Short‑term reliability snapshots; informative but the PASS rule is based on 15 years.

7) Corruption Detection/Correction By Type
- Injected: How many trials actually applied this error type.
- Detected: Trials where replicas disagreed (the system noticed corruption).
- Corrected: Trials where protection fully restored the original.
- High detected% and corrected% are good; if λ_total is still high, shielding is the limiting factor.

8) Breakpoint Analysis (collapse intensity)
- The intensity at which a method’s success falls to ≤1%. It’s a stress test indicator; not used for PASS/FAIL.

Where do I see shielding?
- Per‑scenario lines printed earlier in the log show: Material, Thickness, Reduction factor. Very small factors print as 0.000 due to rounding—trust λ_total and the PASS/FAIL line.

Quick decision flow
- PASS? You’re done.
- FAIL and λ_total dominated by one scenario? Increase thickness only for that scenario (or apply a multilayer stack) and re‑check.
- FAIL everywhere? Try H‑rich/ceramic materials or graded‑Z stacks; then use --auto-thickness to find minimally sufficient thickness.

## Monte Carlo correction vs mission reliability (what differs)

- Monte Carlo validation (per‑event quality):
  - Purpose: Check if protection methods detect and correct injected corruptions within each scenario.
  - Metrics: Per‑method success rates, SDC, mean Hamming distance, per‑type injected/detected/corrected.
  - Insight: “100%” here means when an upset occurs, correction restored the original value in those trials.

- Mission shielding validation (long‑term risk):
  - Purpose: Estimate how often upsets occur after shielding across all scenarios over 15 years.
  - Metrics: Scenario post‑shielding λ_avg, mission λ_total = Σ_s (λ_avg[s] × time_fraction[s]), and 15‑year reliability R = exp(−λ_total × hours_15y) with PASS if R ≥ 95%.
  - Insight: Even with near‑perfect correction per event, reliability can FAIL if the event frequency (λ_total) is high.

- How to reconcile “100% but FAIL”:
  - Reduce λ_avg with better/layered shielding, especially in limiting scenarios shown by the ranking table.
  - Optionally ensure λ aggregation subtracts correction coverage for all counted rate types so successful correction lowers λ_total.

## Detailed explanation of each results block

Average Success Rates Across All GEO Tests
- Meaning: Percent of trials where each protection method reconstructed the exact original value across all scenarios and data types.
- Read: Values near 100% indicate strong correction capability. Lower values highlight patterns that are harder for a given method (e.g., burst vs word).

Memory Protection
- Aligned Memory: Success rate for alignment‑based protection.
- Protected Value: Success for the protected value container (may be inactive in some runs).

Advanced TMR Methods
- Enhanced/Health‑Weighted/Temporal/Physics‑Driven: Indicators for extended redundancy policies beyond basic voting; useful for configuration validation and quality insight.

Memory Management Features
- Scrubber/Effectiveness, Radiation Mapped/Static/Unified managers: Feature coverage and effectiveness; helpful to confirm run settings.

Advanced Error Analysis
- Mean Hamming Distance (bits): Average severity of incorrect votes.
- Silent Data Corruption (%): Undetected error fraction; target ~0%.
- MTBF (hours), Expected Lifetime (years), 30‑Day/1‑Year Reliability: Convenience reliability snapshots; mission PASS is judged on 15 years.

Corruption Detection/Correction By Type
- Injected: Trials where a given corruption type was applied.
- Detected: Trials where replicas disagreed (error observed by the system).
- Corrected: Trials where protection exactly restored the original value.
- Read: High detected% and corrected% indicate robust protection; shielding and scenario mix determine how often each type appears.

Breakpoint Analysis (collapse intensity; success ≤1%)
- Meaning: Stress level where a method’s success drops to ≤1% for each type; useful for margin studies, not direct PASS/FAIL.

Reliability Threshold Check (95% over 15 years)
- Mission reliability (Poisson): R = exp(−λ_total × hours_15y). PASS if ≥95%.
- λ_total (per hour): Σ_s (λ_avg[s] × time_fraction[s]). Primary number to drive down with shielding/material choices.

Top limiting scenarios (by λ contribution)
- Scenario, λ_avg (post‑shield), λ_contrib = λ_avg × time_fraction, % of total.
- Read: Focus on the top contributors first; adjust thickness or stacks globally or per‑scenario; re‑run and compare.
STANDARD PROTECTION METHODS:
  Adaptive Voting     : 93.3646%
  Bit-Level Voting    : 100.0000%
  Burst-Error Voting  : 100.0000%
  Fast Bit Correction : 100.0000%
  Pattern Detection   : 68.3368%
  Standard Voting     : 100.0000%
  Weighted Voting     : 97.6597%
  Word-Error Voting   : 100.0000%

MEMORY PROTECTION:
  Aligned Memory      : 25.0000% (1.0000/4.0000)
  Protected Value     : 0.0000% (0.0000/4.0000)

ADVANCED TMR METHODS:

MEMORY MANAGEMENT FEATURES:
  Enhanced TMR             : 100.0000%
  Health-Weighted TMR      : 100.0000%
  Physics-Driven Protection: 100.0000%
  Temporal Redundancy      : 100.0000%
  Memory Scrubber          : 100.0000%
  Radiation Mapped Allocator: 100.0000%
  Scrubbing Effectiveness  : 100.0000%
  Static Allocator         : 100.0000%
  Unified Memory Manager   : 0.0000%

GEO-SPECIFIC PROTECTION SCENARIOS:
  Eclipse Transition       : 100.0000%
  Long Duration Stability  : 100.0000%
  Solar Storm Survival     : 100.0000%
  Temperature Cycling      : 100.0000%
  Van Allen Recovery       : 100.0000%

ADVANCED ERROR ANALYSIS:
  Mean Hamming Distance     : 0.00 bits
  Silent Data Corruption    : 0.0000%
  15-Year Mission Reliability: 0.000000%
  MTBF (hours)              : 2256.2
  Expected Lifetime (years) : 0.26
  30-Day Reliability        : 55.2264%
  1-Year Reliability        : 5.4583%
  Quantum Tunneling Events  : 400000 total

CORRUPTION DETECTION/CORRECTION BY TYPE:
  ECLIPSE_TRANSITION: injected=44, detected=44 (100.00%), corrected=44 (100.00%)
  END_OF_LIFE     : injected=316, detected=316 (100.00%), corrected=316 (100.00%)
  LONG_DURATION   : injected=105, detected=105 (100.00%), corrected=105 (100.00%)
  SOLAR_STORM     : injected=8, detected=8 (100.00%), corrected=8 (100.00%)
  TEMPERATURE_CYCLING: injected=18, detected=18 (100.00%), corrected=18 (100.00%)

BREAKPOINT ANALYSIS (collapse intensity; success ≤ 1.00%):
  Standard    : SINGLE_BIT=0.10, MULTI_BIT=0.10, BURST=0.10, WORD=0.10
  Bit-Level   : SINGLE_BIT=0.10, MULTI_BIT=0.10, BURST=0.10, WORD=0.10
  Burst-Error : SINGLE_BIT=0.10, MULTI_BIT=0.10, BURST=0.10, WORD=0.10
  Word-Error  : SINGLE_BIT=0.10, MULTI_BIT=0.10, BURST=0.10, WORD=0.10
  Adaptive    : SINGLE_BIT=0.10, MULTI_BIT=0.10, BURST=0.10, WORD=0.10
  Weighted    : SINGLE_BIT=0.10, MULTI_BIT=0.10, BURST=0.10, WORD=0.10
  Fast-Bit    : SINGLE_BIT=0.10, MULTI_BIT=0.10, BURST=0.10, WORD=0.10

RELIABILITY THRESHOLD CHECK (95.00% over 15 years):
  Mission reliability (Poisson, aggregated): 0.000000% -> FAIL
  λ_total (per hour): 0.081022

Top limiting scenarios (by λ contribution):
----------------------------------------------------------------------------
Scenario                        λ_avg (1/h)      λ_contrib      % of total
----------------------------------------------------------------------------
GEO_NOMINAL                         0.011033        0.008826          10.89
GEO_NOMINAL_VAN_ALLEN               0.011033        0.008826          10.89
GEO_NOMINAL_TEMPERATURE             0.011033        0.008826          10.89
GEO_NOMINAL_SOLAR                   0.011033        0.008826          10.89
GEO_NOMINAL_SINGLE                  0.011033        0.008826          10.89
GEO_NOMINAL_MULTI                   0.011033        0.008826          10.89
GEO_NOMINAL_LONG                    0.011033        0.008826          10.89
GEO_NOMINAL_END_OF                  0.011033        0.008826          10.89
GEO_NOMINAL_ECLIPSE                 0.011033        0.008826          10.89
GEO_VAN_ALLEN_PEAK_TEMPERATURE        0.001103        0.000088           0.11
GEO_VAN_ALLEN_PEAK_SOLAR            0.001103        0.000088           0.11
GEO_VAN_ALLEN_PEAK_SINGLE           0.001103        0.000088           0.11
GEO_VAN_ALLEN_PEAK_MULTI            0.001103        0.000088           0.11
GEO_VAN_ALLEN_PEAK_LONG             0.001103        0.000088           0.11
GEO_VAN_ALLEN_PEAK_END_OF           0.001103        0.000088           0.11
GEO_VAN_ALLEN_PEAK_ECLIPSE          0.001103        0.000088           0.11
GEO_VAN_ALLEN_PEAK                  0.001103        0.000088           0.11
GEO_VAN_ALLEN_PEAK_VAN_ALLEN        0.001103        0.000088           0.11
GEO_ECLIPSE                         0.000965        0.000068           0.08
GEO_ECLIPSE_ECLIPSE                 0.000965        0.000068           0.08
GEO_ECLIPSE_END_OF                  0.000965        0.000068           0.08
GEO_ECLIPSE_LONG                    0.000965        0.000068           0.08
GEO_ECLIPSE_MULTI                   0.000965        0.000068           0.08
GEO_ECLIPSE_SINGLE                  0.000965        0.000068           0.08
GEO_ECLIPSE_SOLAR                   0.000965        0.000068           0.08
GEO_ECLIPSE_TEMPERATURE             0.000965        0.000068           0.08
GEO_ECLIPSE_VAN_ALLEN               0.000965        0.000068           0.08
GEO_END_OF_LIFE_VAN_ALLEN           0.000500        0.000015           0.02
GEO_END_OF_LIFE_TEMPERATURE         0.000500        0.000015           0.02
GEO_END_OF_LIFE_SOLAR               0.000500        0.000015           0.02
GEO_END_OF_LIFE_SINGLE              0.000500        0.000015           0.02
GEO_END_OF_LIFE_MULTI               0.000500        0.000015           0.02
GEO_END_OF_LIFE_LONG                0.000500        0.000015           0.02
GEO_END_OF_LIFE_END_OF              0.000500        0.000015           0.02
GEO_END_OF_LIFE_ECLIPSE             0.000500        0.000015           0.02
GEO_END_OF_LIFE                     0.000500        0.000015           0.02
GEO_SOLAR_STORM                     0.000372        0.000004           0.00
GEO_SOLAR_STORM_ECLIPSE             0.000372        0.000004           0.00
GEO_SOLAR_STORM_END_OF              0.000372        0.000004           0.00
GEO_SOLAR_STORM_LONG                0.000372        0.000004           0.00
GEO_SOLAR_STORM_MULTI               0.000372        0.000004           0.00
GEO_SOLAR_STORM_SINGLE              0.000372        0.000004           0.00
GEO_SOLAR_STORM_SOLAR               0.000372        0.000004           0.00
GEO_SOLAR_STORM_TEMPERATURE         0.000372        0.000004           0.00
GEO_SOLAR_STORM_VAN_ALLEN           0.000372        0.000004           0.00
GEO_SOLAR_MAXIMUM_VAN_ALLEN         0.000147        0.000001           0.00
GEO_SOLAR_MAXIMUM_TEMPERATURE        0.000147        0.000001           0.00
GEO_SOLAR_MAXIMUM_SOLAR             0.000147        0.000001           0.00
GEO_SOLAR_MAXIMUM_SINGLE            0.000147        0.000001           0.00
GEO_SOLAR_MAXIMUM_MULTI             0.000147        0.000001           0.00
GEO_SOLAR_MAXIMUM_LONG              0.000147        0.000001           0.00
GEO_SOLAR_MAXIMUM_END_OF            0.000147        0.000001           0.00
GEO_SOLAR_MAXIMUM_ECLIPSE           0.000147        0.000001           0.00
GEO_SOLAR_MAXIMUM                   0.000147        0.000001           0.00
----------------------------------------------------------------------------

------------------------------------------------------------

## Glossary (symbols and terms)

- λ_total (1/h): Mission failure rate per hour. Note: “1/h” means per hour.
- R (reliability): Overall mission reliability. Computed as R = exp(−λ_total × hours_in_15_years).
- ≥, ≤: “greater/less than or equal to.”  ≫, ≪: “much greater/less than.”
- 15y hours: Hours in 15 years; 15 × 365.25 × 24 ≈ 131,487 h.
- PASS threshold: Requirement R ≥ 0.95 (≥95% over 15 years).
- Reduction factor: Unitless attenuation (0..1) from shielding; lower is more attenuation. Values under 0.0005 print as 0.000 due to rounding.
- Areal density (g/cm²): thickness_cm × material_density; the basis for attenuation in the model.
- LET (MeV·cm²/mg): Linear Energy Transfer; higher LET generally means higher upset probability.
- MTBF (hours): Mean time between failures; approximate inverse of λ_total under Poisson assumptions.
- SDC: Silent Data Corruption; undetected error that escapes the protections.
- Hamming distance (bits): Number of differing bits between two values; lower means smaller deviations.
- H‑rich: Hydrogen‑rich materials (e.g., Polyethylene) that help moderate secondaries.
- Graded‑Z stack: Layered shield combining low‑Z/high‑Z/low‑Z materials to balance attenuation and secondaries.
- mm: Millimeters of physical thickness for a layer.
- MeV / eV: Particle energy units (mega‑electronvolt / electronvolt).
- %: Percent; success/reliability figures are proportions expressed as percentages.

## Solar events in GEO and how the test models their frequency

What “how often” means here
- In this test, how often a condition occurs is controlled by each scenario’s time_fraction (share of mission time) and, where applicable, by solar_storm_probability (chance a storm is active during that scenario).

Scenarios and typical occurrence in this model
- Nominal GEO: Majority of mission time (time_fraction set high); represents routine operations.
- Van Allen peak: Short exposure intervals with elevated belts influence; small time_fraction.
- Solar storm: Rare but intense; gated by solar_storm_probability and more impactful during solar‑maximum periods.
- Eclipse: Seasonal windows (temperature shifts during Earth shadowing); modeled by a dedicated scenario with a small time_fraction.
- End‑of‑life (EOL): Represents late‑mission conditions; small time_fraction but included in 15‑year aggregation.
- Solar maximum: Elevated background activity period; modeled as its own scenario.

How this propagates into mission reliability
- For each scenario, the test measures a post‑shielding average failure rate λ_avg. The mission rate is λ_total = Σ_s (λ_avg[s] × time_fraction[s]).
- A scenario that is either frequent (high time_fraction) or intense (high λ_avg) will dominate λ_total. The limiting table shows this explicitly.

## How often corruptions happen (by type)

## Path to 15‑year PASS with better shielding

Target
- PASS at ≥95% over 15 years corresponds to λ_total ≤ ~3.9×10⁻⁷ 1/h. Use the limiting table to see where λ_total originates.

Shielding strategy (graded‑Z)
- Outer low‑Z, H‑rich (e.g., Polyethylene) to reduce protons and moderate secondaries.
- Thin mid high‑Z (e.g., Tungsten) to attenuate electrons/γ; keep thin to limit secondary production.
- Inner low‑Z (e.g., Polyethylene) to capture remaining secondaries.
- Ceramic option (e.g., Boron Carbide) in place of or alongside high‑Z for better per‑mass performance and lower neutron yield.

Tuning procedure
- Identify bottlenecks: run with --limiting and note top λ contributors.
- Localize thickness: prefer --scenario-stack for those scenarios before increasing global mass.
- Iterate: adjust low‑Z up and high‑Z down until λ_contrib drops meaningfully; verify λ_total.
- Converge with search: use --auto-thickness per material or per stack family within a plausible bracket.

Verification checklist
- After changes, confirm:
  - λ_total ≤ 3.9e‑7 1/h and “Mission reliability … -> PASS”.
  - SDC remains ~0%; method success rates stable.
  - Limiting table shows reduced contributions in previously dominant scenarios.

What drives per‑type frequency
- Within a scenario, the test randomly selects an error type using that scenario’s probabilities: single_bit_prob, multi_bit_prob, burst_error_prob, word_error_prob (plus GEO‑specific compound tests like SOLAR_STORM and VAN_ALLEN_EXPOSURE).
- Shielding scales the effective rates before they contribute to λ_avg; protections determine how many are detected and corrected.

How to read your run’s per‑type counts
- Injected: Approximates how frequently that error type was selected across all scenarios, weighted by time spent there.
- Detected: Of those injected, how many created replica disagreements (the system noticed an upset).
- Corrected: Of those injected, how many were fully repaired by the protection logic.

Putting it together
- High injected for a type means it’s common in the chosen scenario mix; high detected and corrected means the protection layer handles it well.
- Even with near‑perfect correction, if λ_total stays high, shielding (not voting) is the limiting factor—reduce λ_avg with better/thicker or multilayer shields, or target only the limiting scenarios.
