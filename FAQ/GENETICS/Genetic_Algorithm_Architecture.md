# 🧬 Genetic Algorithm Architecture: Adaptive Mutation System

## Overview

### Integration with Advanced Quality Diversity (MAP-Elites + Novelty) (New)

- The evolutionary loop now optionally integrates a MAP-Elites archive that:
  - Computes a 6D physics-informed behavior descriptor per evaluated config
  - Updates an elite per behavioral cell using composite fitness (preservation + novelty)
  - Samples elites each generation and replaces the worst K individuals to inject diversity
- Novelty search uses k-nearest neighbors (k=5) in behavioral space to reward exploration.
- Per-generation logs include QD coverage, occupied cells, and elites injected.

Configuration:
```cpp
// Enable in code or via example CLI flag --adv-qd
searcher.enableAdvancedQualityDiversity(true);
```

For a deeper dive on the archive, novelty scoring, and GA integration details, see `FAQ/GENETICS/Quality_Diversity_AutoArch_Integration.md`.

Early-stage interpretation:
- Tiny percent coverage is expected with large archives (6D×10=1e6 cells);
  track rising occupied cells and non-zero elites injected per generation.

This document provides a comprehensive visual and technical breakdown of the **Genetic Algorithm** implementation within the **Adaptive Mutation System** for radiation-tolerant machine learning architecture optimization.

---

## 🎯 High-Level Algorithm Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Initialize    │───▶│   Evaluate       │───▶│   Selection     │
│   Population    │    │   Fitness        │    │   (Tournament)  │
│   (Random)      │    │   Functions      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Adaptive      │───▶│   Crossover      │───▶│   Mutation      │
│   Control       │    │   (Recombine)    │    │   (Perturb)     │
│   (Diversity    │    │                  │    │                 │
│    Monitoring)  │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                               │
         └───────────────────────────────────────────────┘
                                 ▼
                    ┌──────────────────┐
                    │   Next           │
                    │   Generation     │
                    │   (Repeat)       │
                    └──────────────────┘
                                 │
                    ┌──────────────────┤
                    │   Convergence     │
                    │   Check           │
                    └──────────────────┘
                                 │
                    ┌──────────────────▼──────────────────┐
                    │         TERMINATION CRITERIA         │
                    │  • Max Generations Reached           │
                    │  • Fitness Plateau Detected          │
                    │  • Diversity Below Threshold         │
                    │  • Time/Resource Limit Exceeded      │
                    └─────────────────────────────────────┘
```

---

## 🔧 New GA Features (Updated)

### Configurable Crossover Strategies

- Uniform (default) and Single-Point crossover are supported.
- Single-Point preserves input/output layer sizes and swaps a contiguous hidden segment.

Usage:

```cpp
// Probability 0.8, Uniform crossover (default)
searcher.setCrossoverSettings(0.8, AutoArchSearch::CrossoverStrategy::UNIFORM);

// Force Single-Point crossover for experiments
searcher.setCrossoverSettings(1.0, AutoArchSearch::CrossoverStrategy::SINGLE_POINT);
```

### Diversity-Preserving Mutation Operator

- Adds a mutation mode that biases changes away from population modes:
  - Pushes hidden layer widths toward less common options
  - Flips residuals probabilistically
  - Cycles protection levels
  - Moves dropout away from current value

This operator is integrated into the adaptive controller and participates in operator-credit learning.

### Random Immigrants Injection

- When population diversity collapses (≈ below half the configured threshold), a fraction of worst individuals are replaced with random "immigrants" to re-start exploration.

Usage:

```cpp
// Enable injection of ~10% random immigrants when diversity collapses
searcher.setRandomImmigrants(true, 0.10);
```

### Genetics Metrics CSV (Selection/Fitness/Diversity)

- Per-generation CSV is emitted with key theory-aligned metrics:

Header:

```
generation,best_preservation,mean_fitness,fitness_variance,diversity,crossover_rate,crossover_count,population_size
```

Usage and output location:

```cpp
// Bare filenames are written under results/genetic_algorithm/
searcher.setGeneticsMetricsFile("run_metrics.csv");
// → results/genetic_algorithm/run_metrics.csv
```

Notes:
- Diversity is the normalized mean pairwise configuration distance in [0,1].
- Fitness variance helps track selection intensity and convergence.
- `crossover_count` records how often crossover was applied per generation at the configured `crossover_rate`.

---

## 🏗️ Detailed Component Architecture

### 1. Population Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             POPULATION                                      │
│                           (Generation N)                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Individual 1: NetworkConfig {                                              │
│    • layer_sizes: [8, 64, 128, 2]         // Architecture genome           │
│    • dropout_rate: 0.5                    // Parameter gene                 │
│    • has_residual_connections: true       // Structural gene               │
│    • protection_level: FULL_TMR          // Protection gene                │
│    • fitness_score: 98.73%               // Performance metric             │
│  }                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────┐
│  Individual 2: NetworkConfig {                                              │
│    • layer_sizes: [8, 32, 256, 64, 2]     // Different architecture        │
│    • dropout_rate: 0.3                    // Different parameters           │
│    • has_residual_connections: false      // Different structure           │
│    • protection_level: CHECKSUM_ONLY     // Different protection           │
│    • fitness_score: 92.45%               // Different performance          │
│  }                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                        ...
┌─────────────────────────────────────────────────────────────────────────────┐
│  Individual N: NetworkConfig {                                              │
│    • layer_sizes: [8, 128, 1]             // Minimal architecture          │
│    • dropout_rate: 0.4                    // Optimized parameters           │
│    • has_residual_connections: false      // Simplified structure          │
│    • protection_level: SELECTIVE_TMR     // Balanced protection            │
│    • fitness_score: 95.66%               // Good performance               │
│  }                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Genome Representation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GENOME STRUCTURE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  NetworkConfig Genome (Chromosome)                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Gene 1: Layer Sizes Array                                                 │
│  ├─ Locus 1.1: Input Layer Size     (8)                                   │
│  ├─ Locus 1.2: Hidden Layer 1 Size  (64)                                  │
│  ├─ Locus 1.3: Hidden Layer 2 Size  (128)                                 │
│  └─ Locus 1.4: Output Layer Size    (2)                                   │
│                                                                             │
│  Gene 2: Dropout Rate                                                      │
│  ├─ Locus 2.1: Dropout Probability  (0.0 ≤ x ≤ 1.0)                       │
│                                                                             │
│  Gene 3: Residual Connections                                              │
│  ├─ Locus 3.1: Boolean Flag         (true/false)                          │
│                                                                             │
│  Gene 4: Protection Level                                                  │
│  ├─ Locus 4.1: Protection Strategy  (NONE, CHECKSUM, TMR, SPACE_OPT)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Fitness Function

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FITNESS EVALUATION                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Data    │───▶│   Neural         │───▶│   Baseline      │
│   (Training)    │    │   Network        │    │   Accuracy      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Radiation     │───▶│   Neural         │───▶│   Radiation     │
│   Environment   │    │   Network        │    │   Accuracy      │
│   Simulation    │    │   (with errors)  │    │   (with faults) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                    ┌────────────────────────────────────▼────────────────────┐
                    │                ACCURACY PRESERVATION                     │
                    │                                                          │
                    │  Fitness Score = (Radiation_Accuracy / Baseline_Accuracy) │
                    │                   × 100%                                 │
                    │                                                          │
                    │  Higher = Better Radiation Tolerance                     │
                    └──────────────────────────────────────────────────────────┘
```

### 4. Selection Mechanism

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TOURNAMENT SELECTION                               │
└─────────────────────────────────────────────────────────────────────────────┘

Population: [A:98.7%, B:95.6%, C:92.4%, D:90.1%, E:88.3%, F:85.2%]

Tournament 1 (Size 3):
├── Random Select: A(98.7%), C(92.4%), E(88.3%)
├── Winner: A(98.7%) → Selected for reproduction
└── Losers: C, E → Return to population

Tournament 2 (Size 3):
├── Random Select: B(95.6%), D(90.1%), F(85.2%)
├── Winner: B(95.6%) → Selected for reproduction
└── Losers: D, F → Return to population

Result: Parents [A, B] selected for genetic operations
```

### 5. Crossover Operation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             CROSSOVER                                       │
└─────────────────────────────────────────────────────────────────────────────┘

Parent 1: [8, 64, 128, 2] | dropout: 0.5 | residual: true  | prot: TMR
Parent 2: [8, 32, 256, 64, 2] | dropout: 0.3 | residual: false | prot: CHECKSUM

Crossover Point: Random selection between genes

Option 1 - Single Point Crossover:
├── Child 1: [8, 64, 128, 2] | dropout: 0.5 | residual: false | prot: CHECKSUM
├── Child 2: [8, 32, 256, 64, 2] | dropout: 0.3 | residual: true | prot: TMR

Option 2 - Uniform Crossover:
├── Child 1: [8, 64, 256, 64, 2] | dropout: 0.3 | residual: true | prot: CHECKSUM
├── Child 2: [8, 32, 128, 2] | dropout: 0.5 | residual: false | prot: TMR
```

### 6. Mutation Operation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MUTATION                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Original: [8, 64, 128, 2] | dropout: 0.5 | residual: true | prot: TMR

Mutation Rate: 0.2 (20% chance per gene)

Gene 1 (Layer Sizes) - MUTATION TRIGGERED:
├── Original: [8, 64, 128, 2]
├── Random change to layer 2: 128 → 96
└── Result: [8, 64, 96, 2]

Gene 2 (Dropout) - NO MUTATION:
├── Original: 0.5
└── Unchanged: 0.5

Gene 3 (Residual) - MUTATION TRIGGERED:
├── Original: true → false
└── Result: false

Gene 4 (Protection) - NO MUTATION:
├── Original: TMR
└── Unchanged: TMR

Final Mutant: [8, 64, 96, 2] | dropout: 0.5 | residual: false | prot: TMR
```

---

## 🎛️ Adaptive Mutation Control System

### Population Diversity Monitoring

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      POPULATION DIVERSITY ANALYSIS                         │
└─────────────────────────────────────────────────────────────────────────────┘

For each pair of individuals (i,j) in population:
    Distance(i,j) = calculateConfigDistance(config_i, config_j)

Total Distance = Σ Distance(i,j) for all pairs
Average Distance = Total Distance / Number of Pairs

Normalized Diversity = Average Distance / Maximum Possible Distance

Diversity Threshold = 0.3 (configured parameter)

Decision Logic:
├── Diversity > Threshold → Reduce mutation rate (exploit)
├── Diversity < Threshold → Increase mutation rate (explore)
└── Diversity = Threshold → Maintain current rate (balance)
```

### Adaptive Rate Calculation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ADAPTIVE MUTATION RATE                              │
└─────────────────────────────────────────────────────────────────────────────┘

Parameters:
├── base_rate = 0.1        // Default mutation probability
├── diversity_threshold = 0.3  // Diversity target
├── max_rate = 0.5         // Maximum mutation rate
├── min_rate = 0.01        // Minimum mutation rate

Current Population State:
├── current_diversity = 0.25  // Below threshold
├── fitness_variance = 0.15    // Some convergence
├── generation = 7            // Mid-search

Rate Adjustment Formula:

exploration_factor = (diversity_threshold - current_diversity) / diversity_threshold
convergence_factor = fitness_variance / max_fitness_variance
progression_factor = min(generation / 10, 1.0)  // Early vs late search

adaptive_rate = base_rate +
                (max_rate - base_rate) * exploration_factor +
                (max_rate - base_rate) * convergence_factor * progression_factor

Final Rate = clamp(adaptive_rate, min_rate, max_rate)
```

---

## 🔄 Complete Evolutionary Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EVOLUTIONARY ALGORITHM LOOP                          │
└─────────────────────────────────────────────────────────────────────────────┘

Generation = 0
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INITIALIZATION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Generate N random NetworkConfigs                                         │
│ • Evaluate fitness for each configuration                                  │
│ • Calculate population diversity                                            │
│ • Set initial adaptive mutation rate                                       │
└─────────────────────────────────────────────────────────────────────────────┘

while (!termination_condition) {
    Generation++

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          FITNESS EVALUATION                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ • Run Monte Carlo simulation for each individual                      │
    │ • Calculate baseline accuracy (no radiation)                           │
    │ • Calculate radiation accuracy (with simulated faults)                │
    │ • Compute accuracy preservation ratio                                  │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        SELECTION PHASE                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ • Tournament selection (size 3)                                        │
    │ • Select N/2 parent pairs                                              │
    │ • Preserve elitism (keep best individual)                             │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     ADAPTIVE CONTROL                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ • Monitor population diversity                                         │
    │ • Analyze fitness variance                                             │
    │ • Calculate adaptive mutation rate                                     │
    │ • Adjust exploration vs exploitation balance                          │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       GENETIC OPERATIONS                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ • For each parent pair:                                               │
    │   ├── Crossover with probability P_crossover                          │
    │   └── Mutation with probability adaptive_rate                         │
    │ • Generate N new individuals                                           │
    │ • Add elitism (preserve best from previous generation)               │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      CONVERGENCE CHECK                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ • Fitness plateau detected?                                           │
    │ • Diversity below critical threshold?                                 │
    │ • Maximum generations reached?                                        │
    │ • Time/resource budget exceeded?                                      │
    └─────────────────────────────────────────────────────────────────────────┘
}

┌─────────────────────────────────────────────────────────────────────────────┐
│                              RESULTS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Best architecture found                                                 │
│ • Performance statistics                                                  │
│ • Evolutionary history                                                    │
│ • CSV export of all tested configurations                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Algorithm Performance Characteristics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ALGORITHMIC COMPLEXITY                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Operation               │ Complexity    │ Notes                           │
├─────────────────────────┼───────────────┼─────────────────────────────────┤
│ Population Init         │ O(P)         │ P = population size             │
│ Fitness Evaluation      │ O(P × T × M) │ T = Monte Carlo trials          │
│ Selection               │ O(P × log P) │ Tournament selection            │
│ Diversity Calculation   │ O(P² × G)    │ G = genome length               │
│ Crossover               │ O(P × G)     │ Genome recombination            │
│ Mutation                │ O(P × G)     │ Gene perturbation               │
│ Adaptive Control        │ O(1)         │ Constant time rate adjustment   │
└─────────────────────────┼───────────────┼─────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION PARAMETERS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Parameter               │ Typical Value │ Range                          │
├─────────────────────────┼───────────────┼─────────────────────────────────┤
│ Population Size         │ 6-20         │ 4-50                           │
│ Generations             │ 10-20        │ 5-100                          │
│ Tournament Size         │ 3            │ 2-5                            │
│ Crossover Rate          │ 0.8          │ 0.6-0.9                        │
│ Base Mutation Rate      │ 0.1          │ 0.05-0.2                       │
│ Diversity Threshold     │ 0.3          │ 0.1-0.5                        │
│ Monte Carlo Trials      │ 2-5          │ 1-10                           │
└─────────────────────────┼───────────────┼─────────────────────────────────┘
```

---

## 🎯 Key Algorithm Features

### Adaptive Behavior
- **Dynamic Mutation Rates**: Automatically adjusts exploration vs exploitation
- **Diversity Monitoring**: Prevents premature convergence
- **Fitness Landscape Analysis**: Responds to optimization challenges
- **Progressive Convergence**: Balances early exploration with late exploitation

### Radiation-Specific Optimization
- **Accuracy Preservation**: Primary fitness metric for radiation tolerance
- **Protection Level Optimization**: Balances performance vs reliability
- **Architecture Simplification**: Finds minimal effective configurations
- **Environmental Adaptation**: Optimizes for specific radiation environments

### Robustness Features
- **Elitism**: Preserves best solutions across generations
- **Multiple Termination Criteria**: Flexible stopping conditions
- **Statistical Validation**: Monte Carlo evaluation for reliability
- **CSV Export**: Complete results tracking and analysis

This genetic algorithm provides a sophisticated approach to automatically discovering optimal neural network architectures for radiation-tolerant machine learning applications, with adaptive behavior that responds intelligently to the optimization landscape.

---

## 📐 Mathematical Formulation (Updated)

### Genome and Notation

- Let a configuration (individual) be \(c = (\mathbf{L}, d, r, p)\) where:
  - \(\mathbf{L} = (L_0, L_1, \dots, L_{m})\) are layer sizes with fixed endpoints \(L_0 = \text{input}\), \(L_m = \text{output}\).
  - \(d \in [0,1]\) is the dropout rate.
  - \(r \in \{0,1\}\) is the residual flag.
  - \(p \in \mathcal{P}\) is the protection level enum.

### Crossover

- Uniform crossover (default). For matching length parents \(c^{(1)}, c^{(2)}\):
  - For each hidden index \(i \in \{1,\dots,m-1\}\), draw \(b_i \sim \text{Bernoulli}(1/2)\) and set
    \[ L_i^{\text{child}} = b_i\, L_i^{(1)} + (1-b_i)\, L_i^{(2)}. \]
  - For scalar genes \(d, r, p\), sample independently from parents with probability \(1/2\).
  - If parent lengths differ, choose the entire \(\mathbf{L}\) from one parent uniformly.

- Single-point crossover. When lengths match and \(m \ge 2\):
  - Sample \(k \sim \text{Uniform}\{1,\dots,m-2\}\).
  - Fix endpoints and concatenate hidden segments:
    \[ \mathbf{L}^{\text{child}} = (L_0,\ L_1^{(1)},\ \dots,\ L_k^{(1)},\ L_{k+1}^{(2)},\ \dots,\ L_{m-1}^{(2)},\ L_m). \]
  - Scalar genes \(d, r, p\) are inherited uniformly from parents.

- Crossover application. With probability \(p_c\) per offspring, apply crossover; otherwise clone the fitter parent:
  \[ \Pr(\text{crossover}) = p_c. \]

### Mutation and Operator Selection

- Adaptive operator selection via softmax over credit scores \(q_i\):
  \[ p_i = \frac{\exp(q_i - \max_j q_j)}{\sum_j \exp(q_j - \max_j q_j)}. \]

- Credit update (learning rate \(\alpha = 0.1\)) using average improvement \(\bar{\Delta}_i\):
  \[ q_i \leftarrow (1-\alpha)\, q_i + \alpha\, \bar{\Delta}_i. \]

- Success rate smoothing for operator \(i\) with binary success \(s \in \{0,1\}\):
  \[ \hat{s}_i \leftarrow 0.9\,\hat{s}_i + 0.1\, s. \]

- Diversity-preserving mutation biases choices away from the current gene modes (heuristic), e.g., selecting hidden widths farthest in the allowed index set, flipping residuals, cycling \(p\), and pushing \(d\) away from its current value.

### Configuration Diversity

- Let \(\mathcal{C} = \{c^{(1)},\dots,c^{(n)}\}\) be the population and \(d_\text{cfg}(c^{(i)}, c^{(j)})\) the configuration distance used in code.
- The normalized diversity is the mean pairwise distance scaled by a constant bound (code: 4.0):
  \[ D = \frac{2}{n(n-1)} \sum_{1\le i < j \le n} \frac{d_\text{cfg}(c^{(i)}, c^{(j)})}{4.0}. \]
  Thus \(D \in [0,1]\) by construction.

### Random Immigrants Trigger

- With diversity threshold \(\tau\) and fraction \(\rho\), when
  \[ D < 0.5\, \tau, \]
  replace \(K = \lceil \rho\, n \rceil\) worst individuals with new random samples.

### Logged Metrics (per Generation)

- Mean fitness: \(\bar{f} = \frac{1}{n} \sum_i f_i\).
- Unbiased variance: \(\sigma_f^2 = \frac{1}{n-1} \sum_i (f_i - \bar{f})^2\).
- Diversity \(D\) as above.
- Crossover applied count: \(C_\times\), with expectation \(\mathbb{E}[C_\times] \approx p_c \times N_\text{offspring}\).

---

## 📋 Hyperparameter Quick Reference

| **Parameter** | **Symbol** | **Default** | **Range** | **Code Location** |
|---------------|-----------|-------------|-----------|-------------------|
| Population size | $P$ | 10 | 4–50 | ```101:103:/Users/rishabnuguru/space/include/rad_ml/research/auto_arch_search.hpp
SearchResult evolutionarySearch(size_t population_size = 10, size_t generations = 10,
                                double mutation_rate = 0.1, size_t max_epochs = 10,
                                bool use_monte_carlo = false, size_t monte_carlo_trials = 50);
``` |
| Tournament size | $k_t$ | 3 | 2–5 | ```118:127:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/evolutionary.cpp
auto tournament_select = [&](size_t tournament_size) -> const NetworkConfig& {
    std::uniform_int_distribution<size_t> idx_dist(0, population_size - 1);
    size_t best_idx = idx_dist(random_generator_);
    for (size_t t = 1; t < tournament_size; ++t) {
        size_t cand_idx = idx_dist(random_generator_);
        if (fitness[cand_idx] > fitness[best_idx]) best_idx = cand_idx;
    }
    return population[best_idx];
};
``` |
| Crossover rate | $p_c$ | 0.8 | 0.6–0.9 | ```330:331:/Users/rishabnuguru/space/include/rad_ml/research/auto_arch_search.hpp
double crossover_rate_ = 0.8;
``` |
| Base mutation rate | $m_0$ | 0.1 | 0.05–0.2 | ```151:152:/Users/rishabnuguru/space/include/rad_ml/research/auto_arch_search.hpp
void setAdaptiveMutation(bool enable, double base_rate = 0.1, double diversity_threshold = 0.3,
``` |
| Diversity threshold | $\tau$ | 0.3 | 0.1–0.5 | ```151:152:/Users/rishabnuguru/space/include/rad_ml/research/auto_arch_search.hpp
void setAdaptiveMutation(bool enable, double base_rate = 0.1, double diversity_threshold = 0.3,
``` |
| MC trials | $T$ | 50 | 1–100 | ```472:517:/Users/rishabnuguru/space/src/rad_ml/research/architecture_tester.cpp
ArchitectureTestResult ArchitectureTester::testArchitectureMonteCarlo(...)
``` |
| **QD Parameters** |
| Grid resolution | $R$ | 10 | 5–50 | ```79:89:/Users/rishabnuguru/space/include/rad_ml/research/auto_arch/advanced_quality_diversity.hpp
AdvancedQualityDiversityManager() { /* ... */
    const size_t total_cells =
        static_cast<size_t>(std::pow(current_grid_resolution_, BEHAVIORAL_DIMENSIONS));
    behavioral_archive_.resize(total_cells);
}
``` |
| Novelty neighbors | $K$ | 5 | 3–10 | ```66:68:/Users/rishabnuguru/space/include/rad_ml/research/auto_arch/advanced_quality_diversity.hpp
static constexpr size_t K_NEAREST_NEIGHBORS = 5;
``` |
| Novelty weight | $1-\alpha$ | 0.2 | 0.1–0.5 | ```353:356:/Users/rishabnuguru/space/include/rad_ml/research/auto_arch/advanced_quality_diversity.hpp
static double calculateCombinedFitness(const ArchitectureTestResult& result, double novelty)
{
    return 0.8 * result.accuracy_preservation + 0.2 * novelty * 100.0;
}
``` |
| Elite injection % | – | ~20% | 10–30% | ```228:231:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/evolutionary.cpp
auto elites = advanced_qd_manager_->sampleDiverseElites(std::max<size_t>(1, population_size / 5));
``` |

---

## 🔄 Algorithm Pseudocode → Code Mapping

Algorithm 1: QD-Enhanced Evolutionary Search

1:  P ← InitializePopulation(N)
```29:31:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/evolutionary.cpp
for (size_t i = 0; i < population_size; ++i) {
    population.push_back(generateRandomConfig());
}
```
2:  A ← InitializeArchive(R^6)
```114:116:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/evolutionary.cpp
if (advanced_qd_enabled_ && !advanced_qd_manager_) {
    advanced_qd_manager_ = std::make_unique<AdvancedQualityDiversityManager>();
}
```
3:  for g = 1 to G do
4:    for each c ∈ P do
5:      r ← EvaluateMonteCarlo(c, T)
```472:517:/Users/rishabnuguru/space/src/rad_ml/research/architecture_tester.cpp
ArchitectureTestResult ArchitectureTester::testArchitectureMonteCarlo(...)
```
6:      b ← CalculateBehavior(c, r)
```228:252:/Users/rishabnuguru/space/include/rad_ml/research/auto_arch/advanced_quality_diversity.hpp
RadiationAwareBehaviorDescriptor calculateRadiationAwareBehavior(...)
```
7:      η ← CalculateNovelty(b, A)
```255:268:/Users/rishabnuguru/space/include/rad_ml/research/auto_arch/advanced_quality_diversity.hpp
double calculateNoveltyScore(...)
```
8:      UpdateArchive(A, c, r, η)
```91:115:/Users/rishabnuguru/space/include/rad_ml/research/auto_arch/advanced_quality_diversity.hpp
bool addToArchive(const NetworkConfig& config, const ArchitectureTestResult& test_result,
                  size_t generation = 0)
```
9:    end for
10:   P' ← TournamentSelect(P, k=3)
```118:127:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/evolutionary.cpp
auto tournament_select = [&](size_t tournament_size) -> const NetworkConfig& { /* ... */ };
```
11:   P'' ← Crossover(P', p_c)
```139:143:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/evolutionary.cpp
if (cross_dist(random_generator_) < crossover_rate_) {
    child = crossoverConfigs(parent1, parent2);
    ++crossover_applications;
}
```
```52:100:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/variation.cpp
NetworkConfig AutoArchSearch::crossoverConfigs(const NetworkConfig& parent1,
                                               const NetworkConfig& parent2) { /* ... */ }
```
12:   P''' ← Mutate(P'', p_m(g))
```188:195:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/evolutionary.cpp
if (adaptive_mutation_enabled_ && adaptive_controller_) {
    child = adaptive_controller_->adaptiveMutate(child, current_mutation_rate);
} else {
    child = mutateConfig(child, current_mutation_rate);
}
```
```13:21:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/variation.cpp
NetworkConfig AutoArchSearch::mutateConfig(const NetworkConfig& config, double mutation_rate)
```
13:   E ← SampleElites(A, [0.4, 0.3, 0.3])
```147:185:/Users/rishabnuguru/space/include/rad_ml/research/auto_arch/advanced_quality_diversity.hpp
std::vector<NetworkConfig> sampleDiverseElites(size_t sample_size) const
```
14:   P ← ReplaceWorst(P''', E, k)
```227:259:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/evolutionary.cpp
// Replace a tail of the new population with sampled elites and log coverage
```
15: end for
16: return Best(P ∪ GetAllElites(A))
```329:334:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/evolutionary.cpp
return SearchResult(best_config, best_result.baseline_accuracy, best_result.radiation_accuracy,
                    best_result.accuracy_preservation, generations * population_size,
                    best_result.baseline_accuracy_stddev, best_result.radiation_accuracy_stddev,
                    best_result.accuracy_preservation_stddev, best_result.monte_carlo_trials);
```

---

## 🔍 Complexity Analysis: Diversity Calculation

Claim: O(P² × G) where P = population size, G = genome length

Proof (actual code structure):
```216:243:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/auto_arch_search.cpp
// Calculate average distance between all pairs
for (size_t i = 0; i < population.size(); ++i) {
    for (size_t j = i + 1; j < population.size(); ++j) {
        double dist = calculateConfigDistance(population[i], population[j]);
        total_distance += dist;
        pair_count++;
    }
}
// Normalize and clamp
```
Why G appears: `calculateConfigDistance` iterates over hidden-layer positions and scalars.
```246:300:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/auto_arch_search.cpp
double AutoArchSearch::calculateConfigDistance(const NetworkConfig& config1,
                                               const NetworkConfig& config2) const { /* ... */ }
```

For KNN Novelty: O(n) where n = archive size (capped ~1000).
```255:268:/Users/rishabnuguru/space/include/rad_ml/research/auto_arch/advanced_quality_diversity.hpp
double calculateNoveltyScore(const RadiationAwareBehaviorDescriptor& behavior) const { /* ... */ }
```
Novelty archive cap:
```363:368:/Users/rishabnuguru/space/include/rad_ml/research/auto_arch/advanced_quality_diversity.hpp
if (novelty_archive_.size() > 1000) novelty_archive_.erase(novelty_archive_.begin());
```

---

## 🎓 Design Decisions and Rationale

| **Design Choice** | **Value** | **Rationale** | **Alternative Considered** |
|-------------------|-----------|---------------|---------------------------|
| FULL_TMR overhead | 1.0 | 3× hardware redundancy baseline | Empirical profiling per target |
| ADAPTIVE_TMR overhead | 0.7 | Selective redundancy (≈70% of full) | Dynamic per-layer overhead modeling |
| SPACE_OPTIMIZED overhead | 0.3 | Minimal protection with compression | ECC-only protection |
| Novelty weight | 0.2 | Balance quality (80%) vs diversity (20%) | Adaptive weight based on coverage |
| Grid resolution | 10 | 10^6 cells = sufficient granularity | Adaptive refinement in dense regions |
| K nearest neighbors | 5 | Robust local density estimate | K=3 (noisy), K=10 (over-smoothing) |
| Elite injection | ~20% | Diversify without disrupting convergence | 30% (more disruptive) |
| Crossover rate | 0.8 | Standard in GA literature | Mutation-only (slower) |
| Tournament size | 3 | Balanced selection pressure | 2 (weak), 5 (strong pressure) |
| Diversity threshold | 0.3 | Prevents premature convergence empirically | 0.5 (triggers too late) |

Notes on Protection Overhead:
- Relative estimates for search guidance, not absolute hardware measures
- Reflect relative computational/power costs: FULL_TMR > ADAPTIVE > SELECTIVE > SPACE_OPT > CHECKSUM > NONE
- Future work: profile execution time and energy per protection level to calibrate constants

---

## 🏛️ IEEE QRS GA+QD Validation Outputs (Sample)

- Files produced during tests:
  - `results/genetic_algorithm/ieee_qrs_ga_qd_metrics.csv` (per-generation genetics metrics)
  - `auto_search_results.csv` (aggregate configuration results; created by tests/examples in the run directory)

- Typical short-run console indicators (2 gens, pop=6, trials=5):

```text
QD coverage: 0.0002% (occupied 2), elites injected: 1
QD coverage: 0.0006% (occupied 6), elites injected: 1
Best preservation (examples): ~98.4%–99.3% depending on protection level/params
Monte Carlo stddev (preservation): ~0.20%–0.78%
```

- How to reproduce and collect:
  - Build tests and run: `ctest -R RadML_Research_Tests --output-on-failure`
  - Or run example: `./examples/auto_arch_search_example --qd --adv-qd --trials 10 --gens 5 --pop 10`
  - Inspect: `results/genetic_algorithm/*.csv`, `auto_search_results.csv`, and `run_summaries.csv` (from example)

## 🔗 Code Crosswalk (Core GA and Monte Carlo)

### Monte Carlo Fitness Aggregation (Trials)

Math:
\[ f(c) = \frac{1}{T} \sum_{t=1}^T \frac{A_r^{(t)}(c)}{A_b^{(t)}(c)} \times 100 \]

Code (aggregation across trials):
```415:470:/Users/rishabnuguru/space/src/rad_ml/research/architecture_tester.cpp
// Monte Carlo results aggregation
double total_preservation = 0.0;
// ... accumulate baseline, radiation, preservation, etc.
const size_t n = trial_results.size();
double mean_preservation = total_preservation / n;
// stddevs computed; returned in aggregated result with monte_carlo_trials = n
```

Code (running trials and returning aggregated results):
```472:517:/Users/rishabnuguru/space/src/rad_ml/research/architecture_tester.cpp
ArchitectureTestResult ArchitectureTester::testArchitectureMonteCarlo(..., size_t num_trials, ...)
{
    std::vector<ArchitectureTestResult> trial_results;
    for (size_t i = 0; i < num_trials; ++i) {
        auto result = testArchitecture(architecture, dropout_rate, use_residual_connections,
                                       protection_level, epochs, env, i);
        trial_results.push_back(result);
    }
    auto aggregated_result = calculateMonteCarloStatistics(trial_results);
    return aggregated_result;
}
```

Code (per-trial preservation computation):
```196:206:/Users/rishabnuguru/space/src/rad_ml/research/architecture_tester.cpp
// Calculate radiation accuracy
double radiation_acc = baseline_acc * (1.0 - radiation_impact * (1.0 - protection_factor));
radiation_acc += noise(gen) - 1.0;  // Add some randomness
radiation_acc = std::min(baseline_acc, std::max(10.0, radiation_acc));

// Preservation is percentage of accuracy retained under radiation
double preservation = (radiation_acc / baseline_acc) * 100.0;
```

### Configuration Distance and Diversity Normalization

Math:
\[ d_{\text{cfg}}(c_i, c_j) = d_{\text{layers}} + d_{\text{dropout}} + \mathbb{I}[r_i \ne r_j]\cdot 0.5 + \mathbb{I}[p_i \ne p_j]\cdot 0.25 \]
\[ D = \frac{2}{n(n-1)} \sum_{i<j} \frac{d_{\text{cfg}}(c_i, c_j)}{4.0} \in [0,1] \]

Code (distance components):
```246:300:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/auto_arch_search.cpp
double AutoArchSearch::calculateConfigDistance(const NetworkConfig& config1,
                                               const NetworkConfig& config2) const
{
    double distance = 0.0;

    // Architecture distance - compare layer sizes
    const auto& layers1 = config1.layer_sizes;
    const auto& layers2 = config2.layer_sizes;

    // Handle different architecture depths
    size_t max_layers = std::max(layers1.size(), layers2.size());
    size_t min_layers = std::min(layers1.size(), layers2.size());

    // Compare common layers
    if (!width_options_.empty()) {
        double max_width = *std::max_element(width_options_.begin(), width_options_.end());
        for (size_t i = 0; i < min_layers; ++i) {
            if (i < layers1.size() && i < layers2.size() && max_width > 0) {
                // Normalize layer size difference by maximum possible width
                double width_diff =
                    std::abs(static_cast<double>(layers1[i]) - static_cast<double>(layers2[i]));
                double layer_distance = width_diff / max_width;
                distance += layer_distance;
            }
        }
    }

    // Penalize different number of layers
    if (layers1.size() != layers2.size()) {
        distance +=
            std::abs(static_cast<double>(layers1.size()) - static_cast<double>(layers2.size()));
    }

    // Dropout rate distance
    if (!dropout_options_.empty()) {
        double dropout_range = dropout_options_.back() - dropout_options_.front();
        if (dropout_range > 0) {
            double dropout_diff = std::abs(config1.dropout_rate - config2.dropout_rate);
            double dropout_distance = dropout_diff / dropout_range;
            distance += dropout_distance;
        }
    }

    // Residual connections difference
    if (config1.has_residual_connections != config2.has_residual_connections) {
        distance += 0.5;  // Binary difference
    }

    // Protection level difference
    if (config1.protection_level != config2.protection_level) {
        distance += 0.25;  // Categorical difference
    }

    return distance;
}
```

Code (diversity normalization and 4.0 bound):
```228:243:/Users/rishabnuguru/space/src/rad_ml/research/auto_arch/auto_arch_search.cpp
// Average pairwise distance over all pairs
// Maximum possible distance is roughly 4.0 (max differences across all parameters)
double normalized_diversity = avg_distance / max_possible_distance;
return std::max(0.0, std::min(1.0, normalized_diversity));
```

Derivation (why max distance ≈ 4.0):
- Max layer distance: ≈1.0 (normalized by max width across matching positions)
- Max depth penalty: ≤1.0 (difference in layer counts contributes up to O(1))
- Max dropout distance: 1.0 (from 0.0 to 1.0 after normalization)
- Residual flag difference: 0.5
- Protection level difference: 0.25
- Conservative bound: 4.0 to safely cover heterogeneous option sets and ensure clamping to [0,1].

Derivation note (why 4.0): layer width term ≤~1.0 (normalized), depth penalty contributes up to O(1),
dropout term ≤ 1.0, residual 0.5, protection 0.25; using a conservative cap of 4.0 ensures
clamping keeps D in [0,1] even with heterogeneous option sets.
