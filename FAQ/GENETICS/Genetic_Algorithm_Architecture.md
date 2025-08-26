# 🧬 Genetic Algorithm Architecture: Adaptive Mutation System

## Overview

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
