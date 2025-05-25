# Research Module — `rad_ml/research/`

---

## Overview

The `rad_ml/research/` module is the **scientific and experimental core** of the `rad_ml` framework. It provides advanced tools for:

- **Systematic evaluation and optimization** of neural network architectures under radiation.
- **Development and validation** of novel training methodologies.
- **Rigorous, statistically significant testing** using Monte Carlo and other research-grade methods.
- **Academic research and publication-quality experiments** for advancing radiation-tolerant AI.

> This module bridges the gap between practical engineering and scientific inquiry, enabling the design, optimization, and validation of neural networks for extreme environments with reproducible rigor.

---

## File-by-File Analysis

### [`architecture_tester.hpp`](architecture_tester.hpp)
- **Purpose:** Systematic, automated testing of neural network architectures under radiation.
- **Capabilities:**
  - Batch testing of architectures (varying width, dropout, residuals, protection)
  - **Monte Carlo validation**: repeated trials for statistical significance
  - **Performance metrics**: baseline/radiation accuracy, preservation, execution time, error statistics
  - Aggregates results, computes standard deviations, supports CSV export and visualization
- **Scientific Value:** Enables reproducible, statistically robust comparison of architectures and protection strategies.

---

### [`auto_arch_search.hpp`](auto_arch_search.hpp)
- **Purpose:** Automated search for optimal neural network architectures in radiation environments.
- **Capabilities:**
  - Implements **grid search**, **random search**, and **evolutionary (genetic) search** algorithms
  - Supports constraints (fixed input/output, number of layers), residual connections, and protection levels
  - Integrates with `ArchitectureTester` for evaluation and statistical validation
  - Tracks and exports all tested configurations and results
- **Scientific Value:** Provides a research-grade platform for architecture optimization, supporting both exhaustive and heuristic search strategies.

---

### [`radiation_aware_training.hpp`](radiation_aware_training.hpp)
- **Purpose:** Novel training methodologies that inject radiation effects during training.
- **Capabilities:**
  - Simulates bit flips (random or targeted at critical weights) during training
  - Collects training statistics: bit flips, accuracy drop, recovery rate
  - Supports criticality mapping and saving results for further analysis
  - Template-based, works with any compatible network
- **Scientific Value:** Enables the study and development of inherently radiation-resilient networks and training regimes.

---

### [`residual_network.hpp`](residual_network.hpp)
- **Purpose:** Implements protected residual neural networks (ResNets) with skip connections.
- **Capabilities:**
  - Fully protected ResNet architectures, with customizable skip connections and projection functions
  - Radiation-aware forward pass, training, and evaluation
  - Supports saving/loading state, skip connection management, and per-connection protection
- **Scientific Value:** Provides a platform for advanced network architectures, enabling research into the interplay of topology, protection, and resilience.

---

## Research Capabilities

- **Architecture Optimization:**
  - Grid, random, and evolutionary search for optimal architectures under radiation constraints
- **Statistical Validation:**
  - Monte Carlo testing, standard deviation computation, and aggregation of results for scientific rigor
- **Training Paradigms:**
  - Radiation-aware training with bit injection, criticality targeting, and recovery analysis
- **Advanced Architectures:**
  - Protected ResNets with flexible skip connections and per-connection protection

---

## Key Algorithms

- **Monte Carlo Testing:**
  - Repeated trials of architecture performance under randomized radiation effects, yielding statistically significant results (mean, stddev, error rates)
- **Evolutionary Architecture Search:**
  - Population-based search with mutation and crossover, optimizing architectures for resilience and performance
- **Radiation-Aware Training:**
  - Injects bit flips (random or targeted) during training, measuring the network's ability to recover and maintain accuracy
- **Protected Residual Connections:**
  - Implements skip connections with optional projection and radiation protection, supporting robust ResNet-style architectures

---

## Research Workflow

1. **Systematic Architecture Evaluation:**
   - Use `ArchitectureTester` to benchmark a range of architectures and hyperparameters under radiation, collecting detailed metrics.
2. **Automated Optimization:**
   - Employ `AutoArchSearch` to automatically discover optimal architectures using grid, random, or evolutionary strategies.
3. **Algorithm Development:**
   - Use `RadiationAwareTraining` to develop and validate new training methods that improve inherent resilience.
4. **Scientific Validation:**
   - All tools support Monte Carlo/statistical validation, CSV export, and visualization, enabling reproducible, publication-quality research.

---

## Technical Specifications

- **Performance:**
  - Supports batch and parallel testing for high-throughput research
  - Efficient data structures for storing and aggregating results
- **Configuration:**
  - Flexible search spaces (width, depth, dropout, residuals, protection)
  - Customizable training parameters, bit flip probabilities, and environments
- **Integration:**
  - Works seamlessly with the rest of the rad_ml framework (protected networks, simulation, etc.)
  - Results can be exported for further analysis or publication

---

## Summary: Scientific Rigor and Research Enablement

> The `rad_ml/research/` module is designed for academic and industrial researchers who need to:
> - Systematically evaluate and optimize neural networks for radiation tolerance
> - Develop and validate new training and protection strategies
> - Produce statistically significant, reproducible results suitable for publication
> - Advance the state of the art in radiation-tolerant AI through rigorous experimentation and analysis

This module transforms the rad_ml framework from an engineering toolkit into a full-featured research platform for the next generation of robust, space-ready neural networks.
