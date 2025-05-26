# How It Works: rad_ml Framework

---

## Overview

The `rad_ml` framework is a virtual testbed for simulating and protecting digital systems (like memory, neural networks, or embedded devices) against space radiation effects. It lets you model real mission environments, inject realistic errors, and test customizable protection algorithms—all in software.

---

## Workflow

### 1. Installation and Compilation
- Clone the repository and install dependencies.
- Build the project (e.g., with CMake or g++).
- This produces binaries and libraries for running simulations and protection routines.

### 2. Running and Testing
- Launch an example or test (e.g., a mission simulation or memory protection demo).
- The software sets up:
  - The digital system to be protected (memory, neural net, etc.)
  - The radiation environment (LEO, SAA, Mars, custom)
  - Mission phases and durations

### 3. Error Injection
- The framework **injects errors** (bit flips, SEUs) into the system based on:
  - The environment's particle flux and cross-section
  - Mission duration and system size
- Error injection is stochastic (random), so each run can produce different error patterns.

### 4. Protection Algorithms
- You can **customize the protection algorithm**:
  - TMR (Triple Modular Redundancy)
  - Reed-Solomon ECC
  - Adaptive protection
  - Or your own custom strategy
- These algorithms **detect and correct errors** in memory or computation, simulating real-world protection.

### 5. Validation and Output
- The framework tracks:
  - Number of errors injected, detected, and corrected
  - Protection effectiveness and system reliability
- Outputs include:
  - Console logs
  - CSV/HTML reports
  - Statistics for comparison across runs

---

## In Short
> When you compile and run the framework, you're simulating a digital system in a radiation environment. The framework injects errors into memory or data, then uses a customizable protection algorithm to detect and correct those errors. You can test different strategies, environments, and mission profiles to see how your system would survive in space or other harsh conditions.

---

## See Also
- [FAQ](FAQ.md)
- [README](../README.md)
- [Example Simulations](../examples/)
