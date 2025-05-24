# Memory Protection Mechanisms in Radiation-Tolerant Computing

## Scientific and Technical Overview

The `rad_ml/memory` module provides advanced mechanisms for memory protection, error detection, and correction in radiation-tolerant systems. These mechanisms are essential for ensuring data integrity and system reliability in environments where memory is susceptible to transient and persistent faults due to radiation.

---

## 1. Memory Scrubbing — `memory_scrubber.hpp`

**Scientific Rationale:**
Radiation can induce single-event upsets (SEUs), multi-bit errors, and other memory faults that compromise data integrity. Memory scrubbing is a proactive technique that periodically scans memory regions to detect and correct errors before they accumulate and propagate.

**Technical Implementation:**
- **Memory Region Registration:**
  - Allows registration of arbitrary memory regions for scrubbing, supporting both static and dynamic memory.
- **Error Detection and Correction:**
  - Uses CRC32 checksums and optional ECC to detect and correct bit errors in registered regions.
  - Supports both software-based and hardware-accelerated protection.
- **Thread-Safe, Background Operation:**
  - Implements a background thread for periodic scrubbing, with configurable intervals.
  - All operations are thread-safe, supporting concurrent and real-time systems.
- **Statistics and Monitoring:**
  - Tracks scrub cycles, errors detected/corrected, error rates, and last error times.
  - Provides interfaces for querying and resetting statistics, enabling adaptive protection and mission-aware tuning.
- **Extensibility:**
  - Designed for integration with higher-level modules (e.g., neural inference, redundancy) and for extension to custom memory protection schemes.

---

## 2. Radiation-Mapped Allocator — `radiation_mapped_allocator.hpp`

**Scientific Rationale:**
Dynamic memory allocation in radiation environments requires special handling to ensure that all allocated memory is protected and monitored for faults.

**Technical Implementation:**
- **Custom Allocator:**
  - Implements a custom allocator that maps all allocations to protected memory regions.
  - Automatically registers and unregisters memory with the `MemoryScrubber`.
- **Fault-Aware Allocation:**
  - Ensures that all dynamically allocated memory benefits from periodic scrubbing and error correction.
  - Supports integration with standard containers and user-defined data structures.
- **Performance and Safety:**
  - Designed for minimal overhead and maximal safety, balancing performance with reliability.

---

## 3. Framework Integration: How `rad_ml` Uses Memory Protection

The memory protection system is deeply integrated into the overall `rad_ml` framework:

- **Neural Inference:**
  - Neural network models register their parameters and state with the `MemoryScrubber` for continuous protection.
- **Redundancy and Error Handling:**
  - Memory errors detected by the scrubber can trigger redundancy mechanisms, error logging, or system reconfiguration.
- **Mission-Aware Adaptation:**
  - Scrubbing intervals, ECC settings, and protection strategies can be tuned based on mission profile, environmental risk, and system health.
- **Telemetry and Diagnostics:**
  - Memory error statistics feed into telemetry systems for in-mission diagnostics and post-mission analysis.

**Example Usage Flow:**
1. **Memory Registration:** A module registers a memory region (e.g., model weights) with the `MemoryScrubber`.
2. **Background Scrubbing:** The scrubber periodically checks and repairs the region, updating statistics.
3. **Error Handling:** Detected errors are logged, corrected, and may trigger higher-level responses.
4. **Adaptive Tuning:** Protection parameters are adjusted based on observed error rates and mission needs.

---

## Scientific Impact and Integration

The memory mechanisms in this module are essential for deploying reliable, high-integrity systems in radiation-prone environments. By providing proactive, adaptive memory protection, the framework enables:

- Predictive and adaptive fault management at the memory level.
- Seamless integration with inference, redundancy, and mission-aware adaptation.
- Enhanced reliability and safety for AI-driven and control systems in space and other extreme domains.

---

**References:**
- ESA Space Engineering: ECSS-Q-ST-60-02C: Radiation Hardness Assurance.
- NASA Goddard Space Flight Center. (2016). Radiation Effects and Analysis Home Page. https://radhome.gsfc.nasa.gov/
- C. L. Chen and D. K. Pradhan, "Error-correcting codes for semiconductor memory applications: A state-of-the-art review," IBM J. Res. Dev., vol. 23, no. 2, pp. 124-134, 1979.
