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
- Tracks all allocations, deallocations, and memory statistics.
- Supports multiple protection levels: canary, CRC, ECC, TMR.
- Allocator is radiation-aware: memory is mapped to zones (highly shielded, moderately shielded, lightly shielded, unshielded) based on **data criticality** and environmental risk.
- Each allocation is tagged with its criticality and mapped to the most appropriate protection and physical memory region.

**Protection Level Guidance:**

- **Canary:**
  - *Purpose:* Detects buffer overflows and stack corruption by placing known values (canaries) at memory boundaries.
  - *When to Use:*
    - Debugging and development, or for low-criticality, non-persistent data.
    - Lightweight protection for regions where performance is critical and the risk of radiation-induced corruption is low.
  - *Limitations:*
    - Only detects certain overwrites; does not correct errors or protect against bit flips in the main data region.

- **CRC (Cyclic Redundancy Check):**
  - *Purpose:* Detects random bit flips and single-event upsets (SEUs) by maintaining a checksum for each memory region.
  - *When to Use:*
    - For data that is read often but written infrequently (e.g., configuration tables, lookup data).
    - In memory zones with moderate shielding or for data of moderate importance.
    - When error detection is sufficient and correction can be handled at a higher level (e.g., by reloading or recomputing data).
  - *Limitations:*
    - Detects but does not correct errors; not suitable for mission-critical or high-availability data.

- **ECC (Error-Correcting Code):**
  - *Purpose:* Detects and corrects single-bit errors (and sometimes detects double-bit errors) in memory.
  - *When to Use:*
    - For critical data structures, neural network weights, or control parameters in highly or moderately shielded zones.
    - When automatic correction of single-bit errors is required for mission continuity.
    - For data with high read/write frequency where silent corruption cannot be tolerated.
  - *Limitations:*
    - Adds storage and performance overhead; may not correct multi-bit or burst errors.

- **TMR (Triple Modular Redundancy):**
  - *Purpose:* Provides the highest level of protection by triplicating data and using majority voting to mask and correct errors.
  - *When to Use:*
    - For mission-critical, irreplaceable data (e.g., spacecraft state, safety interlocks) in unshielded or high-risk zones.
    - When both detection and correction of multi-bit and burst errors are required.
    - In environments with high SEU rates or for data whose loss would result in mission failure.
  - *Limitations:*
    - Significant resource and performance overhead; use is typically reserved for the most critical system components.

**Allocator Integration:**
- The `RadiationMappedAllocator` automatically places data in the most protected available zone based on its criticality:
  - **MISSION_CRITICAL** data → highly shielded zone + TMR/ECC.
  - **HIGHLY_IMPORTANT** data → moderately shielded zone + ECC/CRC.
  - **MODERATELY_IMPORTANT** data → lightly shielded zone + CRC/canary.
  - **LOW_IMPORTANCE** data → unshielded zone + canary or no protection.
- If a zone is full, the allocator falls back to the next best available zone and logs the placement for diagnostics.

**Summary:**
Protection levels and memory placement are chosen based on a combination of data criticality, environmental risk, and system resource constraints. The allocator and protection mechanisms work together to ensure that the most important data receives the strongest protection, while less critical data is managed efficiently.

---

## 3. Framework Integration: How `

**References:**
- ESA Space Engineering: ECSS-Q-ST-60-02C: Radiation Hardness Assurance.
- NASA Goddard Space Flight Center. (2016). Radiation Effects and Analysis Home Page. https://radhome.gsfc.nasa.gov/
- C. L. Chen and D. K. Pradhan, "Error-correcting codes for semiconductor memory applications: A state-of-the-art review," IBM J. Res. Dev., vol. 23, no. 2, pp. 124-134, 1979.
- R. Baumann, "Radiation-induced soft errors in advanced semiconductor technologies," IEEE Transactions on Device and Materials Reliability, vol. 5, no. 3, pp. 305-316, 2005.
- M. Nicolaidis, "Design for soft error mitigation," IEEE Transactions on Device and Materials Reliability, vol. 5, no. 3, pp. 405-418, 2005.
- S. Rezgui et al., "A survey of fault-tolerance techniques for SRAM-based FPGAs," ACM Computing Surveys, vol. 52, no. 4, Article 77, 2019.
