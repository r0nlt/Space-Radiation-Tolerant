# Memory Protection and Management Mechanisms in Radiation-Tolerant Computing

## Scientific and Technical Overview

The `rad_ml/core/memory` module provides a comprehensive set of memory protection and management mechanisms designed for high-reliability and radiation-tolerant computing environments. These mechanisms address both transient and persistent memory faults, offering layered protection from the hardware level up to high-level data structures. The approaches combine classical error detection/correction with modern, adaptive, and space-optimized techniques.

---

## 1. Memory Scrubbing Utilities — `memory_scrubbing.hpp` & `memory_scrubber.hpp`

**Scientific Rationale:**
Radiation can induce bit flips (Single Event Upsets, SEUs) in memory, even when memory is not actively accessed. Memory scrubbing is a proactive technique that periodically scans and repairs memory regions to prevent the accumulation of undetected errors.

**Technical Implementation:**
- `memory_scrubbing.hpp` provides static utility functions for:
  - CRC32 checksum calculation and verification
  - Backing up, restoring, and comparing memory regions
- `memory_scrubber.hpp` implements a background service:
  - Allows registration of memory regions and user-defined scrubbing functions
  - Periodically (in a dedicated thread) invokes scrubbing to detect and correct errors
  - Supports manual and automatic scrubbing cycles

**Relevance:**
Essential for maintaining data integrity in environments where silent memory corruption is a risk, such as space or high-radiation terrestrial applications.

---

## 2. Alignment-Based and Interleaved Memory Protection — `aligned_memory.hpp`

**Scientific Rationale:**
Spatial locality in memory can cause a single radiation event to corrupt multiple adjacent values. By physically aligning and separating redundant copies, or interleaving bits, the probability of multi-value corruption is reduced.

**Technical Implementation:**
- **AlignedProtectedMemory:**
  - Stores three redundant copies of a value, each aligned and padded to reduce the chance of simultaneous corruption.
  - Uses advanced voting and optional scrubbing for error correction.
- **InterleavedBitMemory:**
  - Stores redundant copies with bit-level interleaving, ensuring that adjacent bit errors do not affect the same logical bit across all copies.
  - Uses voting to reconstruct the correct value.

**Relevance:**
Improves resilience against burst and spatially correlated errors, complementing logical redundancy with physical memory layout strategies.

---

## 3. Fixed-Size Containers for Space Flight — `fixed_containers.hpp`

**Scientific Rationale:**
Dynamic memory allocation is unpredictable and can be a source of failure in safety-critical and real-time systems. Fixed-size containers provide deterministic memory usage and timing, which is crucial for space and embedded applications.

**Technical Implementation:**
- Implements fixed-capacity vector and map containers with static pre-allocation.
- No dynamic memory allocation; all storage is determined at compile time.
- Provides error codes for overflow, out-of-bounds, and invalid operations.

**Relevance:**
Enables safe, predictable data structures for use in mission-critical software where dynamic allocation is undesirable or forbidden.

---

## 4. Protected Value Containers — `protected_value.hpp`

**Scientific Rationale:**
Critical values require explicit error detection, correction, and reporting. Wrapping values in a protected container allows for robust error handling and monadic operations, supporting functional programming paradigms in safety-critical code.

**Technical Implementation:**
- Stores three copies of a value and uses advanced voting for error correction.
- Provides explicit error reporting via `std::variant`, including confidence scores and fault pattern classification.
- Supports monadic `transform` and `bind` operations for safe functional composition.
- Includes scrubbing to repair detected errors.

**Relevance:**
Facilitates robust, composable, and transparent error handling for critical data in radiation-tolerant systems.

---

## 5. Unified Memory Management — `unified_memory.hpp`

**Scientific Rationale:**
A unified approach to memory management enables system-wide tracking, protection, and error correction, integrating multiple protection schemes (canaries, CRC, ECC, TMR) under a single interface.

**Technical Implementation:**
- Tracks all allocations, deallocations, and memory statistics.
- Supports multiple protection levels: canary, CRC, ECC, TMR.
- Provides APIs for allocation, deallocation, protection, verification, and repair.
- Includes smart pointers (`RadiationTolerantPtr`) for automatic memory management with built-in protection.
- Supports memory corruption callbacks and leak detection.

**Relevance:**
Centralizes memory safety, making it easier to enforce and monitor protection policies across large, complex systems.

---

## 6. Static Memory Allocator — `static_allocator.hpp`

**Scientific Rationale:**
Static allocation ensures that all memory usage is known at compile time, eliminating runtime allocation failures and fragmentation. This is vital for real-time and space systems where predictability is paramount.

**Technical Implementation:**
- Preallocates a fixed-size memory pool at compile time.
- Allocates and constructs objects from this pool without dynamic memory.
- No deallocation or reuse in the basic implementation (can be extended for more complex schemes).

**Relevance:**
Guarantees deterministic memory usage and timing, supporting the strict requirements of space, avionics, and other high-reliability domains.

---

## Scientific Impact and Integration

The memory mechanisms in this module are designed to be layered and composable, providing protection from the lowest hardware level up to high-level data structures. By combining proactive scrubbing, physical alignment, logical redundancy, and unified management, the framework delivers robust, adaptive memory safety for radiation-tolerant computing.

---

**References:**
- Ziegler, J. F., & Lanford, W. A. (1979). Effect of cosmic rays on computer memories. Science, 206(4420), 776-788.
- NASA Goddard Space Flight Center. (2016). Radiation Effects and Analysis Home Page. https://radhome.gsfc.nasa.gov/
- ESA Space Engineering: ECSS-Q-ST-60-02C: Radiation Hardness Assurance.
