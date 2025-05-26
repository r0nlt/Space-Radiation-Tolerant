# Alignment-Based Memory Protection Strategies

## Overview

This module provides **alignment-based and bit-interleaved memory protection** for critical data in space and high-reliability systems. It implements redundancy and error correction to defend against radiation-induced memory corruption.

- **Header:** `aligned_memory.hpp`
- **Namespace:** `rad_ml::core::memory`

---

## Motivation: Why Alignment-Based Memory Protection?

In space and other mission-critical environments, single-event upsets (SEUs) caused by radiation can flip bits in memory, leading to data corruption. Standard memory storage is vulnerable to such events, especially when multiple redundant copies are stored adjacently.

By using alignment and bit interleaving, this module:
- **Physically separates redundant copies** to reduce the chance of multi-copy corruption.
- **Implements triple modular redundancy (TMR)** and voting to correct single-bit errors.
- **Provides scrubbing mechanisms** to repair corrupted data automatically or on demand.

---

## AlignedProtectedMemory

A class that stores three physically separated, aligned copies of a value. Uses voting and scrubbing to ensure data integrity.

### Template Parameters
- `T`: Arithmetic type to protect
- `Alignment`: Memory alignment (default: 64 bytes)

### Key Methods
- `get()`: Returns the voted value, auto-scrubs if needed
- `set(const T&)`: Sets all three copies
- `scrub()`: Scrubs using voting
- `scrub(const T&)`: Scrubs using a known-correct value
- `enableScrubbing(bool)`: Enables/disables auto-scrubbing
- `getRawCopy(size_t)`: Accesses a specific copy (for testing)
- `corruptCopy(size_t, const T&)`: Corrupts a specific copy (for testing)

### Example
```cpp
#include "aligned_memory.hpp"
using namespace rad_ml::core::memory;

AlignedProtectedMemory<int> safe_value(42);
int value = safe_value.get(); // Returns the voted value
safe_value.set(100);         // Sets all three copies to 100
safe_value.scrub();          // Scrubs using voting
```

---

## InterleavedBitMemory

A class that stores three bit-interleaved copies of an integral value. Protects against adjacent bit errors and uses voting for correction.

### Template Parameters
- `T`: Integral type to protect

### Key Methods
- `get()`: Returns the voted value after de-interleaving
- `set(const T&)`: Sets all three interleaved copies
- `scrub()`: Scrubs by rewriting the interleaved data

### Example
```cpp
#include "aligned_memory.hpp"
using namespace rad_ml::core::memory;

InterleavedBitMemory<uint32_t> safe_bits(0xDEADBEEF);
uint32_t value = safe_bits.get(); // Returns the voted value
safe_bits.set(0x12345678);        // Sets all three interleaved copies
safe_bits.scrub();                // Scrubs the interleaved data
```

---

## Usage Analysis

### How to Use
- **Protect critical variables** in space or high-radiation environments.
- **Store system state, configuration, or sensor readings** that must be resilient to bit flips.
- **Test fault tolerance** by corrupting individual copies and verifying correction.
- **Use in embedded and real-time systems** where data integrity is paramount.

### How It Works
- **Redundancy:** Three copies are stored, either physically separated (alignment) or bit-interleaved.
- **Voting:** On read, a voting algorithm determines the most likely correct value.
- **Scrubbing:** Detected errors are automatically or manually corrected by overwriting corrupted copies.
- **Physical/Logical Separation:** Alignment and interleaving reduce the risk of a single event corrupting all copies.

---

## Scientific and Engineering Rationale

- **Radiation Hardening:** Mitigates single-event upsets (SEUs) by redundancy and separation.
- **Deterministic Correction:** Voting and scrubbing ensure that errors are detected and corrected predictably.
- **Testability:** Methods for corrupting and accessing raw copies allow for robust testing and validation.

---

## When to Use
- Spacecraft flight software
- High-reliability embedded systems
- Safety-critical applications
- Any context where memory corruption is a risk

---

## Limitations
- Only supports arithmetic (aligned) or integral (interleaved) types.
- Overhead: Uses 3x the memory for redundancy.
- Not a substitute for full ECC or hardware-level protection in all cases.

---

## References
- [Triple Modular Redundancy (TMR)](https://en.wikipedia.org/wiki/Triple_modular_redundancy)
- [Radiation Effects on Electronics](https://nepp.nasa.gov/)
- [NASA JPL Flight Software Coding Standards](https://flightsoftware.jpl.nasa.gov/)
