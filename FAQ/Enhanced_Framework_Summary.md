# RadML Enhanced Framework Summary

## Quick Reference

For complete details, see the [Complete Technical Documentation](../docs/RadML_Complete_Technical.pdf).

---

## Core Architecture

RadML implements a **10-layer defense-in-depth** architecture:

| Layer | Component | Key File |
|-------|-----------|----------|
| 1 | Physical Memory Placement | `memory/radiation_mapped_allocator.hpp` |
| 2 | Memory Scrubbing | `core/memory/memory_scrubber.hpp` |
| 3 | ECC (Hamming/Reed-Solomon) | `neural/galois_field.hpp` |
| 4-6 | TMR Variants | `tmr/*.hpp` |
| 7 | Hybrid Redundancy | `tmr/hybrid_redundancy.hpp` |
| 8 | Checkpointing | `core/recovery/checkpoint_manager.hpp` |
| 9 | Error Tracking | `core/runtime/error_tracker.hpp` |
| 10 | Power-Aware Protection | `power/power_aware_protection.hpp` |

---

## Key Physics Models

### Empirical (Production Use)
- **Weibull Cross-Section**: Heavy ion SEU prediction
- **Bendel Proton Model**: Proton-induced upsets
- **NASA AP-8/AP-9, AE-8/AE-9**: Trapped particle flux

### Quantum Enhancements
- **Dirac Equation Solver**: Relativistic electron behavior
- **Bethe-Salpeter Equation**: Defect clustering / MBU prediction
- **Green's Function Propagator**: Charge collection dynamics

See: `physics/advanced_quantum_models.hpp`, `physics/quantum_enhanced_radiation.hpp`

---

## Error Correction Stack

```
Data Input
    ↓
Hamming(7,4) — Single-bit correction
    ↓
Reed-Solomon — Multi-symbol correction (Berlekamp-Massey decoder)
    ↓
TMR Voting — Redundancy-based correction
    ↓
Protected Output
```

See: [Galois Field Deep Dive](./Galois_Field_Algorithm_Deep_Dive.md)

---

## Detailed Documentation

| Topic | FAQ Document |
|-------|--------------|
| Quantum Physics | [Enhanced Physics Radiation Simulator](./Enhanced_Physics_Radiation_Simulator.md) |
| ECC Algorithms | [Galois Field Deep Dive](./Galois_Field_Algorithm_Deep_Dive.md) |
| Resource Allocation | [Resource Allocation Deep Dive](./Resource_Allocation_Algorithm_Deep_Dive.md) |
| Enabling Protection | [Radiation Tolerance Enablement Guide](./RADIATION_TOLERANCE_ENABLEMENT_GUIDE.md) |
| Application Layer | [Application Layer Overview](./Application_Layer_Technical_Overview.md) |
| Field Theory | [Classical Physics / Field Theory](./CLASSICAL-PHYSICS/FIELD_THEORY.md) |

---

## Complete Reference

 **[RadML Complete Technical Documentation](../docs/RadML_Complete_Technical.pdf)** — 65+ page comprehensive manual with mathematical foundations, code cross-references, and validation methodology.
