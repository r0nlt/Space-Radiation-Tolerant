# Selective Hardening Improvements (August 2025)

This document summarizes recent improvements to `include/rad_ml/neural/selective_hardening.hpp` and related components. These changes improve performance, safety, and maintainability with no breaking API changes.

## Highlights

- Performance: O(1) protection lookups
  - `SensitivityAnalysisResult::protection_map` is an `std::unordered_map<std::string, ProtectionLevel>`.
  - `applyProtection` prefers map lookup, with legacy linear-scan fallback.

- Behavior: Fail-safe defaults
  - Unknown/unspecified protection levels now default to `NONE` (return original value), aligning with tests.

- Maintainability: Policy helpers
  - `applyProtection` refactored into helper/policy functions:
    - `protectNone`, `protectChecksumOnly`, `protectChecksumWithRecovery`
    - `protectTmrBasic`, `protectTmrApproximate`, `protectTmrHealthWeighted`, `protectTmrEnhanced`

- Safety: CRC constraints and semantics
  - CRC helpers require trivially copyable types at compile time (`static_assert(std::is_trivially_copyable_v<T>)`).
  - Byte-wise processing uses `std::byte`; helpers marked `noexcept`.

- Strategy coverage
  - `analyzeAndProtect` now explicitly handles `ALL_LAYERS`, `CRITICAL_LAYERS`, `WEIGHT_THRESHOLD`, and `ADAPTIVE` in addition to existing strategies.

- Determinism and ordering
  - `protection_map` uses `unordered_map` for O(1) semantics. If stable iteration order is needed for reporting/telemetry, sort keys at use sites.

## API Impact

- No public API changes; existing call sites continue to work.
- Compile-time enforcement on CRC templates only affects misuse (non-trivially-copyable types).

## Code Pointers

- Header: `include/rad_ml/neural/selective_hardening.hpp`
- CRC Helper: `CRC32Helper` within the header
- Result type: `ProtectionResult<T>`
- Strategies: `HardeningStrategy` and helper methods within `SelectiveHardening`

## Migration Notes

- If any consumer relied on ordered iteration over `protection_map`, sort keys when iterating.
- For consistent reporting, prefer generating a sorted view rather than changing the container type.

## Validation

- Focused tests (checksum failure, recovery, unknown components) pass.
- Full Monte Carlo and verification suites pass with no regressions.
