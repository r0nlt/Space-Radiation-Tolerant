## Selective Hardening polish follow-ups

Scope: `include/rad_ml/neural/selective_hardening.hpp`, CRC helpers, error handling ergonomics

- [ ] Use `std::is_trivially_copyable_v<T>` instead of `.value` for clarity (C++17)
- [ ] Consider `std::byte` for CRC byte-wise loops to better express intent
- [ ] Mark CRC helpers and protection helper functions `noexcept` where applicable
- [ ] Accept `std::string_view component_id` in `applyProtection` and helpers to avoid copies
- [ ] Evaluate switching `SensitivityAnalysisResult::protection_map` to `std::unordered_map` for O(1) lookups (API change; needs migration plan)
- [ ] Unify `ProtectionResult<T>` with framework-wide `ErrorInfo`/`Result` style (or add a small `expected`-like wrapper)
- [ ] Address `switch (config_.strategy)` warning: handle all enum values or add a default path with explicit comment
- [ ] Tidy `error_handling.hpp` predefined identifier warnings: avoid `__func__` in default parameters; prefer a macro that expands at call sites
- [ ] Optional: optimize CRC with a table-driven implementation if profiling shows it hot

Notes
- Functional fixes already merged: O(1) protection lookup, default handling to `NONE` on unknown, refactored helpers, CRC trivially-copyable constraints.
