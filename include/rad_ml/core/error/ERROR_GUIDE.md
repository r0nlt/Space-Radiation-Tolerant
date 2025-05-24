# Comprehensive Error Handling Guide

## Overview

The `rad_ml/core/error` directory provides a deterministic, status code-based error handling system designed for high-reliability, safety-critical, and space flight software. This approach avoids exceptions in favor of explicit status codes, supporting predictable control flow, robust diagnostics, and mission assurance.

---

## 1. Error Domains and Classification

**Error Domains:**
Errors are categorized by domain, enabling fine-grained filtering, reporting, and response. The main domains are:
- `SYSTEM`: System-level errors (e.g., invalid arguments, system failures)
- `MEMORY`: Memory allocation and protection errors
- `RADIATION`: Radiation-induced events and calibration errors
- `REDUNDANCY`: Redundancy mechanism failures
- `NETWORK`: Neural network and communication errors
- `COMPUTATION`: Arithmetic, overflow, and underflow errors
- `IO`: Input/output errors
- `VALIDATION`: Data and configuration validation errors
- `APPLICATION`: Application-specific errors

---

## 2. Status Code System (`status_code.hpp`)

**Philosophy:**
- **No Exceptions:** All errors are reported via status codes, not exceptions, following best practices for space and embedded systems.
- **Determinism:** Status codes provide deterministic error handling, supporting static analysis and mission assurance.

**StatusCode Class:**
- Encapsulates an error domain, a numeric code, and a human-readable message.
- Provides pre-defined codes for common error types (e.g., `SUCCESS`, `MEMORY_ALLOCATION_FAILURE`, `RADIATION_DETECTION`, `VALIDATION_FAILURE`).
- Methods:
  - `isSuccess()`: Returns true if the status represents success.
  - `isError()`: Returns true if the status represents an error.
  - `getDomain()`, `getCode()`, `getMessage()`: Accessors for error details.
  - Equality/inequality operators for comparison.

**Best Practices:**
- Always check the returned status code after any operation that can fail.
- Use domain-specific codes to enable targeted error handling and reporting.
- Prefer pre-defined codes for common errors; define new codes for application-specific cases as needed.

---

## 3. Result Pattern

**Result<T> Template:**
- Encapsulates either a valid value of type `T` or an error `StatusCode`.
- Methods:
  - `isSuccess()`, `isError()`: Check result state.
  - `getValue(T&)`: Retrieve the value if successful.
  - `getStatus()`: Retrieve the status code.

**Result<void> Specialization:**
- For operations that do not return a value, but may still fail.

**Usage Example:**
```cpp
rad_ml::core::error::Result<int> computeSomething();
auto result = computeSomething();
if (result.isSuccess()) {
    int value;
    result.getValue(value);
    // Use value
} else {
    auto status = result.getStatus();
    // Handle error, log status.getMessage()
}
```

---

## 4. Integration and Best Practices

- **Systematic Error Propagation:**
  - All core modules (memory, redundancy, neural, mission, inference) use status codes and results for error propagation.
- **No Silent Failures:**
  - Always check and handle status codes; never ignore errors.
- **Telemetry and Diagnostics:**
  - Status codes can be logged, reported, and analyzed for mission assurance and post-mission review.
- **Mission Assurance:**
  - Deterministic error handling supports static analysis, formal verification, and certification for space flight.

---

## 5. Extending the Error System

- **Defining New Status Codes:**
  - Use the `StatusCode` constructor with a new domain, code, and message.
  - Example:
    ```cpp
    static const StatusCode CUSTOM_ERROR(ErrorDomain::APPLICATION, 42, "Custom application error");
    ```
- **Custom Domains:**
  - For large projects, define additional domains as needed for modularity and clarity.

---

## 6. References and Further Reading

- NASA Goddard Space Flight Center. (2016). Radiation Effects and Analysis Home Page. https://radhome.gsfc.nasa.gov/
- ECSS-Q-ST-60-02C: Space Product Assurance – Radiation Hardness Assurance.
- A. Avizienis et al., "Basic Concepts and Taxonomy of Dependable and Secure Computing," IEEE TDSC, 2004.
- JPL Flight Software Coding Standards.
