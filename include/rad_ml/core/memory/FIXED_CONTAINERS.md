# 🚀 Space Flight Fixed Containers &nbsp; ![Version](https://img.shields.io/badge/version-1.0.0-blue) ![C++](https://img.shields.io/badge/language-C++17-blue)

## Overview

**Space Flight Fixed Containers** are deterministic, statically-allocated replacements for STL containers, designed for mission-critical, radiation-tolerant machine learning and control software in space environments.

- **No dynamic memory allocation:** All storage is statically allocated at compile time.
- **Deterministic behavior:** Predictable timing and memory usage, essential for real-time and safety-critical systems.
- **Radiation tolerance:** Avoids heap fragmentation and undefined states that can be exacerbated by radiation-induced bit flips.
- **Production-ready:** Used in the Space-Radiation-Tolerant ML Framework v1.0.0.

---

## Why Fixed Containers for Space?

| Challenge in Space         | Fixed Containers Solution                |
|---------------------------|------------------------------------------|
| Radiation-induced bit flips| No heap metadata, less fragmentation    |
| Dynamic allocation failure | No `new`/`delete`, no heap exhaustion   |
| Real-time constraints      | O(1) or O(N) with small, known N        |
| Memory limitations         | Compile-time sizing, no over-allocation |
| Deterministic execution    | No hidden allocations, no exceptions    |

**In space, reliability and predictability are paramount.** STL containers rely on dynamic memory, which can fail or behave unpredictably under radiation. Fixed containers guarantee safe, bounded, and deterministic operation.

---

## API Reference

### `FixedVector<T, Capacity>`

| Method                      | Description                                      | Complexity      |
|-----------------------------|--------------------------------------------------|-----------------|
| `push_back(const T&)`       | Add element if space available                   | O(1)            |
| `at(size_t, T&)`            | Get element by index with error code             | O(1)            |
| `operator[](size_t)`        | Array-like access (no bounds check)              | O(1)            |
| `size()`                    | Current number of elements                       | O(1)            |
| `capacity()`                | Maximum capacity                                 | O(1)            |
| `clear()`                   | Remove all elements                              | O(1)            |
| `empty()`                   | Check if container is empty                      | O(1)            |

### `FixedMap<Key, Value, Capacity>`

| Method                      | Description                                      | Complexity      |
|-----------------------------|--------------------------------------------------|-----------------|
| `insert(const Key&, const Value&)` | Insert or update key-value pair           | O(N) (linear, small N) |
| `find(const Key&, Value&)`  | Find value by key with error code                | O(N)            |
| `erase(const Key&)`         | Remove entry by key                              | O(N)            |
| `size()`                    | Current number of elements                       | O(1)            |
| `capacity()`                | Maximum capacity                                 | O(1)            |
| `clear()`                   | Remove all elements                              | O(1)            |

#### Error Codes

- `SUCCESS`
- `FULL`
- `OUT_OF_BOUNDS`
- `NOT_FOUND`
- `INVALID_OPERATION`

---

## Usage Examples

### Telemetry Collection

```cpp
#include "rad_ml/core/memory/fixed_containers.hpp"

rad_ml::core::memory::FixedVector<float, 128> telemetry_buffer;

// Collect sensor data
if (telemetry_buffer.push_back(sensor_reading) == ContainerError::FULL) {
    // Handle buffer full (e.g., transmit or clear)
}
```

### Command Processing

```cpp
rad_ml::core::memory::FixedMap<uint16_t, Command, 32> command_map;

// Insert command
command_map.insert(command_id, command);

// Find and execute
Command cmd;
if (command_map.find(command_id, cmd) == ContainerError::SUCCESS) {
    cmd.execute();
}
```

### Sensor Data Management

```cpp
rad_ml::core::memory::FixedVector<SensorPacket, 16> sensor_packets;
```

### System Status Monitoring

```cpp
rad_ml::core::memory::FixedMap<std::string, float, 8> status_map;
status_map.insert("battery_voltage", 3.7f);
```

---

## Performance Characteristics

| Operation         | FixedVector      | FixedMap (small N) | STL Vector/Map (dynamic) |
|-------------------|------------------|--------------------|--------------------------|
| Allocation        | O(1), static     | O(1), static       | O(1), dynamic (heap)     |
| Insertion         | O(1)             | O(N)               | O(1)/O(log N)            |
| Lookup            | O(1)             | O(N)               | O(1)/O(log N)            |
| Memory Overhead   | Fixed, known     | Fixed, known       | Variable, heap           |
| Exception Safety  | No exceptions    | No exceptions      | May throw                |
| Radiation Tolerance | High           | High               | Lower (heap metadata)    |

---

## Best Practices

- **Size containers for worst-case mission needs** (compile-time).
- **Check error codes** on all operations—never assume success.
- **Avoid operator[] for untrusted indices**; use `at()` for safety.
- **Clear containers before reuse** to avoid stale data.
- **Use for small, critical data sets** (telemetry, commands, status).
- **Integrate with static memory pools** for further safety.

---

## Design Rationale

- **No dynamic allocation:** Avoids heap fragmentation and allocation failure.
- **No exceptions:** All errors are handled via return codes, not exceptions.
- **Deterministic memory layout:** All storage is contiguous and known at compile time.
- **Simple, auditable code:** Easy to review and verify for flight certification.

---

## Space Environment Considerations

- **Radiation:** Static allocation avoids heap corruption from bit flips.
- **Real-time:** Predictable timing for all operations.
- **Limited memory:** Compile-time sizing prevents over-allocation.
- **Integration:** Drop-in replacement for STL containers in most flight code.

---

## Integration Guidelines

- Include the header:
  `#include "rad_ml/core/memory/fixed_containers.hpp"`
- Use in place of `std::vector` or `std::map` for all mission-critical data.
- Size containers based on mission analysis and worst-case needs.
- Test with simulated radiation and memory faults.

---

## STL vs Fixed Containers

| Feature            | STL Containers      | Fixed Containers         |
|--------------------|--------------------|-------------------------|
| Allocation         | Dynamic (heap)     | Static (stack/data)     |
| Exception Safety   | May throw          | No exceptions           |
| Predictability     | Variable           | Deterministic           |
| Radiation Tolerance| Lower              | High                    |
| Real-time Suitability | Lower           | High                    |

---

## Error Handling Patterns

- Always check the return value of `push_back`, `insert`, `find`, and `erase`.
- Never use exceptions for control flow.
- Use `clear()` before reusing containers.

---

## Future Enhancements!

- Iterator support for range-based for loops.
- Compile-time assertions for sizing.
- Optional bounds-checked operator[].
- More container types (queue, stack, ring buffer).
