# Fixed-Size Containers for Space-Grade Memory Safety

## Overview

This module provides **fixed-size, statically allocated replacements** for standard C++ containers such as `std::vector` and `std::map`. These containers are designed for use in space flight and other safety-critical systems where dynamic memory allocation is undesirable or forbidden.

- **Header:** `fixed_containers.hpp`
- **Namespace:** `rad_ml::core::memory`

---

## Motivation: Why Fixed-Size Containers?

In space and other mission-critical environments, dynamic memory allocation (heap usage) can lead to:
- **Unpredictable memory usage**
- **Fragmentation**
- **Allocation failures**
- **Non-deterministic behavior**

By using containers with memory allocated at compile time, we ensure:
- **Deterministic memory usage**
- **No runtime allocation or deallocation**
- **No fragmentation**
- **Predictable, certifiable behavior**

---

## Error Codes

All containers use the following error codes for safe operations:

```cpp
enum class ContainerError {
    SUCCESS,
    FULL,
    OUT_OF_BOUNDS,
    NOT_FOUND,
    INVALID_OPERATION
};
```

---

## FixedVector

A statically allocated, vector-like container with a fixed maximum capacity.

### Template Parameters
- `T`: Element type
- `Capacity`: Maximum number of elements (compile-time constant)

### Key Methods
- `push_back(const T&)`: Add element if space is available
- `at(size_t, T&)`: Safe access with bounds checking
- `operator[](size_t)`: Fast access (no bounds checking)
- `size()`: Current number of elements
- `capacity()`: Maximum capacity
- `clear()`: Remove all elements
- `empty()`: Check if container is empty

### Example
```cpp
#include "fixed_containers.hpp"
using namespace rad_ml::core::memory;

FixedVector<int, 100> numbers;
numbers.push_back(42);
numbers.push_back(17);
// numbers.size() == 2
```

---

## FixedMap

A statically allocated, key-value map with fixed capacity and linear search. Optimized for small maps.

### Template Parameters
- `Key`: Key type
- `Value`: Value type
- `Capacity`: Maximum number of key-value pairs

### Key Methods
- `insert(const Key&, const Value&)`: Insert or update a key-value pair
- `find(const Key&, Value&)`: Retrieve value by key
- `erase(const Key&)`: Remove entry by key
- `size()`: Current number of elements
- `capacity()`: Maximum capacity
- `clear()`: Remove all elements

### Example
```cpp
#include "fixed_containers.hpp"
using namespace rad_ml::core::memory;

FixedMap<std::string, int, 50> config;
config.insert("temperature", 25);
config.insert("pressure", 101);
int temp;
if (config.find("temperature", temp) == ContainerError::SUCCESS) {
    // temp == 25
}
```

---

## Scientific and Engineering Rationale

- **No Heap Usage:** All memory is statically allocated, eliminating heap fragmentation and allocation failures.
- **Deterministic Behavior:** Memory usage and timing are predictable, critical for real-time and safety-critical systems.
- **Simplicity:** Linear search and fixed capacity are chosen for simplicity and reliability in small, critical data sets.

---

## When to Use
- Space flight software
- Embedded and real-time systems
- Safety-critical applications
- Any context where dynamic memory is forbidden or dangerous

---

## Limitations
- Capacity is fixed at compile time; exceeding it returns an error.
- Linear search in `FixedMap` is not suitable for large maps.

---

## References
- [C++ Core Guidelines: No Dynamic Allocation in Safety-Critical Code](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-resource)
- [NASA JPL Flight Software Coding Standards](https://flightsoftware.jpl.nasa.gov/)
