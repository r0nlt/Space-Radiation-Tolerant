# Space Radiation Framework: Technical Architecture Reference

## Table of Contents
1. [Introduction](#introduction)
2. [Component Architecture](#component-architecture)
3. [Class Hierarchy](#class-hierarchy)
4. [Key Algorithms](#key-algorithms)
5. [Memory Management](#memory-management)
6. [Error Handling](#error-handling)
7. [Threading and Synchronization](#threading-and-synchronization)
8. [Implementation Details](#implementation-details)
9. [Integration Points](#integration-points)
10. [Test Framework](#test-framework)
11. [Performance Considerations](#performance-considerations)

## Introduction

This document provides detailed technical specifications of the Space Radiation Tolerant Machine Learning Framework architecture. It is intended for software engineers who need to understand, maintain, extend, or integrate with the framework.

## Component Architecture

### Complete Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     User Applications                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                         API Layer                                               │
│                                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ C++ Core API    │  │ Python Bindings │  │ Configuration   │  │ Serialization   │            │
│  │                 │  │                 │  │ Parser          │  │ Services        │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                     Functional Layers                                           │
│                                                                                                 │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────────────────────────────┐ │
│  │       Neural Network Layer      │   │            Redundancy Protection Layer              │ │
│  │                                 │   │                                                     │ │
│  │ ┌─────────────┐ ┌─────────────┐ │   │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │ │
│  │ │ NN Base     │ │ Layer       │ │   │ │ Basic TMR   │ │ Enhanced    │ │ Space-      │    │ │
│  │ │ Components  │ │ Protection  │ │   │ │ Core        │ │ Voting      │ │ Enhanced    │    │ │
│  │ └─────────────┘ └─────────────┘ │   │ └─────────────┘ └─────────────┘ └─────────────┘    │ │
│  │                                 │   │                                                     │ │
│  │ ┌─────────────┐ ┌─────────────┐ │   │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │ │
│  │ │ Weight      │ │ Activation  │ │   │ │ IEEE-754    │ │ Reed-       │ │ Error       │    │ │
│  │ │ Protection  │ │ Hardening   │ │   │ │ FP Voting   │ │ Solomon     │ │ Statistics  │    │ │
│  │ └─────────────┘ └─────────────┘ │   │ └─────────────┘ └─────────────┘ └─────────────┘    │ │
│  └─────────────────────────────────┘   └─────────────────────────────────────────────────────┘ │
│                                                                                                 │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────────────────────────────┐ │
│  │    Radiation Simulation Layer   │   │            Physics Modeling Layer                   │ │
│  │                                 │   │                                                     │ │
│  │ ┌─────────────┐ ┌─────────────┐ │   │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │ │
│  │ │ Environment │ │ Mission     │ │   │ │ Quantum     │ │ Particle    │ │ Material    │    │ │
│  │ │ Models      │ │ Profiles    │ │   │ │ Field       │ │ Transport   │ │ Database    │    │ │
│  │ └─────────────┘ └─────────────┘ │   │ └─────────────┘ └─────────────┘ └─────────────┘    │ │
│  │                                 │   │                                                     │ │
│  │ ┌─────────────┐ ┌─────────────┐ │   │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │ │
│  │ │ Fault       │ │ Rate        │ │   │ │ Electron    │ │ Proton      │ │ Heavy Ion   │    │ │
│  │ │ Injection   │ │ Scheduling  │ │   │ │ Models      │ │ Models      │ │ Models      │    │ │
│  │ └─────────────┘ └─────────────┘ │   │ └─────────────┘ └─────────────┘ └─────────────┘    │ │
│  └─────────────────────────────────┘   └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                        Core Services                                            │
│                                                                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │ Memory      │ │ Error       │ │ Runtime     │ │ Logging     │ │ Power       │ │ Hardware  │ │
│  │ Management  │ │ Handling    │ │ Services    │ │ Services    │ │ Management  │ │ Introspect│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
│                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                             Platform Abstraction Layer                                      ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Matrix

| Component              | Depends On                                 | Provides Services To                       |
|------------------------|--------------------------------------------|-------------------------------------------|
| API Layer              | Core Services, Functional Layers           | User Applications                         |
| Neural Network Layer   | Core Services, Redundancy Protection       | API Layer                                 |
| Redundancy Protection  | Core Services, Memory Management           | Neural Network Layer, API Layer           |
| Radiation Simulation   | Core Services, Physics Models              | Neural Network Layer, Testing Framework    |
| Physics Modeling       | Core Services                              | Radiation Simulation                      |
| Memory Management      | Platform Abstraction                       | All Components                            |
| Error Handling         | Logging Services                           | All Components                            |
| Runtime Services       | Memory Management, Platform Abstraction    | All Components                            |

## Class Hierarchy

### Redundancy Protection Layer Classes

```
┌────────────────────────┐
│ TMRBase<T>             │
│ ──────────────────     │
│ +get(): T              │
│ +set(value: T): void   │
│ +repair(): bool        │
└──────────┬─────────────┘
           │
           ├──────────────────────┬──────────────────────────┬───────────────────────────┐
           │                      │                          │                           │
┌──────────▼─────────────┐ ┌─────▼──────────────────┐ ┌─────▼─────────────────┐  ┌──────▼─────────────────┐
│ BasicTMR<T>            │ │ EnhancedTMR<T>         │ │ SpaceEnhancedTMR<T>   │  │ AdaptiveTMR<T>         │
│ ──────────────────     │ │ ──────────────────     │ │ ──────────────────    │  │ ──────────────────     │
│ -values[3]: T          │ │ -values[3]: T          │ │ -values[3]: T         │  │ -values[3]: T          │
│ -Majority voting       │ │ -checksums[3]: uint32_t │ │ -checksums[3]: uint32 │  │ -checksums[3]: uint32_t│
│                        │ │ -error_stats: Stats     │ │ -error_stats: Stats   │  │ -env_monitor: Monitor  │
└────────────────────────┘ └──────────┬─────────────┘ └─────────┬─────────────┘  └──────────────────────────┘
                                      │                         │
                                      │                         │
                           ┌──────────▼─────────────┐ ┌─────────▼─────────────────┐
                           │ FloatTMR               │ │ IEEE754TMR                │
                           │ ──────────────────     │ │ ──────────────────        │
                           │ +bitLevelVoteFloat()   │ │ +handleSpecialValues()    │
                           └────────────────────────┘ │ +componentWiseVoting()    │
                                                      └───────────────────────────┘
```

### Neural Network Component Classes

```
┌─────────────────────┐       ┌────────────────────────┐
│ NeuralNetworkBase   │◄─────►│ ProtectedLayer         │
└─────────┬───────────┘       └──────────┬─────────────┘
          │                              │
          │                              │
┌─────────▼───────────┐       ┌──────────▼─────────────┐
│ RadHardenedNetwork  │◄─────►│ RadHardenedLayer       │
└─────────────────────┘       └────────────────────────┘
          │                              │
          │                              │
┌─────────▼───────────┐       ┌──────────▼─────────────┐
│ TMRNetwork          │◄─────►│ TMRLayer               │
└─────────────────────┘       └────────────────────────┘
```

### Memory Management Classes

```
┌──────────────────────┐
│ MemoryManager        │
└─────────┬────────────┘
          │
          ├─────────────────────┬─────────────────────┐
          │                     │                     │
┌─────────▼────────────┐ ┌─────▼───────────────┐ ┌───▼───────────────────┐
│ FixedAllocator       │ │ ProtectedAllocator  │ │ AlignedAllocator      │
└──────────────────────┘ └─────────────────────┘ └─────────────────────────┘
```

## Key Algorithms

### IEEE-754 Floating-Point Bit-Level Voting

```cpp
/**
 * IEEE-754 aware bit-level voting for floating-point types
 *
 * This pseudocode handles the special structure of IEEE-754 floating-point numbers
 */
template<typename T>
T IEEE754BitLevelVoting(T a, T b, T c) {
    // Handle special cases (NaN, Infinity)
    if (isSpecialValue(a) || isSpecialValue(b) || isSpecialValue(c)) {
        return handleSpecialValues(a, b, c);
    }

    // Convert to bit patterns
    BitPattern bits_a = convertToBits(a);
    BitPattern bits_b = convertToBits(b);
    BitPattern bits_c = convertToBits(c);

    // Initialize result pattern
    BitPattern result = 0;

    // Vote on sign bit
    result |= voteOnBit(bits_a, bits_b, bits_c, SIGN_BIT_POSITION);

    // Vote on exponent bits
    BitPattern exp_result = 0;
    for (int i = EXPONENT_START; i <= EXPONENT_END; i++) {
        exp_result |= voteOnBit(bits_a, bits_b, bits_c, i);
    }

    // Validate exponent (special cases)
    if (isInvalidExponent(exp_result)) {
        exp_result = findNearestValidExponent(bits_a, bits_b, bits_c);
    }

    result |= exp_result;

    // Vote on mantissa bits
    for (int i = MANTISSA_START; i <= MANTISSA_END; i++) {
        result |= voteOnBit(bits_a, bits_b, bits_c, i);
    }

    // Convert back to floating-point
    T float_result = convertToFloat(result);

    // Final validation
    if (!isValid(float_result)) {
        return findBestFallback(a, b, c);
    }

    return float_result;
}
```

### Fault Injection Algorithm

```cpp
/**
 * Simulates radiation-induced bit flips based on environment model
 */
void injectFaults(void* memory_block, size_t size, Environment env, Duration time) {
    // Calculate expected number of bit flips
    double bit_flip_rate = env.getSEURate(); // SEUs per bit per time
    double total_bits = size * 8;
    double expected_flips = bit_flip_rate * total_bits * time.seconds();

    // Use Poisson distribution to determine actual number of flips
    int num_flips = poissonRandom(expected_flips);

    // For each flip
    for (int i = 0; i < num_flips; i++) {
        // Determine if it's a single bit or multi-bit upset
        bool is_mbu = (randomUniform() < env.getMBUProbability());

        if (is_mbu) {
            // Multi-bit upset
            int first_bit = randomInt(0, total_bits - 1);
            int pattern_type = selectMBUPattern(env);
            flipBitsInPattern(memory_block, first_bit, pattern_type);
        } else {
            // Single bit upset
            int bit_position = randomInt(0, total_bits - 1);
            flipSingleBit(memory_block, bit_position);
        }
    }
}
```

## Memory Management

### Fixed Memory Allocation

The framework uses fixed memory allocators to ensure deterministic behavior in space environments:

```cpp
template<typename T, size_t MaxSize>
class FixedAllocator {
private:
    union Slot {
        T value;
        Slot* next;
    };

    Slot storage_[MaxSize];
    Slot* free_list_;
    std::atomic<size_t> used_count_;

public:
    FixedAllocator() : free_list_(nullptr), used_count_(0) {
        // Initialize free list
        for (size_t i = 0; i < MaxSize - 1; ++i) {
            storage_[i].next = &storage_[i + 1];
        }
        storage_[MaxSize - 1].next = nullptr;
        free_list_ = &storage_[0];
    }

    T* allocate() {
        Slot* slot = free_list_;
        if (!slot) return nullptr; // Out of memory

        free_list_ = slot->next;
        used_count_++;

        return &(slot->value);
    }

    void deallocate(T* ptr) {
        if (!ptr) return;

        Slot* slot = reinterpret_cast<Slot*>(ptr);
        slot->next = free_list_;
        free_list_ = slot;
        used_count_--;
    }

    size_t used() const { return used_count_; }
    size_t available() const { return MaxSize - used_count_; }
};
```

### Protected Containers

The framework includes radiation-hardened containers that apply TMR protection to entire collections:

```cpp
template<typename T, size_t N>
class ProtectedArray {
private:
    SpaceEnhancedTMR<T> data_[N];

public:
    T get(size_t index) {
        if (index >= N) {
            return T{}; // Default value for out-of-bounds
        }

        T result;
        auto status = data_[index].get(result);
        if (status != StatusCode::SUCCESS) {
            // Log and handle error
            error_stats_.addError(status);
        }
        return result;
    }

    void set(size_t index, const T& value) {
        if (index >= N) return;
        data_[index].set(value);
    }

    size_t size() const { return N; }

    void repair() {
        for (size_t i = 0; i < N; ++i) {
            data_[i].repair();
        }
    }
};
```

## Error Handling

The framework implements a comprehensive error handling system based on status codes rather than exceptions, which is more suitable for critical space applications:

```cpp
enum class StatusCode {
    SUCCESS = 0,
    MEMORY_ERROR = 1,
    RADIATION_DETECTION = 2,
    REDUNDANCY_FAILURE = 3,
    UNCORRECTABLE_ERROR = 4,
    VALIDATION_ERROR = 5,
    HARDWARE_ERROR = 6,
    CONFIGURATION_ERROR = 7,
    UNKNOWN_ERROR = 999
};

struct ErrorEvent {
    StatusCode code;
    std::string component;
    ErrorSeverity severity;
    uint64_t timestamp;
    std::string details;
};

class ErrorHandler {
private:
    std::atomic<uint64_t> error_counts_[8]; // One for each StatusCode
    CircularBuffer<ErrorEvent, 128> recent_errors_;
    std::function<void(const ErrorEvent&)> critical_callback_;

public:
    void handleError(StatusCode code, const std::string& component,
                    ErrorSeverity severity, const std::string& details) {
        // Record the error
        ErrorEvent event{code, component, severity, getCurrentTimestamp(), details};

        // Update statistics
        error_counts_[static_cast<int>(code)]++;
        recent_errors_.push(event);

        // For critical errors, call the callback if set
        if (severity == ErrorSeverity::CRITICAL && critical_callback_) {
            critical_callback_(event);
        }

        // Log the error
        Logger::log(LogLevel::ERROR,
                   formatString("Error in %s: %s (Code %d)",
                               component.c_str(), details.c_str(),
                               static_cast<int>(code)));
    }

    void setCriticalErrorCallback(std::function<void(const ErrorEvent&)> callback) {
        critical_callback_ = callback;
    }

    // Additional methods for error statistics, history, etc.
};
```

## Threading and Synchronization

The framework uses a controlled threading model suitable for space applications:

```cpp
class SpaceSafeExecutor {
private:
    FixedThreadPool<4> worker_pool_; // Limited to 4 threads maximum
    std::atomic<bool> active_;
    std::atomic<uint32_t> task_count_;

public:
    template<typename Func, typename... Args>
    auto submitTask(Func&& func, Args&&... args) {
        if (!active_) {
            return std::future<std::invoke_result_t<Func, Args...>>();
        }

        task_count_++;
        auto future = worker_pool_.enqueue(std::forward<Func>(func),
                                         std::forward<Args>(args)...);

        return future;
    }

    void waitForCompletion() {
        worker_pool_.waitForCompletion();
    }

    // For critical operations that must not be interrupted
    void pauseAndWait() {
        active_ = false;
        waitForCompletion();
    }

    void resume() {
        active_ = true;
    }
};
```

## Implementation Details

### TMR Implementation Flow

The following sequence diagram illustrates how the TMR system functions when accessing a protected value:

```
┌─────────┐         ┌──────────────┐         ┌───────────────┐         ┌────────────┐
│ Client  │         │ EnhancedTMR  │         │ Error Handler │         │ Statistics │
└────┬────┘         └──────┬───────┘         └───────┬───────┘         └─────┬──────┘
     │  get()              │                         │                       │
     │ ─────────────────► │                         │                       │
     │                     │                         │                       │
     │                     │ verifyChecksums()       │                       │
     │                     │ ◄──────────────────────►│                       │
     │                     │                         │                       │
     │                     │ performMajorityVoting() │                       │
     │                     │ ─────────────────────────────────────────────► │
     │                     │                         │                       │
     │                     │                         │                       │
     │                     │ ◄─────────────────────────────────────────────┐│
     │                     │                         │                     ││
     │                     │ incrementErrorStats()   │                     ││
     │                     │ ──────────────────────────────────────────────┘│
     │                     │                         │                       │
     │ ◄─────────────────┐ │                         │                       │
     │   return result   │ │                         │                       │
     │                   │ │                         │                       │
┌────┴────┐         ┌──────┴───────┐         ┌───────┴───────┐         ┌─────┴──────┐
│ Client  │         │ EnhancedTMR  │         │ Error Handler │         │ Statistics │
└─────────┘         └──────────────┘         └───────────────┘         └────────────┘
```

### IEEE-754 FP Protection Implementation Flow

```
┌─────────┐         ┌──────────────┐         ┌───────────────┐         ┌────────────┐
│ Client  │         │ IEEE754TMR   │         │ BitPatterns   │         │ Validator  │
└────┬────┘         └──────┬───────┘         └───────┬───────┘         └─────┬──────┘
     │  get()              │                         │                       │
     │ ─────────────────► │                         │                       │
     │                     │                         │                       │
     │                     │ isSpecialValue()        │                       │
     │                     │ ─────────────────────► │                       │
     │                     │                         │                       │
     │                     │ ◄─────────────────────┐ │                       │
     │                     │                       │ │                       │
     │                     │                       │ │                       │
     │                     │ convertToBitPattern() │ │                       │
     │                     │ ─────────────────────► │                       │
     │                     │                         │                       │
     │                     │ ◄─────────────────────┐ │                       │
     │                     │                       │ │                       │
     │                     │ voteSignBit()         │ │                       │
     │                     │ voteExponentBits()    │ │                       │
     │                     │ voteMantissaBits()    │ │                       │
     │                     │ ─────────────────────► │                       │
     │                     │                         │                       │
     │                     │ ◄─────────────────────┐ │                       │
     │                     │                       │ │                       │
     │                     │ validateResult()      │ │                       │
     │                     │ ───────────────────────────────────────────────►│
     │                     │                         │                       │
     │                     │ ◄──────────────────────────────────────────────┐│
     │                     │                         │                     ││
     │ ◄─────────────────┐ │                         │                     ││
     │  return result    │ │                         │                     ││
     │                   │ │                         │                     ││
┌────┴────┐         ┌──────┴───────┐         ┌───────┴───────┐         ┌─────┴──────┐
│ Client  │         │ IEEE754TMR   │         │ BitPatterns   │         │ Validator  │
└─────────┘         └──────────────┘         └───────────────┘         └────────────┘
```

## Integration Points

### C++ API Integration

```cpp
// Example of integrating the framework into a C++ application
#include <rad_ml/api/rad_ml.hpp>
#include <rad_ml/neural/protected_neural_network.hpp>

int main() {
    // Initialize the framework with configuration
    rad_ml::Config config;
    config.setEnvironment(rad_ml::Environment::MARS_ORBIT);
    config.setProtectionLevel(rad_ml::ProtectionLevel::SPACE_OPTIMIZED);

    rad_ml::RadML framework(config);

    // Create a protected neural network
    auto network = framework.createNetwork<float>({32, 64, 32, 10});

    // Load weights (they will be automatically protected)
    network->loadWeights("model_weights.bin");

    // Run inference with protection
    std::vector<float> input = readSensorData();
    auto result = network->forward(input);

    // Access the protected result
    for (const auto& value : result) {
        std::cout << value << " ";
    }

    return 0;
}
```

### Python Bindings Integration

```python
# Example of using the Python bindings
import rad_ml
from rad_ml.neural import ProtectedNetwork

# Configure the framework
config = rad_ml.Configuration()
config.environment = rad_ml.Environment.MARS_ORBIT
config.protection_level = rad_ml.ProtectionLevel.SPACE_OPTIMIZED

# Initialize the framework
framework = rad_ml.Framework(config)

# Create a protected neural network
network = framework.create_network([32, 64, 32, 10])

# Load weights
network.load_weights("model_weights.bin")

# Run inference with protection
input_data = read_sensor_data()
result = network.forward(input_data)

# Process results
for value in result:
    print(value)
```

## Test Framework

The framework includes a comprehensive test infrastructure:

```cpp
class RadiationTest {
public:
    // Configure the test
    void setEnvironment(Environment env);
    void setDuration(Duration duration);
    void setProtectionLevel(ProtectionLevel level);
    void setSeed(uint64_t seed);

    // Test execution
    TestResults runTest();

    // Monte Carlo simulation
    MonteCarloResults runMonteCarloSimulation(int num_runs);

    // Statistical analysis
    StatisticalResults analyzeResults(const MonteCarloResults& results);
};
```

### Monte Carlo Test Flow

```
┌─────────────────┐     ┌───────────────┐     ┌────────────────────┐     ┌──────────────┐
│ TestFramework   │     │ Fault Injector│     │ Protected System   │     │ Validator    │
└────────┬────────┘     └───────┬───────┘     └─────────┬──────────┘     └──────┬───────┘
         │                      │                       │                       │
         │ configure()          │                       │                       │
         │─────────────────────►│                       │                       │
         │                      │                       │                       │
         │ injectFaults()       │                       │                       │
         │─────────────────────►│                       │                       │
         │                      │ applyFaults()         │                       │
         │                      │──────────────────────►│                       │
         │                      │                       │                       │
         │                      │                       │ detectAndCorrect()    │
         │                      │                       │◄──────────────────────│
         │                      │                       │                       │
         │ validate()           │                       │                       │
         │──────────────────────────────────────────────────────────────────────►
         │                      │                       │                       │
         │◄──────────────────────────────────────────────────────────────────────
         │                      │                       │                       │
         │ collectStats()       │                       │                       │
         │────────────────────────────────────────────────────────────────────────────►
         │                      │                       │                       │
         │◄────────────────────────────────────────────────────────────────────────────
         │                      │                       │                       │
┌────────┴────────┐     ┌───────┴───────┐     ┌─────────┴──────────┐     ┌──────┴───────┐
│ TestFramework   │     │ Fault Injector│     │ Protected System   │     │ Validator    │
└─────────────────┘     └───────────────┘     └────────────────────┘     └──────────────┘
```

## Performance Considerations

### Memory Footprint Analysis

```
┌───────────────────────────────────────────────────────────────────────────┐
│                     Memory Footprint by Component                         │
├───────────────────────────────┬───────────────────────────────────────────┤
│ Component                     │ Memory Utilization                        │
├───────────────────────────────┼───────────────────────────────────────────┤
│ Basic TMR<int32_t>            │ sizeof(T) * 3 = 12 bytes                  │
├───────────────────────────────┼───────────────────────────────────────────┤
│ Enhanced TMR<int32_t>         │ sizeof(T) * 3 + 12 + 12 = 36 bytes        │
├───────────────────────────────┼───────────────────────────────────────────┤
│ SpaceEnhancedTMR<int32_t>     │ sizeof(T) * 3 + 12 + 24 = 48 bytes        │
├───────────────────────────────┼───────────────────────────────────────────┤
│ SpaceEnhancedTMR<float>       │ sizeof(T) * 3 + 12 + 24 = 48 bytes        │
├───────────────────────────────┼───────────────────────────────────────────┤
│ Protected Neural Network      │ Base × 3.2 (average overhead)             │
├───────────────────────────────┼───────────────────────────────────────────┤
│ Environment Simulator         │ ~500KB (static environment data)          │
└───────────────────────────────┴───────────────────────────────────────────┘
```

### Compute Performance by Protection Level

```
┌───────────────────────────────────────────────────────────────────────────┐
│                     Compute Performance by Protection                     │
├───────────────────────────────┬───────────────────────────────────────────┤
│ Protection Method             │ Performance Impact                        │
├───────────────────────────────┼───────────────────────────────────────────┤
│ No Protection (Baseline)      │ 1.0x                                      │
├───────────────────────────────┼───────────────────────────────────────────┤
│ Basic TMR                     │ 2.1x slower                               │
├───────────────────────────────┼───────────────────────────────────────────┤
│ Enhanced TMR                  │ 2.3x slower                               │
├───────────────────────────────┼───────────────────────────────────────────┤
│ IEEE-754 FP Protection        │ 2.4x slower                               │
├───────────────────────────────┼───────────────────────────────────────────┤
│ Full Neural Network Protection│ 2.8x slower                               │
└───────────────────────────────┴───────────────────────────────────────────┘
```

## Thread Safety Considerations

The framework implements a thread safety strategy appropriate for spacecraft embedded systems:

1. **Fixed Thread Pool**: Limited to a configurable maximum (typically 2-4)
2. **Thread Affinity**: Threads can be pinned to specific cores
3. **Lock-Free Data Structures**: Used wherever possible to avoid priority inversion
4. **Atomic Operations**: Leveraged for counters and status fields
5. **Memory Barriers**: Explicit barriers for cross-thread visibility

```cpp
// Example of thread-safe counter implementation
class RadiationHardenedCounter {
private:
    TMR<std::atomic<uint64_t>> counts_;

public:
    void increment() {
        uint64_t current;
        counts_.get(current);
        counts_.set(current + 1);
    }

    uint64_t get() {
        uint64_t result;
        counts_.get(result);
        return result;
    }
};
```

## Implementation Guidelines for Developers

When extending or modifying the framework, follow these guidelines:

1. **Deterministic Operations**: Avoid non-deterministic behavior, including:
   - Dynamic memory allocation after initialization
   - Iterators over unordered containers
   - Floating-point operations dependent on compiler flags

2. **Error Handling**: Use status codes rather than exceptions

3. **Resource Management**: All resources must be pre-allocated

4. **Thread Safety**: Assume multi-threaded environment for all components

5. **Testing**: Every component must have radiation simulation tests

6. **Power Awareness**: Include power consumption metrics for critical operations
