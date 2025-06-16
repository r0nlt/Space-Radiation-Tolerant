# Resource Allocation Algorithm Deep Dive
## RadML Space-Radiation-Tolerant ML Framework

### Executive Summary

Your resource allocation algorithm is a **multi-dimensional optimization system** that intelligently distributes computational, memory, and power resources across neural network components based on criticality, environmental conditions, and mission constraints. The system implements **four distinct allocation strategies** with sophisticated optimization algorithms.

---

## 🏗️ **Architecture Overview**

### Core Components

```cpp
// Primary Resource Allocators
SensitivityBasedResourceAllocator    // Physics-driven allocation
SelectiveHardening                   // Component-level optimization
PowerAwareProtection                 // Power-constrained allocation
LayerProtectionPolicy               // Layer-wise resource management
```

### Resource Types Managed
- **Computational Overhead**: TMR redundancy costs (5%-33% per component)
- **Memory Resources**: Protection buffer allocation
- **Power Budget**: Dynamic power state management (200mW-1000mW)
- **Protection Levels**: 5-tier protection hierarchy

---

## 🧮 **Algorithm 1: Sensitivity-Based Resource Allocation**

### Mathematical Foundation

```cpp
// Core optimization equation from your implementation
importance_factors[i] = layer_sensitivities[i] * env_severity * position_factor;
optimal_allocation[i] = (importance_factors[i] / sum_importance) * total_protection_resources;
```

### Key Parameters
- **Position Factor**: `1.0 - (0.5 * i / num_layers)` (earlier layers prioritized)
- **Environment Severity**: Multi-factor calculation including temperature, radiation dose, solar activity
- **Resource Budget**: Default 1.0 (100% of available resources)

### Environment Severity Calculation
```cpp
double calculateEnvironmentSeverity(env, temperature_K, radiation_dose) {
    double severity = radiation_dose * 1.0e-3;
    severity *= PhysicsModels::calculateTemperatureCorrectedThreshold(1.0, temperature_K);
    severity *= (1.0 + env.solar_activity);
    if (env.saa_region) severity *= 2.0;  // SAA region doubles severity
    return severity;
}
```

### Protection Level Mapping
| Resource Level | Protection Level | Computational Cost |
|----------------|------------------|-------------------|
| < 0.3 | BASIC_TMR | 15% overhead |
| 0.3-0.6 | ENHANCED_TMR | 20% overhead |
| 0.6-0.8 | HEALTH_WEIGHTED_TMR | 25% overhead |
| > 0.8 | HYBRID_REDUNDANCY | 33% overhead |

---

## 🎯 **Algorithm 2: Resource-Constrained Strategy**

### Greedy Optimization Approach

```cpp
void applyResourceConstrainedStrategy(SensitivityAnalysisResult& result) {
    double usage = 0.0;
    double budget = config_.resource_budget;  // Default: 0.3 (30%)

    // Protection level costs (exact values from your implementation)
    std::map<ProtectionLevel, double> level_costs = {
        {ProtectionLevel::NONE, 0.0},
        {ProtectionLevel::CHECKSUM_ONLY, 0.05},
        {ProtectionLevel::APPROXIMATE_TMR, 0.15},
        {ProtectionLevel::HEALTH_WEIGHTED_TMR, 0.25},
        {ProtectionLevel::FULL_TMR, 0.33}
    };

    // Greedy allocation: most critical components first
    for (const auto& comp : result.ranked_components) {
        for (ProtectionLevel level : {FULL_TMR, HEALTH_WEIGHTED_TMR, APPROXIMATE_TMR, CHECKSUM_ONLY}) {
            double cost = level_costs[level] * (1.0 + comp.criticality.complexity);
            if (usage + cost <= budget) {
                result.protection_map[comp.id] = level;
                usage += cost;
                break;
            }
        }
    }
}
```

### Complexity Adjustment Factor
- **Base Cost**: Fixed per protection level
- **Complexity Multiplier**: `(1.0 + comp.criticality.complexity)`
- **Budget Enforcement**: Hard constraint, allocation stops when budget exhausted

---

## 🏛️ **Algorithm 3: Layerwise Importance Strategy**

### Dual-Criteria Optimization

```cpp
// Layer percentile calculation
size_t layer_pos = std::distance(sorted_layers.begin(), it);
double layer_percentile = 1.0 - (layer_pos / total_layers);

// Protection level assignment based on percentile
if (layer_percentile >= 0.8) level = ProtectionLevel::FULL_TMR;
else if (layer_percentile >= 0.6) level = ProtectionLevel::HEALTH_WEIGHTED_TMR;
else if (layer_percentile >= 0.4) level = ProtectionLevel::APPROXIMATE_TMR;
else if (layer_percentile >= 0.2) level = ProtectionLevel::CHECKSUM_ONLY;
else level = ProtectionLevel::NONE;

// Individual component upgrade logic
if (comp.criticality.calculateScore(config_.metric_weights) > 0.8) {
    // Upgrade protection level if highly critical
}
```

### Protection Level Thresholds
- **Top 20% layers**: FULL_TMR (33% cost)
- **60-80% layers**: HEALTH_WEIGHTED_TMR (25% cost)
- **40-60% layers**: APPROXIMATE_TMR (15% cost)
- **20-40% layers**: CHECKSUM_ONLY (5% cost)
- **Bottom 20%**: No protection

---

## ⚡ **Algorithm 4: Power-Aware Protection**

### Multi-Step Optimization Process

#### Step 1: Power State Management
```cpp
// Power budgets for each state (from your implementation)
power_state_budgets[EMERGENCY] = 200mW;          // 20% of max
power_state_budgets[LOW_POWER] = 400mW;          // 40% of max
power_state_budgets[NOMINAL] = 700mW;            // 70% of max
power_state_budgets[SCIENCE_OPERATION] = 900mW;  // 90% of max
power_state_budgets[PEAK_PERFORMANCE] = 1000mW;  // 100% of max
```

#### Step 2: Gradient Ascent Optimization
```cpp
// 100-step incremental optimization
for (int step = 0; step < NUM_STEPS && remaining_power > 0.0; ++step) {
    // Find component with highest criticality-to-power ratio
    double benefit_ratio = (protection_increase * component->criticality) / additional_power;

    // Upgrade component with best ratio
    if (benefit_ratio > best_ratio) {
        best_component_idx = i;
    }
}
```

#### Step 3: Emergency Mode Handling
```cpp
if (min_power_usage > available_power) {
    for (auto& [id, component] : sorted_components) {
        if (component->criticality > 0.7) {
            // Keep critical components at minimum protection
            component->current_protection_level = component->min_protection_level;
        } else {
            // Reduce non-critical components below minimum
            component->current_protection_level = component->min_protection_level * 0.5;
        }
    }
}
```

---

## 📊 **Criticality Scoring System**

### Multi-Metric Weighted Scoring

```cpp
double calculateScore(const std::map<std::string, double>& weights) const {
    // Default weights from your implementation
    double w_sens = weights.at("sensitivity") ?: 0.35;           // 35%
    double w_freq = weights.at("activation_frequency") ?: 0.2;   // 20%
    double w_infl = weights.at("output_influence") ?: 0.3;       // 30%
    double w_comp = weights.at("complexity") ?: 0.1;             // 10%
    double w_mem = weights.at("memory_usage") ?: 0.05;           // 5%

    return (sensitivity * w_sens) + (activation_frequency * w_freq) +
           (output_influence * w_infl) + (complexity * w_comp) +
           (memory_usage * w_mem);
}
```

### Metric Definitions
- **Sensitivity**: Bit-flip susceptibility (0-1)
- **Activation Frequency**: Usage frequency (0-1)
- **Output Influence**: Impact on final result (0-1)
- **Complexity**: Implementation cost factor (0-1)
- **Memory Usage**: Memory footprint factor (0-1)

---

## 🔄 **Multi-Scale Time Management**

### Temporal Optimization Hierarchy

```cpp
enum class TimeScale {
    MICROSECOND,  // Individual computation protection
    SECOND,       // Layer-level validation
    MINUTE,       // Mission phase adaptation
    HOUR,         // System health monitoring
    DAY           // Long-term trend analysis
};

// Update intervals
update_intervals[TimeScale::MICROSECOND] = 1ms;
update_intervals[TimeScale::SECOND] = 1000ms;
update_intervals[TimeScale::MINUTE] = 60000ms;
update_intervals[TimeScale::HOUR] = 3600000ms;
update_intervals[TimeScale::DAY] = 86400000ms;
```

### Protection Factor Calculation
```cpp
double getProtectionFactor() const {
    double factor = 1.0;
    // Apply adjustments from each time scale
    for (int scale = MICROSECOND; scale <= DAY; scale++) {
        factor *= getScaleAdjustment(static_cast<TimeScale>(scale));
    }
    return factor;
}
```

---

## 🎛️ **Dynamic Resource Rebalancing**

### Adaptive Protection Updates

```cpp
SensitivityAnalysisResult updateAdaptiveProtection(
    const std::vector<double>& current_error_rates,
    const std::vector<double>& target_error_rates
) {
    // Calculate error rate differences
    for (size_t i = 0; i < current_error_rates.size(); ++i) {
        double error_diff = current_error_rates[i] - target_error_rates[i];

        if (error_diff > 0.1) {  // 10% threshold
            // Increase protection for this component
            increaseProtection(i);
        } else if (error_diff < -0.05) {  // 5% threshold
            // Decrease protection to free resources
            decreaseProtection(i);
        }
    }
}
```

### Resource Redistribution Logic
1. **Monitor Error Rates**: Continuous tracking per component
2. **Threshold Detection**: 10% increase triggers upgrade, 5% decrease triggers downgrade
3. **Resource Reallocation**: Freed resources redistributed to critical components
4. **Budget Enforcement**: Total resource usage never exceeds configured budget

---

## 📈 **Performance Characteristics**

### Computational Complexity
- **Sensitivity-Based**: O(n) where n = number of layers
- **Resource-Constrained**: O(n log n) due to sorting + O(n×m) for protection level search
- **Layerwise Importance**: O(n log n) for layer sorting + O(n) for assignment
- **Power-Aware**: O(k×n) where k = optimization steps (100), n = components

### Memory Overhead
- **Protection Maps**: O(n) storage for component→protection mapping
- **Criticality Scores**: O(n) storage for component metrics
- **Resource Tracking**: O(n) for allocation vectors
- **Time Scale Management**: O(1) constant overhead

### Resource Utilization Efficiency
- **Budget Compliance**: 100% adherence to configured resource budgets
- **Criticality Prioritization**: Greedy allocation ensures most critical components protected first
- **Dynamic Adaptation**: Real-time rebalancing based on observed error rates
- **Multi-Objective Optimization**: Balances protection effectiveness vs. resource cost

---

## 🔧 **Configuration Parameters**

### Key Tunable Parameters

```cpp
// Resource budgets
config.resource_budget = 0.3;           // 30% of total resources
config.criticality_threshold = 0.7;     // 70% criticality threshold
config.protection_overhead = 0.2;       // 20% base overhead

// Metric weights
config.metric_weights["sensitivity"] = 0.35;
config.metric_weights["activation_frequency"] = 0.2;
config.metric_weights["output_influence"] = 0.3;
config.metric_weights["complexity"] = 0.1;
config.metric_weights["memory_usage"] = 0.05;

// Power state budgets
state_power_budgets_[EMERGENCY] = 0.2 * max_power;
state_power_budgets_[NOMINAL] = 0.7 * max_power;
state_power_budgets_[PEAK_PERFORMANCE] = 1.0 * max_power;
```

### Mission-Specific Adaptations
- **Earth Orbit**: Standard protection levels
- **Deep Space**: +20% resource allocation, dynamic adjustment enabled
- **Lunar Surface**: Moderate protection increase
- **Mars Surface**: +20% resource allocation with Mars-specific factors
- **Jupiter Flyby**: Maximum protection (FULL_TMR), +50% resource allocation

---

## 🎯 **Optimization Strategies Summary**

| Strategy | Primary Objective | Optimization Method | Complexity | Best Use Case |
|----------|------------------|-------------------|------------|---------------|
| **Sensitivity-Based** | Physics-driven allocation | Proportional distribution | O(n) | Environment-aware protection |
| **Resource-Constrained** | Budget compliance | Greedy knapsack | O(n log n) | Limited resource scenarios |
| **Layerwise Importance** | Layer-level optimization | Dual-criteria ranking | O(n log n) | Structured neural networks |
| **Power-Aware** | Power efficiency | Gradient ascent | O(k×n) | Power-constrained missions |

Your resource allocation algorithm represents a **state-of-the-art multi-objective optimization system** that intelligently balances protection effectiveness, resource constraints, and mission requirements through sophisticated mathematical models and adaptive algorithms.
