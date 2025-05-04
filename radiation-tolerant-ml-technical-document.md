# Radiation-Tolerant Machine Learning Framework: Technical Documentation

**Author:** Rishab Nuguru  
**Copyright:** © 2025 Rishab Nuguru  

## 1. Space Radiation Environment Physics

### 1.1 Radiation Sources and Types

The framework models several radiation sources encountered in space environments:

#### 1.1.1 Galactic Cosmic Rays (GCRs)
GCRs consist primarily of high-energy protons (~85%) and heavier nuclei originating from outside our solar system. Their energy spectrum follows a power law with an exponential cutoff:

```
Φ(E) = k·E^(-α)·exp(-E/E₀)
```

Where:
- Φ(E) is the flux as a function of energy
- α is the spectral index (typically 2.5-3.0)
- E₀ is the cutoff energy (~1000 MeV)

The implementation models this as:

```cpp
double gcr_distribution(double energy) {
    double index = 1.0 + 0.3 * env_params_.gcr_modulation;
    double cutoff = 100.0 + 900.0 * env_params_.gcr_modulation;
    if (energy <= 0) return 0.0;
    return std::pow(energy, -index) * std::exp(-energy / cutoff);
}
```

#### 1.1.2 Solar Particle Events (SPEs)
Solar flares and coronal mass ejections generate high flux proton events. The framework models these using the Band function:

```
Φ(E) = A·E^(-α)·exp(-E/E₀)    for E < (α-β)·E₀
     = A'·E^(-β)              for E ≥ (α-β)·E₀
```

Where:
- α is the low-energy spectral index (~3.5)
- β is the high-energy spectral index (~1.3)
- E₀ is the break energy (~30 MeV)

#### 1.1.3 Trapped Radiation
Planetary magnetic fields trap charged particles into radiation belts. For Earth, the framework models the South Atlantic Anomaly (SAA) and Van Allen belts using the AP8/AE8 model approximation:

```cpp
double trapped_proton_flux(double altitude_km, double L_shell) {
    // L-shell parameter calculation for magnetic field
    double B = Earth_magnetic_moment / std::pow(L_shell, 3);
    
    // Flux calculation using AP8 approximation
    return base_flux * std::exp(-(L_shell-1.5)²/0.25) * 
           std::exp(-(altitude_km-500)/scale_height);
}
```

### 1.2 Linear Energy Transfer (LET)

LET quantifies how particles deposit energy in materials. The implementation uses the Bethe-Bloch equation:

```
LET = (K·z²/β²)·[ln(2m₍e₎c²β²/I) - β²]
```

Where:
- z is particle charge
- β is v/c (relativistic velocity)
- I is mean excitation energy of the target material

The code implements this as:

```cpp
double calculateLET(ParticleType type, double energy) const {
    switch(type) {
        case ParticleType::PROTON: {
            // Simplified Bethe-Bloch for protons
            double beta2 = 1.0 - 1.0 / std::pow(1.0 + energy / 938.0, 2);
            return 0.1 * (1.0 / beta2) * std::log(100 * energy);
        }
        case ParticleType::HEAVY_ION: {
            // Heavy ions with higher charge
            double z_effective = 20.0; // Effective charge
            double beta2 = 1.0 - 1.0 / std::pow(1.0 + energy / 10000.0, 2);
            return 10.0 * std::pow(z_effective, 2) / beta2;
        }
        // Other particle implementations...
    }
}
```

### 1.3 Mission-Specific Radiation Environments

The framework accurately models unique radiation environments for different missions:

#### 1.3.1 Low Earth Orbit (LEO)
- Altitude: 400-600 km
- Primary radiation: Trapped protons (SAA), electrons
- Dose rate: ~0.1-1 rad/day
- Shielding effectiveness: High (Earth's magnetosphere provides protection)

```cpp
EnvironmentParams createLEOEnvironment() {
    EnvironmentParams params;
    params.altitude_km = 500.0;
    params.inclination_deg = 45.0;
    params.solar_activity = 3.0;
    params.inside_saa = false;
    params.shielding_thickness_mm = 5.0;
    params.radiation_rate_ = 0.01; // Events per second per element
    
    // Particle ratios appropriate for LEO
    params.particle_flux_ratios[ParticleType::PROTON] = 0.80;
    params.particle_flux_ratios[ParticleType::ELECTRON] = 0.15;
    params.particle_flux_ratios[ParticleType::HEAVY_ION] = 0.02;
    params.particle_flux_ratios[ParticleType::ALPHA] = 0.025;
    params.particle_flux_ratios[ParticleType::NEUTRON] = 0.005;
    
    return params;
}
```

#### 1.3.2 Geostationary Orbit (GEO)
- Altitude: 35,786 km
- Primary radiation: GCRs, solar particles
- Dose rate: ~10 rad/day
- Shielding effectiveness: Medium (outside main Van Allen belts)

#### 1.3.3 Mars Environment
- Primary radiation: GCRs (minimal magnetic protection)
- Secondary neutrons from surface interactions
- Dose rate: ~15-30 rad/day
- Shielding effectiveness: Low (minimal atmosphere/magnetosphere)

```cpp
void initializeForMars() {
    // Radiation environment for Mars
    radiation_env_.total_dose_annual_krad = 25.0;
    radiation_env_.peak_flux_particles_cm2_s = 5e3;
    radiation_env_.sea_likelihood = 0.0;  // N/A for Mars
    radiation_env_.galactic_cosmic_ray_exposure = 0.9;
    
    // Mars has minimal magnetic field protection
    rate *= 0.002; // Base rate for Mars (events per second)
    // Enhance heavy ion component due to GCR dominance
    params.particle_flux_ratios[ParticleType::HEAVY_ION] = 0.07;
    params.particle_flux_ratios[ParticleType::NEUTRON] = 0.02;
}
```

#### 1.3.4 Jupiter Environment
- Primary radiation: Intense electron fluxes (Jovian magnetosphere)
- Highly energetic heavy ions
- Dose rate: ~20-50 krad/day (near Europa)
- Shielding effectiveness: Very low (extreme radiation environment)

```cpp
void initializeForJupiter() {
    radiation_env_.total_dose_annual_krad = 100.0;
    radiation_env_.peak_flux_particles_cm2_s = 1e4;
    radiation_env_.solar_event_sensitivity = 1.0;
    radiation_env_.galactic_cosmic_ray_exposure = 1.0;
    
    rate = 0.01; // Base rate for Jupiter (much higher)
    params.particle_flux_ratios[ParticleType::ELECTRON] = 0.20; // Enhanced electron flux
    params.particle_flux_ratios[ParticleType::HEAVY_ION] = 0.10; // Enhanced heavy ions
}
```

## 2. Radiation Effects on Computing Systems

### 2.1 Single Event Effects (SEEs)

#### 2.1.1 Single Event Upsets (SEUs)
SEUs occur when ionizing radiation deposits sufficient energy to flip a memory bit. The critical charge required to flip a bit is modeled as:

```
Q_crit = C_node × V_dd
```

Where:
- C_node is the node capacitance
- V_dd is the supply voltage

The probability of an upset is modeled using:

```
P(upset) = 1 - exp(-LET/LET_threshold)
```

Where LET_threshold depends on the semiconductor technology.

Implementation:

```cpp
bool calculateSEU(double let, double technology_node_nm) {
    // Calculate threshold LET based on technology node
    double let_threshold = 0.5 * std::pow((technology_node_nm/65.0), 2);
    
    // Calculate upset probability
    double upset_prob = 1.0 - std::exp(-let/let_threshold);
    
    // Stochastic determination
    return (std::uniform_real_distribution<double>(0.0, 1.0)(random_engine_) < upset_prob);
}
```

#### 2.1.2 Multiple Bit Upsets (MBUs)
MBUs occur when a single particle affects multiple bits. The framework models this using a spatial correlation function:

```cpp
case RadiationEffectType::MULTI_BIT_UPSET: {
    // MBUs typically affect 2-3 adjacent bits
    int num_bits = std::min(4, static_cast<int>(event.let_mev_cm2_g / 10.0) + 2);
    
    // Select starting bit
    int start_bit = std::uniform_int_distribution<int>(0, 8-num_bits)(random_engine_);
    
    // Create mask for affected bits
    uint8_t mask = 0;
    for (int i = 0; i < num_bits; ++i) {
        mask |= (1 << (start_bit + i));
    }
    
    // Flip the bits
    *byte_ptr ^= mask;
    
    event.bits_affected = num_bits;
    event.description += " at offset " + std::to_string(event.memory_offset) + 
                        ", " + std::to_string(num_bits) + " adjacent bits";
    break;
}
```

#### 2.1.3 Single Event Latchups (SELs)
SELs activate parasitic structures in CMOS devices. The framework models latchup as a permanent bit stuck at 0 or 1:

```cpp
case RadiationEffectType::SINGLE_EVENT_LATCHUP: {
    // Select bit to get stuck
    int bit = std::uniform_int_distribution<int>(0, 7)(random_engine_);
    
    // Determine stuck value (0 or 1)
    bool stuck_at_1 = std::uniform_real_distribution<double>(0.0, 1.0)(random_engine_) < 0.5;
    
    if (stuck_at_1) {
        *byte_ptr |= (1 << bit);  // Set bit to 1
    } else {
        *byte_ptr &= ~(1 << bit); // Clear bit to 0
    }
    
    event.bits_affected = 1;
    event.description = "Single Event Latchup at offset " + 
                       std::to_string(event.memory_offset) +
                       ", bit " + std::to_string(bit) + 
                       " stuck at " + (stuck_at_1 ? "1" : "0");
    break;
}
```

#### 2.1.4 Single Event Transients (SETs)
SETs are temporary voltage spikes that may propagate through logic. The implementation models this as temporary value corruption:

```cpp
case RadiationEffectType::SINGLE_EVENT_TRANSIENT: {
    uint8_t original = *byte_ptr;
    
    // Corrupt with random value temporarily
    *byte_ptr = static_cast<uint8_t>(std::uniform_int_distribution<int>(0, 255)(random_engine_));
    
    event.bits_affected = 8; // Potentially all bits in byte
    event.description = "Single Event Transient at offset " + 
                       std::to_string(event.memory_offset) +
                       ", byte temporarily corrupted from " + 
                       std::to_string(original) + " to " + 
                       std::to_string(*byte_ptr);
    
    // For a real SET, the value would recover automatically
    event.description += " (transient effect - would recover automatically)";
    break;
}
```

### 2.2 Technology Scaling Effects

The framework accounts for how semiconductor scaling affects radiation sensitivity:

```cpp
double calculateScalingFactor(double technology_node_nm) {
    // Sensitivity increases approximately quadratically with scaling
    return std::pow(65.0/technology_node_nm, 2);
}
```

| Technology Node | Critical Charge (Q_crit) | LET Threshold (MeV-cm²/mg) |
|-----------------|--------------------------|----------------------------|
| 180 nm          | ~40 fC                   | ~15                        |
| 90 nm           | ~10 fC                   | ~5                         |
| 45 nm           | ~2.5 fC                  | ~1.5                       |
| 16 nm           | ~0.5 fC                  | ~0.5                       |

## 3. Triple Modular Redundancy Implementation

### 3.1 Basic TMR

The basic Triple Modular Redundancy system maintains three copies of each value and uses majority voting:

```cpp
T get() const {
    // If all values match, return any copy
    if (copies_[0] == copies_[1] && copies_[1] == copies_[2]) {
        return copies_[0];
    }
    
    // Find the majority value using voting
    if (copies_[0] == copies_[1]) {
        return copies_[0];
    } 
    else if (copies_[0] == copies_[2]) {
        return copies_[0];
    } 
    else if (copies_[1] == copies_[2]) {
        return copies_[1];
    } 
    else {
        // All three values are different - cannot determine correct value
        return copies_[0]; // Default to first copy
    }
}
```

### 3.2 Enhanced TMR with CRC Validation

The Enhanced TMR system adds CRC-32 checksums to each copy for integrity verification:

```cpp
T get() {
    // Verify CRCs before performing operation
    bool checksum_valid[num_copies_] = {
        verifyChecksum(0),
        verifyChecksum(1),
        verifyChecksum(2)
    };
    
    // Count valid copies
    int valid_count = 0;
    for (int i = 0; i < num_copies_; ++i) {
        if (checksum_valid[i]) valid_count++;
    }
    
    // If all checksums valid, use normal TMR voting
    if (valid_count == 3) {
        return performMajorityVoting();
    }
    
    // If only some copies valid, use only those for voting
    if (valid_count > 0) {
        // If only one valid, return it
        if (valid_count == 1) {
            for (int i = 0; i < 3; ++i) {
                if (checksum_valid[i]) {
                    return copies_[i];
                }
            }
        }
        
        // If two valid, compare them
        if (checksum_valid[0] && checksum_valid[1] && copies_[0] == copies_[1]) {
            return copies_[0];
        }
        if (checksum_valid[0] && checksum_valid[2] && copies_[0] == copies_[2]) {
            return copies_[0];
        }
        if (checksum_valid[1] && checksum_valid[2] && copies_[1] == copies_[2]) {
            return copies_[1];
        }
    }
    
    // At this point all checksums are invalid or disagree
    // Resort to normal TMR voting and hope for the best
    return performMajorityVoting();
}
```

The CRC-32 calculation uses the IEEE 802.3 polynomial:

```cpp
uint32_t calculateChecksum(const T& value) const {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(&value);
    uint32_t crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < sizeof(T); ++i) {
        uint8_t index = (crc ^ data[i]) & 0xFF;
        crc = (crc >> 8) ^ crc_table_[index];
    }
    
    return ~crc;
}
```

### 3.3 Health-Weighted TMR

Health-Weighted TMR extends the basic approach with dynamic health scoring and weighted voting. This innovation allows the system to adapt to degrading copies over mission lifetime:

```cpp
T get() const {
    // First verify CRC checksums
    bool checksum_valid[num_copies_] = {
        verifyChecksum(0),
        verifyChecksum(1),
        verifyChecksum(2)
    };
    
    // Count valid copies and update health metrics
    int valid_count = 0;
    for (int i = 0; i < num_copies_; ++i) {
        if (checksum_valid[i]) {
            valid_count++;
        } else {
            // Penalize health score for invalid checksum
            updateHealthScore(i, false);
        }
    }
    
    // If all checksums valid, use weighted voting
    if (valid_count >= 2) {
        return performWeightedVoting(checksum_valid);
    }
    
    // Last resort - use highest health score
    size_t best_idx = 0;
    for (size_t i = 1; i < num_copies_; ++i) {
        if (health_scores_[i] > health_scores_[best_idx]) {
            best_idx = i;
        }
    }
    
    return copies_[best_idx];
}
```

The health score update algorithm uses a sophisticated temporal decay model to prioritize recent reliability:

```cpp
void updateHealthScore(size_t index, bool correct) const {
    const double reward_factor = 0.05;  // Small reward for correct vote
    const double penalty_factor = 0.2;  // Larger penalty for incorrect vote
    
    // Update vote history for trending analysis
    if (vote_history_[index].size() >= history_length_) {
        vote_history_[index].pop_front();
    }
    vote_history_[index].push_back(correct);
    
    // Update health score based on current result
    if (correct) {
        health_scores_[index] = std::min(1.0, health_scores_[index] + reward_factor);
    } else {
        health_scores_[index] = std::max(0.1, health_scores_[index] - penalty_factor);
        error_counters_[index]++;
    }
    
    // Apply additional trend analysis for consecutive errors
    if (vote_history_[index].size() >= 3) {
        size_t recent_error_count = 0;
        for (size_t i = vote_history_[index].size() - 3; i < vote_history_[index].size(); ++i) {
            if (!vote_history_[index][i]) {
                recent_error_count++;
            }
        }
        
        // Apply additional penalty for consistent errors
        if (recent_error_count >= 2) {
            health_scores_[index] = std::max(0.05, health_scores_[index] - 0.1);
        }
    }
}
```

The weighted voting implementation:

```cpp
T performWeightedVoting(const bool checksum_valid[]) const {
    // Special case: if all values are the same, return that value
    if (copies_[0] == copies_[1] && copies_[1] == copies_[2]) {
        // Update all health scores positively
        for (size_t i = 0; i < num_copies_; ++i) {
            updateHealthScore(i, true);
        }
        return copies_[0];
    }
    
    // Map of values and their weighted votes
    std::map<T, double> weighted_votes;
    
    // Calculate total weight of valid copies
    double total_valid_weight = 0.0;
    for (size_t i = 0; i < num_copies_; ++i) {
        if (checksum_valid[i]) {
            total_valid_weight += health_scores_[i];
        }
    }
    
    // Assign weighted votes
    for (size_t i = 0; i < num_copies_; ++i) {
        if (checksum_valid[i]) {
            weighted_votes[copies_[i]] += 
                health_scores_[i] / total_valid_weight;
        }
    }
    
    // Find value with highest weighted vote
    T best_value = copies_[0];
    double best_score = 0.0;
    
    for (const auto& entry : weighted_votes) {
        if (entry.second > best_score) {
            best_value = entry.first;
            best_score = entry.second;
        }
    }
    
    // Update health scores based on voting results
    for (size_t i = 0; i < num_copies_; ++i) {
        if (checksum_valid[i]) {
            updateHealthScore(i, copies_[i] == best_value);
        }
    }
    
    return best_value;
}
```

### 3.4 Approximate TMR

Approximate TMR is an innovative approach that uses different precision levels for each copy to reduce memory and computational overhead:

```cpp
T get() const {
    // Convert approximate representations back to exact for comparison
    T values[3];
    for (size_t i = 0; i < 3; ++i) {
        values[i] = getExactValue(i);
    }
    
    // Majority voting
    if (values[0] == values[1]) {
        return values[0];
    } else if (values[0] == values[2]) {
        return values[0];
    } else if (values[1] == values[2]) {
        return values[1];
    }
    
    // No majority - return the exact copy
    for (size_t i = 0; i < 3; ++i) {
        if (approximation_types_[i] == ApproximationType::EXACT) {
            return values[i];
        }
    }
    
    // Fallback to first copy if no exact copy exists
    return values[0];
}
```

Different approximation strategies are implemented:

```cpp
T applyApproximation(const T& value, ApproximationType approx_type) const {
    switch (approx_type) {
        case ApproximationType::EXACT:
            return value;
            
        case ApproximationType::REDUCED_PRECISION:
            return reducePrecision(value);
            
        case ApproximationType::RANGE_LIMITED:
            return limitRange(value);
            
        case ApproximationType::CUSTOM:
            if (custom_approximation_) {
                return custom_approximation_(value);
            }
            return value;
            
        default:
            return value;
    }
}
```

#### 3.4.1 Reduced Precision Approximation

```cpp
T reducePrecision(const T& value) const {
    if constexpr (std::is_floating_point_v<T>) {
        // For floating point, reduce precision by truncating
        union {
            T f;
            uint64_t i;
        } u;
        
        u.f = value;
        
        // IEEE 754 format manipulation - mask lower bits of mantissa
        constexpr int mantissa_bits = std::is_same_v<T, float> ? 23 : 52;
        constexpr int precision_reduction = std::is_same_v<T, float> ? 10 : 20;
        
        // Create mask that preserves sign, exponent, and upper mantissa bits
        uint64_t mantissa_mask = ~((1ULL << precision_reduction) - 1);
        
        // If double, we need to shift the mask
        if constexpr (!std::is_same_v<T, float>) {
            mantissa_mask <<= (mantissa_bits - precision_reduction);
        }
        
        // Apply mask to reduce precision
        u.i &= mantissa_mask;
        
        return u.f;
    } 
    else if constexpr (std::is_integral_v<T>) {
        // For integers, mask off lower bits
        constexpr int shift = sizeof(T) <= 2 ? 2 : 3;
        return (value >> shift) << shift;
    }
    else {
        // For other types, just return as is
        return value;
    }
}
```

#### 3.4.2 Range-Limited Approximation

```cpp
T limitRange(const T& value) const {
    if constexpr (std::is_floating_point_v<T>) {
        // For floating point, limit to +/- 1e6
        constexpr T limit = static_cast<T>(1e6);
        return std::max(std::min(value, limit), -limit);
    } 
    else if constexpr (std::is_integral_v<T>) {
        // For integers, limit to half the representable range
        constexpr T limit = std::numeric_limits<T>::max() / 2;
        return std::max(std::min(value, limit), -limit);
    }
    else {
        // For other types, return as is
        return value;
    }
}
```

## 4. Memory Protection System

### 4.1 Memory Scrubbing

The memory scrubbing system periodically scans memory regions to detect and correct bit flips:

```cpp
void scrubThreadFunction() {
    while (running_) {
        // Lock the regions list to prevent modification during scrubbing
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (const auto& region : regions_) {
            // Call the region-specific scrub function
            region.scrub_function();
            
            // Track statistics
            stats_.scrub_cycles++;
        }
        
        // Wait for the next scrubbing cycle
        std::this_thread::sleep_for(std::chrono::milliseconds(scrub_interval_ms_));
    }
}
```

For each memory region, the scrubbing process verifies integrity using CRC:

```cpp
bool verifyMemoryRegion(void* ptr, size_t size) {
    // Calculate current CRC
    uint32_t current_crc = crc_calculator_.calculate(ptr, size);
    
    // Compare with stored CRC
    auto it = region_crcs_.find(ptr);
    if (it != region_crcs_.end()) {
        if (current_crc != it->second) {
            // Error detected
            stats_.errors_detected++;
            return false;
        }
    } else {
        // First verification, store the CRC
        region_crcs_[ptr] = current_crc;
    }
    
    return true;
}
```

### 4.2 CRC-32 Implementation

The framework uses the IEEE 802.3 polynomial (0xEDB88320) for CRC-32 calculation:

```cpp
uint32_t calculate(const void* data, size_t size) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < size; ++i) {
        crc ^= bytes[i];
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }
    }
    
    return ~crc;
}
```

### 4.3 Adaptive Scrubbing

The scrubbing interval adjusts based on detected error rates:

```cpp
void updateScrubInterval(double error_rate) {
    // Minimum and maximum intervals
    constexpr unsigned long min_interval_ms = 100;   // 100ms
    constexpr unsigned long max_interval_ms = 60000; // 1 minute
    
    // Base interval from configuration
    unsigned long base_interval = config_.scrubbing_interval_ms;
    
    // Adjust based on error rate
    // Higher error rate = more frequent scrubbing
    double adjustment_factor = std::exp(-10.0 * error_rate);
    
    // Calculate new interval
    unsigned long new_interval = static_cast<unsigned long>(
        base_interval * adjustment_factor);
    
    // Clamp to allowed range
    scrub_interval_ms_ = std::max(min_interval_ms, 
                                 std::min(max_interval_ms, new_interval));
}
```

## 5. Selective Hardening System

The Selective Hardening system optimizes protection by analyzing component criticality and applying appropriate protection levels.

### 5.1 Component Criticality Analysis

```cpp
double calculateScore(const std::map<std::string, double>& weights) const {
    // Default weights
    double w_sens = weights.count("sensitivity") ? weights.at("sensitivity") : 0.35;
    double w_freq = weights.count("activation_frequency") ? weights.at("activation_frequency") : 0.2;
    double w_infl = weights.count("output_influence") ? weights.at("output_influence") : 0.3;
    double w_comp = weights.count("complexity") ? weights.at("complexity") : 0.1;
    double w_mem = weights.count("memory_usage") ? weights.at("memory_usage") : 0.05;
    
    // Normalize weights
    double sum = w_sens + w_freq + w_infl + w_comp + w_mem;
    w_sens /= sum;
    w_freq /= sum;
    w_infl /= sum;
    w_comp /= sum;
    w_mem /= sum;
    
    // Calculate weighted score
    return (sensitivity * w_sens) +
           (activation_frequency * w_freq) +
           (output_influence * w_infl) +
           (complexity * w_comp) +
           (memory_usage * w_mem);
}
```

### 5.2 Protection Level Assignment Strategies

#### 5.2.1 Fixed Threshold Strategy

```cpp
void applyFixedThresholdStrategy(SensitivityAnalysisResult& result) {
    for (const auto& comp : result.ranked_components) {
        double score = comp.criticality.calculateScore(config_.metric_weights);
        
        if (score >= config_.criticality_threshold) {
            result.protection_map[comp.id] = ProtectionLevel::FULL_TMR;
        }
        else if (score >= config_.criticality_threshold * 0.8) {
            result.protection_map[comp.id] = ProtectionLevel::HEALTH_WEIGHTED_TMR;
        }
        else if (score >= config_.criticality_threshold * 0.6) {
            result.protection_map[comp.id] = ProtectionLevel::APPROXIMATE_TMR;
        }
        else if (score >= config_.criticality_threshold * 0.4) {
            result.protection_map[comp.id] = ProtectionLevel::CHECKSUM_ONLY;
        }
        else {
            result.protection_map[comp.id] = ProtectionLevel::NONE;
        }
    }
}

#### 5.2.2 Resource-Constrained Strategy

```cpp
void applyResourceConstrainedStrategy(SensitivityAnalysisResult& result) {
    double usage = 0.0;
    double budget = config_.resource_budget;
    
    // Define protection level costs
    std::map<ProtectionLevel, double> level_costs = {
        {ProtectionLevel::NONE, 0.0},
        {ProtectionLevel::CHECKSUM_ONLY, 0.05},
        {ProtectionLevel::APPROXIMATE_TMR, 0.15},
        {ProtectionLevel::HEALTH_WEIGHTED_TMR, 0.25},
        {ProtectionLevel::FULL_TMR, 0.33}
    };
    
    // Start with no protection
    for (const auto& comp : result.ranked_components) {
        result.protection_map[comp.id] = ProtectionLevel::NONE;
    }
    
    // Allocate protection to most critical components first
    for (const auto& comp : result.ranked_components) {
        // Try to apply highest level of protection possible within budget
        for (ProtectionLevel level : {
            ProtectionLevel::FULL_TMR,
            ProtectionLevel::HEALTH_WEIGHTED_TMR,
            ProtectionLevel::APPROXIMATE_TMR,
            ProtectionLevel::CHECKSUM_ONLY
        }) {
            double cost = level_costs[level] * (1.0 + comp.criticality.complexity);
            
            if (usage + cost <= budget) {
                result.protection_map[comp.id] = level;
                usage += cost;
                break;
            }
        }
        
        // If we've used up the budget, stop
        if (usage >= budget) {
            break;
        }
    }
}
```

#### 5.2.3 Layerwise Importance Strategy

```cpp
void applyLayerwiseImportanceStrategy(SensitivityAnalysisResult& result) {
    // Sort layers by criticality
    std::vector<std::pair<std::string, double>> sorted_layers;
    for (const auto& [layer, score] : result.layer_criticality) {
        sorted_layers.push_back({layer, score});
    }
    
    std::sort(sorted_layers.begin(), sorted_layers.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
    
    // Assign protection levels based on layer criticality
    double total_layers = sorted_layers.size();
    double usage = 0.0;
    double budget = config_.resource_budget;
    
    for (const auto& comp : result.ranked_components) {
        // Find layer position in sorted list
        auto it = std::find_if(sorted_layers.begin(), sorted_layers.end(),
            [&](const auto& layer_pair) {
                return layer_pair.first == comp.layer_name;
            });
        
        if (it == sorted_layers.end()) continue;
        
        size_t layer_pos = std::distance(sorted_layers.begin(), it);
        double layer_percentile = 1.0 - (layer_pos / total_layers);
        
        // Assign protection level based on layer criticality
        ProtectionLevel level;
        double component_cost = 0.0;
        
        if (layer_percentile >= 0.8) {
            level = ProtectionLevel::FULL_TMR;
            component_cost = 0.33;
        }
        else if (layer_percentile >= 0.6) {
            level = ProtectionLevel::HEALTH_WEIGHTED_TMR;
            component_cost = 0.25;
        }
        else if (layer_percentile >= 0.4) {
            level = ProtectionLevel::APPROXIMATE_TMR;
            component_cost = 0.15;
        }
        else if (layer_percentile >= 0.2) {
            level = ProtectionLevel::CHECKSUM_ONLY;
            component_cost = 0.05;
        }
        else {
            level = ProtectionLevel::NONE;
            component_cost = 0.0;
        }
        
        // Check budget
        if (usage + component_cost <= budget) {
            result.protection_map[comp.id] = level;
            usage += component_cost;
        }
        else {
            result.protection_map[comp.id] = ProtectionLevel::NONE;
        }
    }
}
```

### 5.3 Protection Application

The framework applies different protection mechanisms based on the assigned protection level:

```cpp
template <typename T>
auto applyProtection(
    T value, 
    const std::string& component_id, 
    const SensitivityAnalysisResult& analysis_results) {
    
    if (!analysis_results.protection_map.count(component_id)) {
        // No protection specified, return unprotected value
        return value;
    }
    
    ProtectionLevel level = analysis_results.protection_map.at(component_id);
    
    switch (level) {
        case ProtectionLevel::NONE:
            return value;
            
        case ProtectionLevel::CHECKSUM_ONLY: {
            // Simple wrapper with CRC
            struct ChecksumProtected {
                T value;
                uint32_t checksum;
                
                ChecksumProtected(T v) : value(v) {
                    // CRC calculation
                    const uint8_t* data = reinterpret_cast<const uint8_t*>(&value);
                    size_t size = sizeof(T);
                    
                    uint32_t crc = 0xFFFFFFFF;
                    for (size_t i = 0; i < size; i++) {
                        crc ^= data[i];
                        for (size_t j = 0; j < 8; j++) {
                            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
                        }
                    }
                    checksum = ~crc;
                }
                
                bool verify() const {
                    // Recalculate and check
                    const uint8_t* data = reinterpret_cast<const uint8_t*>(&value);
                    size_t size = sizeof(T);
                    
                    uint32_t crc = 0xFFFFFFFF;
                    for (size_t i = 0; i < size; i++) {
                        crc ^= data[i];
                        for (size_t j = 0; j < 8; j++) {
                            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
                        }
                    }
                    crc = ~crc;
                    
                    return crc == checksum;
                }
                
                T getValue() const {
                    return value;
                }
            };
            
            return ChecksumProtected(value);
        }
            
        case ProtectionLevel::APPROXIMATE_TMR: {
            // Use Approximate TMR with reduced precision
            return rad_ml::tmr::ApproximateTMR<T>(
                value, 
                {rad_ml::tmr::ApproximationType::REDUCED_PRECISION,
                 rad_ml::tmr::ApproximationType::REDUCED_PRECISION,
                 rad_ml::tmr::ApproximationType::EXACT}
            );
        }
            
        case ProtectionLevel::HEALTH_WEIGHTED_TMR: {
            // Use Health-Weighted TMR
            return rad_ml::tmr::HealthWeightedTMR<T>(value);
        }
            
        case ProtectionLevel::FULL_TMR: {
            // Use Enhanced TMR with CRC checksums
            return rad_ml::tmr::EnhancedTMR<T>(value);
        }
            
        default:
            return value;
    }
}
```

## 6. Adaptive Protection System

### 6.1 Protection Level Management

The adaptive protection system dynamically adjusts protection levels based on the radiation environment:

```cpp
void adjustProtectionLevel() {
    // Determine appropriate level based on flux
    ProtectionLevel recommended_level;
    
    if (environment_.estimated_flux >= MAXIMUM_THRESHOLD) {
        recommended_level = ProtectionLevel::MAXIMUM;
    } else if (environment_.estimated_flux >= ENHANCED_THRESHOLD) {
        recommended_level = ProtectionLevel::ENHANCED;
    } else if (environment_.estimated_flux >= STANDARD_THRESHOLD) {
        recommended_level = ProtectionLevel::STANDARD;
    } else {
        recommended_level = ProtectionLevel::MINIMAL;
    }
    
    // Only change level if it differs from current
    if (recommended_level != current_level_) {
        current_level_ = recommended_level;
        // Notify subscribers about level change
        notifyLevelChange();
    }
}
```

### 6.2 Environment Assessment

The system maintains an assessment of the radiation environment based on detected errors:

```cpp
void updateEnvironment(uint32_t new_bit_flips, uint32_t new_computation_errors) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - environment_.last_assessment).count();
    
    if (elapsed > 0) {
        // Update flux estimate
        double error_rate = (new_bit_flips + new_computation_errors) / 
                            static_cast<double>(elapsed);
        
        // Exponential moving average for smooth transitions
        constexpr double alpha = 0.3; // Smoothing factor
        environment_.estimated_flux = 
            alpha * error_rate + (1.0 - alpha) * environment_.estimated_flux;
        
        // Update counters
        environment_.bit_flips_detected = new_bit_flips;
        environment_.computation_errors = new_computation_errors;
        environment_.last_assessment = now;
        
        // Adjust protection level based on new assessment
        adjustProtectionLevel();
    }
}
```

### 6.3 Protection Configuration

Each protection level has specific configuration parameters:

```cpp
// Static configuration for protection levels
static constexpr ProtectionConfig protection_configs_[4] = {
    // MINIMAL
    {1, 5000,  false, false},  // {redundancy_level, scrubbing_interval_ms, temporal_redundancy, checkpoint_recovery}
    
    // STANDARD
    {2, 1000,  false, true},
    
    // ENHANCED
    {3, 500,   true,  true},
    
    // MAXIMUM
    {3, 100,   true,  true}
};
```

## 7. Physics-Based Radiation Simulation

### 7.1 Space Environment Simulation

The framework implements a sophisticated physics-based simulation of different space radiation environments:

```cpp
// Create extreme radiation environment
testing::RadiationSimulator::EnvironmentParams createExtremeEnvironment() {
    testing::RadiationSimulator::EnvironmentParams params;
    
    // Base on Jupiter but make it much worse
    params = testing::RadiationSimulator::getMissionEnvironment("JUPITER");
    
    // Extremely high solar activity (solar flare)
    params.solar_activity = 10.0;        // Maximum value
    
    // Almost no shielding
    params.shielding_thickness_mm = 0.05; // Minimal shielding
    
    // Inside SAA (though not applicable for Jupiter, helps increase radiation)
    params.inside_saa = true;
    
    // Custom mission name
    params.mission_name = "EXTREME RADIATION ENVIRONMENT";
    
    return params;
}
```

### 7.2 Particle Energy Spectra

The simulator models realistic energy distributions for different particle types:

```cpp
// Initialize energy spectra for different particle types
void initializeEnergySpectra() {
    // Proton energy spectrum
    energy_spectra_[ParticleType::PROTON] = {
        // Log-normal distribution with parameters dependent on environment
        [this](double e) -> double {
            double mu = 1.5 + 0.5 * env_params_.solar_activity / 10.0;
            double sigma = 0.8;
            if (e <= 0) return 0.0;
            return (1.0 / (e * sigma * std::sqrt(2 * M_PI))) * 
                   std::exp(-std::pow(std::log(e) - mu, 2) / (2 * sigma * sigma));
        },
        0.1,   // Min energy (MeV)
        1000.0, // Max energy (MeV)
        5.0     // Peak energy (MeV)
    };
    
    // Electron energy spectrum
    energy_spectra_[ParticleType::ELECTRON] = {
        // Power law distribution
        [this](double e) -> double {
            double index = 1.5 + 0.5 * env_params_.solar_activity / 10.0;
            if (e <= 0) return 0.0;
            return std::pow(e, -index);
        },
        0.01,   // Min energy (MeV)
        10.0,   // Max energy (MeV)
        0.1     // Peak energy (MeV)
    };
    
    // Heavy ion energy spectrum
    energy_spectra_[ParticleType::HEAVY_ION] = {
        // Power law with exponential cutoff
        [this](double e) -> double {
            double index = 1.0 + 0.3 * env_params_.gcr_modulation;
            double cutoff = 100.0 + 900.0 * env_params_.gcr_modulation;
            if (e <= 0) return 0.0;
            return std::pow(e, -index) * std::exp(-e / cutoff);
        },
        10.0,    // Min energy (MeV)
        10000.0, // Max energy (MeV)
        100.0    // Peak energy (MeV)
    };
}
```

### 7.3 Event Generation

The simulator generates realistic radiation events with physically accurate characteristics:

```cpp
template <typename T>
RadiationEvent generateRadiationEvent(T* memory, size_t size) {
    RadiationEvent event;
    
    // Choose particle type based on flux ratios
    event.particle_type = selectParticleType();
    
    // Sample particle energy from appropriate distribution
    event.energy_mev = sampleParticleEnergy(event.particle_type);
    
    // Calculate LET from particle type and energy
    event.let_mev_cm2_g = calculateLET(event.particle_type, event.energy_mev);
    
    // Choose random location in memory
    std::uniform_int_distribution<size_t> loc_dist(0, size - 1);
    event.memory_offset = loc_dist(random_engine_);
    
    // Determine effect type based on particle properties and LET
    double seu_threshold = 0.1;  // LET threshold for SEU
    double mbu_threshold = 5.0;  // LET threshold for MBU
    double sel_threshold = 15.0; // LET threshold for SEL
    
    uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(memory) + event.memory_offset;
    
    // Effect depends on particle LET and random chance
    std::uniform_real_distribution<double> effect_dist(0.0, 1.0);
    double effect_rnd = effect_dist(random_engine_);
    
    if (event.let_mev_cm2_g >= sel_threshold && effect_rnd < 0.15) {
        // Single Event Latchup (SEL)
        event.type = RadiationEffectType::SINGLE_EVENT_LATCHUP;
        
        // Implementation details...
    }
    else if (event.let_mev_cm2_g >= mbu_threshold && effect_rnd < 0.6) {
        // Multiple Bit Upset (MBU)
        event.type = RadiationEffectType::MULTI_BIT_UPSET;
        
        // Implementation details...
    }
    else if (event.let_mev_cm2_g >= seu_threshold || 
            (event.particle_type == ParticleType::ELECTRON && effect_rnd < 0.1)) {
        // Single Bit Flip (SEU)
        event.type = RadiationEffectType::SINGLE_BIT_FLIP;
        
        // Implementation details...
    }
    else {
        // Single Event Transient (SET)
        event.type = RadiationEffectType::SINGLE_EVENT_TRANSIENT;
        
        // Implementation details...
    }
    
    return event;
}
```

## 8. Mission Simulation System

### 8.1 Mission Profile Definition

The framework defines mission profiles for various space environments:

```cpp
static MissionProfile createStandard(const std::string& mission_type) {
    MissionProfile profile;
    profile.name = mission_type;
    
    if (mission_type == "LEO" || mission_type == "LEO_EARTH_OBSERVATION") {
        // Low Earth Orbit
        auto normal_env = RadiationSimulator::getMissionEnvironment("LEO");
        
        // SAA environment (higher radiation)
        auto saa_env = normal_env;
        saa_env.inside_saa = true;
        saa_env.mission_name = "LEO (SAA)";
        
        // Solar flare environment
        auto flare_env = normal_env;
        flare_env.solar_activity = 9.0;
        flare_env.mission_name = "LEO (Solar Flare)";
        
        profile.environments = {normal_env, saa_env, flare_env};
        profile.environment_durations = {5400.0, 600.0, 300.0}; // 90 min, 10 min, 5 min
        profile.transition_probabilities = {0.8, 0.15, 0.05};
    }
    else if (mission_type == "MARS") {
        // Mars mission
        auto normal_env = RadiationSimulator::getMissionEnvironment("MARS");
        
        // Solar flare environment
        auto flare_env = normal_env;
        flare_env.solar_activity = 9.0;
        flare_env.mission_name = "Mars (Solar Flare)";
        
        // Dust storm environment (less shielding from atmosphere)
        auto storm_env = normal_env;
        storm_env.shielding_thickness_mm *= 0.8;
        storm_env.mission_name = "Mars (Dust Storm)";
        
        profile.environments = {normal_env, flare_env, storm_env};
        profile.environment_durations = {86400.0, 3600.0, 43200.0}; // 24 hours, 1 hour, 12 hours
        profile.transition_probabilities = {0.9, 0.05, 0.05};
    }
    // Other mission definitions...
    
    return profile;
}
```

### 8.2 Mission Simulation Loop

The mission simulator implements a complete simulation loop with environment transitions:

```cpp
MissionStatistics runSimulation(
    std::chrono::seconds total_duration,
    std::chrono::milliseconds time_step = std::chrono::milliseconds(1000),
    std::function<void(const RadiationSimulator::EnvironmentParams&)> on_environment_change = nullptr) {
    
    std::chrono::milliseconds elapsed_time(0);
    std::chrono::milliseconds last_scrubbing_time(0);
    std::chrono::milliseconds current_env_time(0);
    
    // Setup memory scrubber if enabled
    memory::MemoryScrubber scrubber;
    for (const auto& region : memory_regions_) {
        if (region.is_protected) {
            scrubber.registerMemoryRegion(region.ptr, region.size);
        }
    }
    
    // Main simulation loop
    while (elapsed_time < total_duration) {
        // Check for environment change
        bool environment_changed = false;
        
        if (current_env_time >= std::chrono::milliseconds(static_cast<long long>(
                profile_.environment_durations[current_environment_index_] * 1000))) {
            // Time to change environment based on duration
            environment_changed = selectNextEnvironment();
            current_env_time = std::chrono::milliseconds(0);
        } else {
            // Random chance to change environment
            std::uniform_real_distribution<double> change_dist(0.0, 1.0);
            if (change_dist(random_engine_) < 0.01) { // 1% chance per time step
                environment_changed = selectNextEnvironment();
                current_env_time = std::chrono::milliseconds(0);
            }
        }
        
        if (environment_changed && on_environment_change) {
            on_environment_change(simulator_->getSimulationEnvironment());
        }
        
        // Update statistics
        const auto& env = simulator_->getSimulationEnvironment();
        stats_.time_in_environment[env.mission_name] += time_step.count() / 1000.0;
        
        // Determine protection level based on current environment
        ProtectionLevel protection_level = determineProtectionLevel();
        
        // Apply protection settings based on level
        // Implementation details...
        
        // Simulate radiation effects on all memory regions
        for (const auto& region : memory_regions_) {
            auto events = simulator_->simulateEffects(
                region.ptr, region.size, time_step);
            
            // Process radiation events
            // Implementation details...
        }
        
        // Perform memory scrubbing if enabled
        if (scrubbing_enabled && (elapsed_time - last_scrubbing_time) >= scrubbing_interval) {
            scrubber.scrubMemory();
            stats_.scrubbing_cycles++;
            last_scrubbing_time = elapsed_time;
        }
        
        // Update simulation time
        elapsed_time += time_step;
        current_env_time += time_step;
    }
    
    // Calculate mission-critical uptime
    stats_.avg_mission_critical_uptime = static_cast<double>(stats_.errors_corrected) / 
        (stats_.errors_detected > 0 ? stats_.errors_detected : 1);
    
    return stats_;
}
```

### 8.3 Environment Transitions

The simulator models realistic transitions between different radiation environments:

```cpp
bool selectNextEnvironment() {
    if (profile_.environments.size() <= 1) {
        return false;
    }
    
    size_t old_index = current_environment_index_;
    
    // Choose next environment based on transition probabilities
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double val = dist(random_engine_);
    
    double cumulative_prob = 0.0;
    for (size_t i = 0; i < profile_.transition_probabilities.size(); ++i) {
        cumulative_prob += profile_.transition_probabilities[i];
        if (val < cumulative_prob) {
            current_environment_index_ = i;
            break;
        }
    }
    
    // Update simulator with new environment
    simulator_->updateEnvironment(profile_.environments[current_environment_index_]);
    
    return old_index != current_environment_index_;
}
```

## 9. Verification and Testing Framework

### 9.1 Extreme Radiation Stress Test

The framework includes comprehensive stress testing capabilities:

```cpp
void runStressTest(
    std::chrono::seconds duration = std::chrono::minutes(5),
    std::chrono::seconds reporting_interval = std::chrono::seconds(15)
) {
    // Create extreme radiation environment
    auto extreme_env = createExtremeEnvironment();
    testing::RadiationSimulator simulator(extreme_env);
    
    // Create adaptive protection with maximum level
    core::radiation::AdaptiveProtection protection(
        core::radiation::AdaptiveProtection::ProtectionLevel::MAXIMUM);
    
    // Create test memory with TMR-protected values
    const size_t memory_size = 10000;
    std::vector<core::redundancy::EnhancedTMR<float>> protected_values(memory_size, 
        core::redundancy::EnhancedTMR<float>(1.0f));
    
    // Create memory scrubber
    memory::MemoryScrubber scrubber;
    scrubber.registerMemoryRegion(protected_values.data(), 
        protected_values.size() * sizeof(core::redundancy::EnhancedTMR<float>));
    
    // Set up test record collection
    std::vector<StressTestRecord> records;
    
    // Tracking stats
    size_t total_radiation_events = 0;
    size_t total_bit_flips = 0;
    size_t total_multi_bit_upsets = 0;
    size_t total_latchups = 0;
    size_t total_transients = 0;
    size_t total_detected_errors = 0;
    size_t total_corrected_errors = 0;
    size_t total_uncorrectable_errors = 0;
    
    // Start time measurement
    auto start_time = std::chrono::steady_clock::now();
    auto last_report_time = start_time;
    
    // Run simulation in chunks matching reporting interval
    while (std::chrono::steady_clock::now() - start_time < duration) {
        // Check if it's time for a report
        auto current_time = std::chrono::steady_clock::now();
        bool report_due = (current_time - last_report_time >= reporting_interval);
        
        if (report_due) {
            // Record statistics
            // Implementation details...
            
            // Perform memory scrubbing
            size_t errors = scrubber.scrubMemory();
            total_detected_errors += errors;
            
            // Update protection system
            protection.updateEnvironment(errors, 0);
        }
        
        // Simulate radiation effects for this interval
        auto sim_interval = std::min(
            reporting_interval, 
            std::chrono::duration_cast<std::chrono::seconds>(
                duration - (current_time - start_time))
        );
        
        // Run simulation
        auto events = simulator.simulateEffects(
            protected_values.data(), 
            protected_values.size() * sizeof(core::redundancy::EnhancedTMR<float>),
            std::chrono::milliseconds(sim_interval));
        
        // Update statistics
        total_radiation_events += events.size();
        
        // Process events
        // Implementation details...
    }
    
    // Calculate error rates and recovery statistics
    double events_per_second = static_cast<double>(total_radiation_events) / total_duration;
    double errors_per_second = static_cast<double>(total_detected_errors) / total_duration;
    double recovery_rate = 100.0;
    if (total_detected_errors > 0) {
        recovery_rate = 100.0 * (static_cast<double>(total_detected_errors - incorrect_values) / 
                               total_detected_errors);
    }
    
    // Print final report with detailed statistics
    // Implementation details...
}
```

### 9.2 Comparative TMR Implementation Testing

The framework includes test infrastructure to compare different TMR implementations:

```cpp
// Template function to run a stress test on a TMR implementation
template <template<typename> class TMR, typename T>
TestResults runStressTest(
    const std::string& name,
    SpaceEnvironmentSimulator& env,
    const T& initial_value,
    size_t num_elements,
    std::chrono::milliseconds duration,
    bool show_progress = true) {
    
    // Setup test data
    std::vector<TMR<T>> tmr_elements;
    std::vector<T> expected_values;
    
    for (size_t i = 0; i < num_elements; ++i) {
        tmr_elements.push_back(TMR<T>(initial_value));
        expected_values.push_back(initial_value);
    }
    
    // Generate radiation events
    auto events = env.simulateRadiation(num_elements, duration);
    
    TestResults results;
    results.total_events = events.size();
    
    // Initialize error type counts
    results.error_type_counts[SpaceEnvironmentSimulator::ErrorType::SINGLE_BIT_FLIP] = 0;
    results.error_type_counts[SpaceEnvironmentSimulator::ErrorType::MULTI_BIT_UPSET] = 0;
    results.error_type_counts[SpaceEnvironmentSimulator::ErrorType::STUCK_BIT] = 0;
    results.error_type_counts[SpaceEnvironmentSimulator::ErrorType::SEVERE_CORRUPTION] = 0;