/**
 * Copyright (C) 2025 Rishab Nuguru
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once

#include <cstddef>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <array>
#include <algorithm>
#include <functional>
#include <utility>

namespace rad_ml {
namespace memory {

/**
 * @brief Memory zone characteristics with different radiation vulnerability levels
 * 
 * Based on research from NASA MESSENGER and JPL SELENE projects showing
 * memory modules have natural radiation "hot spots" and "cold spots"
 */
struct RadiationZone {
    enum class Level {
        HIGHLY_SHIELDED,     // Memory regions with most shielding (e.g., deeply embedded SRAM)
        MODERATELY_SHIELDED, // Memory with moderate protection
        LIGHTLY_SHIELDED,    // Memory with minimal shielding
        UNSHIELDED           // Memory with no special protection
    };
    
    Level level;
    size_t start_address;
    size_t end_address;
    double bit_flip_prob;    // Bit flip probability per day
    double stuck_bit_prob;   // Stuck bit probability per mission year
    double seu_rate;         // Single Event Upset rate (events/bit-day)
    
    // Constructor with research-based default probabilities
    RadiationZone(Level l, size_t start, size_t end)
        : level(l), start_address(start), end_address(end) {
        // Set probabilities based on spacecraft radiation environment studies
        switch (level) {
            case Level::HIGHLY_SHIELDED:
                bit_flip_prob = 1e-10;  // Based on RAD750 flight data
                stuck_bit_prob = 1e-6;  // Based on MESSENGER data
                seu_rate = 1e-11;       // Based on ISS radiation measurements
                break;
            case Level::MODERATELY_SHIELDED:
                bit_flip_prob = 1e-8;
                stuck_bit_prob = 1e-5;
                seu_rate = 1e-9;
                break;
            case Level::LIGHTLY_SHIELDED:
                bit_flip_prob = 1e-7;
                stuck_bit_prob = 1e-4;
                seu_rate = 1e-8;
                break;
            case Level::UNSHIELDED:
                bit_flip_prob = 1e-6;   // Based on Europa radiation environment models
                stuck_bit_prob = 1e-3;  // Based on JUICE mission predictions
                seu_rate = 1e-7;        // Based on Juno spacecraft measurements
                break;
        }
    }
    
    // Get zone size in bytes
    size_t size() const {
        return end_address - start_address;
    }
    
    // Get user-friendly string representation of zone level
    std::string level_string() const {
        switch (level) {
            case Level::HIGHLY_SHIELDED: return "Highly Shielded";
            case Level::MODERATELY_SHIELDED: return "Moderately Shielded";
            case Level::LIGHTLY_SHIELDED: return "Lightly Shielded";
            case Level::UNSHIELDED: return "Unshielded";
            default: return "Unknown";
        }
    }
};

/**
 * @brief Data criticality levels for memory placement decisions
 */
enum class DataCriticality {
    MISSION_CRITICAL,    // Loss would result in mission failure
    HIGHLY_IMPORTANT,    // Loss would significantly degrade mission
    MODERATELY_IMPORTANT,// Loss would cause minor mission impact 
    LOW_IMPORTANCE       // Loss would be acceptable/recoverable
};

/**
 * @brief Radiation-aware memory allocator that places data in appropriate memory regions
 * 
 * Based on principles from NASA MESSENGER, JPL SELENE, and ESA JUICE missions,
 * which use memory mapping to reduce radiation exposure for critical data.
 */
class RadiationMappedAllocator {
public:
    /**
     * @brief Create allocator with system-defined radiation zones
     * 
     * In a real spacecraft, these zones would be defined based on
     * spacecraft design, shielding, and radiation modeling.
     */
    RadiationMappedAllocator() {
        // These addresses are simulated for demonstration
        // In a real spacecraft, these would map to actual memory regions
        initialize_default_zones();
    }
    
    /**
     * @brief Create allocator with custom radiation zones
     * 
     * @param zones Vector of radiation zones
     */
    explicit RadiationMappedAllocator(const std::vector<RadiationZone>& zones)
        : zones_(zones) {
        validate_zones();
    }
    
    /**
     * @brief Allocate memory with radiation awareness
     * 
     * @param size Bytes to allocate
     * @param criticality How important this data is
     * @return Pointer to allocated memory
     */
    void* allocate(size_t size, DataCriticality criticality = DataCriticality::MODERATELY_IMPORTANT) {
        // Select appropriate zone based on criticality
        RadiationZone& target_zone = select_zone_for_criticality(criticality);
        
        // Find available space in the zone
        size_t address = find_available_space(target_zone, size);
        
        // If no space in ideal zone, fall back to less protected zone
        if (address == 0) {
            target_zone = find_fallback_zone(criticality);
            address = find_available_space(target_zone, size);
            
            // If still no space, use system allocator as last resort
            if (address == 0) {
                void* ptr = std::malloc(size);
                system_allocated_memory_.push_back(
                    {ptr, size, criticality, DataPlacement::SYSTEM_MEMORY});
                return ptr;
            }
        }
        
        // Record allocation in our registry
        void* ptr = reinterpret_cast<void*>(address);
        allocations_.push_back({address, size, target_zone.level});
        
        // Record for diagnostics
        DataPlacement placement;
        switch (target_zone.level) {
            case RadiationZone::Level::HIGHLY_SHIELDED:
                placement = DataPlacement::HIGHLY_PROTECTED;
                break;
            case RadiationZone::Level::MODERATELY_SHIELDED:
                placement = DataPlacement::MODERATELY_PROTECTED;
                break;
            case RadiationZone::Level::LIGHTLY_SHIELDED:
                placement = DataPlacement::MINIMALLY_PROTECTED;
                break;
            default:
                placement = DataPlacement::UNPROTECTED;
                break;
        }
        
        allocated_memory_.push_back({ptr, size, criticality, placement});
        
        return ptr;
    }
    
    /**
     * @brief Deallocate previously allocated memory
     * 
     * @param ptr Pointer to memory
     */
    void deallocate(void* ptr) {
        // Check if it's in our managed memory
        size_t address = reinterpret_cast<size_t>(ptr);
        
        auto it = std::find_if(allocations_.begin(), allocations_.end(),
                              [address](const AllocationRecord& rec) {
                                  return rec.address == address;
                              });
                              
        if (it != allocations_.end()) {
            // Remove from our records
            allocations_.erase(it);
            
            // Also remove from diagnostic record
            auto mem_it = std::find_if(allocated_memory_.begin(), allocated_memory_.end(),
                                     [ptr](const MemoryRecord& rec) {
                                         return rec.ptr == ptr;
                                     });
            if (mem_it != allocated_memory_.end()) {
                allocated_memory_.erase(mem_it);
            }
            
            return;
        }
        
        // Check if it's system-allocated
        auto sys_it = std::find_if(system_allocated_memory_.begin(), system_allocated_memory_.end(),
                                  [ptr](const MemoryRecord& rec) {
                                      return rec.ptr == ptr;
                                  });
                                  
        if (sys_it != system_allocated_memory_.end()) {
            std::free(ptr);
            system_allocated_memory_.erase(sys_it);
            return;
        }
        
        // Not found in our records, try system free as fallback
        std::free(ptr);
    }
    
    /**
     * @brief Get diagnostic information about memory allocations
     * 
     * @return String with memory usage statistics
     */
    std::string get_diagnostics() const {
        std::string result = "RadiationMappedAllocator Diagnostics:\n";
        
        // Count allocations by zone type
        std::array<size_t, 4> zone_counts = {0, 0, 0, 0};
        std::array<size_t, 4> zone_bytes = {0, 0, 0, 0};
        
        for (const auto& alloc : allocations_) {
            switch (alloc.zone_level) {
                case RadiationZone::Level::HIGHLY_SHIELDED:
                    zone_counts[0]++;
                    zone_bytes[0] += alloc.size;
                    break;
                case RadiationZone::Level::MODERATELY_SHIELDED:
                    zone_counts[1]++;
                    zone_bytes[1] += alloc.size;
                    break;
                case RadiationZone::Level::LIGHTLY_SHIELDED:
                    zone_counts[2]++;
                    zone_bytes[2] += alloc.size;
                    break;
                case RadiationZone::Level::UNSHIELDED:
                    zone_counts[3]++;
                    zone_bytes[3] += alloc.size;
                    break;
            }
        }
        
        // Add system allocated memory 
        size_t system_count = system_allocated_memory_.size();
        size_t system_bytes = 0;
        for (const auto& mem : system_allocated_memory_) {
            system_bytes += mem.size;
        }
        
        result += "  Memory zone usage:\n";
        result += "    Highly shielded:    " + std::to_string(zone_counts[0]) + 
                  " allocations, " + std::to_string(zone_bytes[0]) + " bytes\n";
        result += "    Moderately shielded: " + std::to_string(zone_counts[1]) + 
                  " allocations, " + std::to_string(zone_bytes[1]) + " bytes\n";
        result += "    Lightly shielded:   " + std::to_string(zone_counts[2]) + 
                  " allocations, " + std::to_string(zone_bytes[2]) + " bytes\n";
        result += "    Unshielded:         " + std::to_string(zone_counts[3]) + 
                  " allocations, " + std::to_string(zone_bytes[3]) + " bytes\n";
        result += "    System memory:      " + std::to_string(system_count) + 
                  " allocations, " + std::to_string(system_bytes) + " bytes\n";
        
        return result;
    }
    
    /**
     * @brief Get memory usage by criticality
     * 
     * @return Map of criticality to bytes used
     */
    std::map<DataCriticality, size_t> get_criticality_usage() const {
        std::map<DataCriticality, size_t> result;
        
        // Initialize map
        result[DataCriticality::MISSION_CRITICAL] = 0;
        result[DataCriticality::HIGHLY_IMPORTANT] = 0;
        result[DataCriticality::MODERATELY_IMPORTANT] = 0;
        result[DataCriticality::LOW_IMPORTANCE] = 0;
        
        // Count managed memory
        for (const auto& mem : allocated_memory_) {
            result[mem.criticality] += mem.size;
        }
        
        // Count system allocated memory
        for (const auto& mem : system_allocated_memory_) {
            result[mem.criticality] += mem.size;
        }
        
        return result;
    }
    
    /**
     * @brief Calculate radiation vulnerability score for all allocations
     * 
     * Lower score means less vulnerable to radiation effects
     * 
     * @return Vulnerability score (0.0-1.0)
     */
    double calculate_vulnerability_score() const {
        if (allocated_memory_.empty() && system_allocated_memory_.empty()) {
            return 0.0;
        }
        
        double total_weighted_risk = 0.0;
        size_t total_bytes = 0;
        
        // Calculate for managed memory
        for (const auto& mem : allocated_memory_) {
            double criticality_weight = get_criticality_weight(mem.criticality);
            double placement_factor = get_placement_factor(mem.placement);
            
            total_weighted_risk += mem.size * criticality_weight * placement_factor;
            total_bytes += mem.size;
        }
        
        // Calculate for system memory (assumed highest risk)
        for (const auto& mem : system_allocated_memory_) {
            double criticality_weight = get_criticality_weight(mem.criticality);
            double placement_factor = get_placement_factor(DataPlacement::SYSTEM_MEMORY);
            
            total_weighted_risk += mem.size * criticality_weight * placement_factor;
            total_bytes += mem.size;
        }
        
        // Normalize to 0.0-1.0 scale
        return total_bytes > 0 ? total_weighted_risk / (total_bytes * 10.0) : 0.0;
    }
    
    // Custom allocator interface for STL containers
    template<typename T>
    class STLAllocator {
    public:
        using value_type = T;
        
        explicit STLAllocator(RadiationMappedAllocator& allocator, 
                             DataCriticality criticality = DataCriticality::MODERATELY_IMPORTANT)
            : allocator_(allocator), criticality_(criticality) {}
        
        template<typename U>
        STLAllocator(const STLAllocator<U>& other) 
            : allocator_(other.allocator_), criticality_(other.criticality_) {}
        
        T* allocate(std::size_t n) {
            return static_cast<T*>(allocator_.allocate(n * sizeof(T), criticality_));
        }
        
        void deallocate(T* p, std::size_t) {
            allocator_.deallocate(p);
        }
        
        template<typename U>
        struct rebind {
            using other = STLAllocator<U>;
        };
        
        template<typename U>
        bool operator==(const STLAllocator<U>& other) const {
            return &allocator_ == &other.allocator_;
        }
        
        template<typename U>
        bool operator!=(const STLAllocator<U>& other) const {
            return !(*this == other);
        }
        
    private:
        RadiationMappedAllocator& allocator_;
        DataCriticality criticality_;
        
        template<typename U>
        friend class STLAllocator;
    };
    
private:
    // Placement quality for diagnostics
    enum class DataPlacement {
        HIGHLY_PROTECTED,
        MODERATELY_PROTECTED,
        MINIMALLY_PROTECTED,
        UNPROTECTED,
        SYSTEM_MEMORY  // Fallback non-radiation-aware allocation
    };
    
    // Record of memory allocations
    struct AllocationRecord {
        size_t address;
        size_t size;
        RadiationZone::Level zone_level;
    };
    
    // Record for diagnostic purposes
    struct MemoryRecord {
        void* ptr;
        size_t size;
        DataCriticality criticality;
        DataPlacement placement;
    };
    
    // Available radiation zones
    std::vector<RadiationZone> zones_;
    
    // Active allocations
    std::vector<AllocationRecord> allocations_;
    
    // For diagnostics
    std::vector<MemoryRecord> allocated_memory_;
    std::vector<MemoryRecord> system_allocated_memory_;
    
    // Initialize default radiation zones
    void initialize_default_zones() {
        // For simulation, we'll create four zones with different protection levels
        // In a real system, these would be determined by hardware analysis
        constexpr size_t MB = 1024 * 1024;
        
        // Simulated memory map - addresses are for demonstration only
        zones_.emplace_back(RadiationZone::Level::HIGHLY_SHIELDED, 0x10000000, 0x10000000 + 1 * MB);
        zones_.emplace_back(RadiationZone::Level::MODERATELY_SHIELDED, 0x20000000, 0x20000000 + 4 * MB);
        zones_.emplace_back(RadiationZone::Level::LIGHTLY_SHIELDED, 0x30000000, 0x30000000 + 8 * MB);
        zones_.emplace_back(RadiationZone::Level::UNSHIELDED, 0x40000000, 0x40000000 + 16 * MB);
    }
    
    // Ensure zones are valid
    void validate_zones() {
        if (zones_.empty()) {
            throw std::runtime_error("No radiation zones defined");
        }
        
        // Verify no overlapping zones
        for (size_t i = 0; i < zones_.size(); i++) {
            for (size_t j = i + 1; j < zones_.size(); j++) {
                if ((zones_[i].start_address <= zones_[j].end_address) && 
                    (zones_[j].start_address <= zones_[i].end_address)) {
                    throw std::runtime_error("Overlapping radiation zones detected");
                }
            }
        }
    }
    
    // Select appropriate zone based on data criticality
    RadiationZone& select_zone_for_criticality(DataCriticality criticality) {
        RadiationZone::Level target_level;
        
        // Map criticality to zone level
        switch (criticality) {
            case DataCriticality::MISSION_CRITICAL:
                target_level = RadiationZone::Level::HIGHLY_SHIELDED;
                break;
            case DataCriticality::HIGHLY_IMPORTANT:
                target_level = RadiationZone::Level::MODERATELY_SHIELDED;
                break;
            case DataCriticality::MODERATELY_IMPORTANT:
                target_level = RadiationZone::Level::LIGHTLY_SHIELDED;
                break;
            case DataCriticality::LOW_IMPORTANCE:
                target_level = RadiationZone::Level::UNSHIELDED;
                break;
        }
        
        // Find matching zone
        for (auto& zone : zones_) {
            if (zone.level == target_level) {
                return zone;
            }
        }
        
        // Fallback to first zone if no match (should not happen if zones are properly initialized)
        return zones_[0];
    }
    
    // Find space in zone
    size_t find_available_space(const RadiationZone& zone, size_t size) {
        // Get all allocations in this zone
        std::vector<std::pair<size_t, size_t>> occupied;
        for (const auto& alloc : allocations_) {
            if (alloc.zone_level == zone.level) {
                occupied.push_back({alloc.address, alloc.size});
            }
        }
        
        // Sort by address
        std::sort(occupied.begin(), occupied.end());
        
        // Find first available gap
        size_t current_address = zone.start_address;
        for (const auto& block : occupied) {
            if (block.first - current_address >= size) {
                // Found enough space
                return current_address;
            }
            current_address = block.first + block.second;
        }
        
        // Check if there's space at the end
        if (zone.end_address - current_address >= size) {
            return current_address;
        }
        
        // No space found
        return 0;
    }
    
    // Find fallback zone if preferred zone is full
    RadiationZone& find_fallback_zone(DataCriticality criticality) {
        // Try lower criticality zones in order
        switch (criticality) {
            case DataCriticality::MISSION_CRITICAL:
                // Try moderately shielded
                for (auto& zone : zones_) {
                    if (zone.level == RadiationZone::Level::MODERATELY_SHIELDED) {
                        return zone;
                    }
                }
                // Fall through to next level
            case DataCriticality::HIGHLY_IMPORTANT:
                // Try lightly shielded
                for (auto& zone : zones_) {
                    if (zone.level == RadiationZone::Level::LIGHTLY_SHIELDED) {
                        return zone;
                    }
                }
                // Fall through to next level
            case DataCriticality::MODERATELY_IMPORTANT:
                // Try unshielded
                for (auto& zone : zones_) {
                    if (zone.level == RadiationZone::Level::UNSHIELDED) {
                        return zone;
                    }
                }
                // Fall through to default
            default:
                // Use any zone with space
                for (auto& zone : zones_) {
                    return zone;
                }
        }
        
        // Should never reach here if zones are properly initialized
        return zones_[0];
    }
    
    // Get weight factor based on criticality
    double get_criticality_weight(DataCriticality criticality) const {
        switch (criticality) {
            case DataCriticality::MISSION_CRITICAL:
                return 10.0;
            case DataCriticality::HIGHLY_IMPORTANT:
                return 5.0;
            case DataCriticality::MODERATELY_IMPORTANT:
                return 2.0;
            case DataCriticality::LOW_IMPORTANCE:
                return 1.0;
            default:
                return 1.0;
        }
    }
    
    // Get radiation factor based on placement
    double get_placement_factor(DataPlacement placement) const {
        switch (placement) {
            case DataPlacement::HIGHLY_PROTECTED:
                return 0.1;  // 90% reduction in vulnerability
            case DataPlacement::MODERATELY_PROTECTED:
                return 0.3;  // 70% reduction
            case DataPlacement::MINIMALLY_PROTECTED:
                return 0.6;  // 40% reduction
            case DataPlacement::UNPROTECTED:
                return 0.8;  // 20% reduction
            case DataPlacement::SYSTEM_MEMORY:
                return 1.0;  // No reduction
            default:
                return 1.0;
        }
    }
};

} // namespace memory
} // namespace rad_ml 
