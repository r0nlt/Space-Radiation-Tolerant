/**
 * @file fixed_containers.hpp
 * @brief Fixed-size container implementations for space flight
 * 
 * This file provides space-optimized fixed-size container replacements for
 * STL containers like vector, map, etc. They use static pre-allocation
 * for deterministic behavior required in space applications.
 */

#ifndef RAD_ML_FIXED_CONTAINERS_HPP
#define RAD_ML_FIXED_CONTAINERS_HPP

#include <array>
#include <cstddef>
#include "../space_flight_config.hpp"

namespace rad_ml {
namespace core {
namespace memory {

/**
 * Error codes for fixed container operations
 */
enum class ContainerError {
    SUCCESS,
    FULL,
    OUT_OF_BOUNDS,
    NOT_FOUND,
    INVALID_OPERATION
};

/**
 * @brief Fixed-size vector replacement with deterministic behavior
 * 
 * This class provides a fixed-capacity array with vector-like interface
 * but without any dynamic memory allocation.
 * 
 * @tparam T Element type
 * @tparam Capacity Maximum container capacity
 */
template<typename T, size_t Capacity>
class FixedVector {
public:
    /**
     * Default constructor
     */
    FixedVector() : size_(0) {}
    
    /**
     * @brief Add element to the end if space available
     * 
     * @param value Value to add
     * @return ContainerError SUCCESS if added, FULL if no space left
     */
    ContainerError push_back(const T& value) {
        if (size_ >= Capacity) {
            return ContainerError::FULL;
        }
        
        data_[size_++] = value;
        return ContainerError::SUCCESS;
    }
    
    /**
     * @brief Get element at specified index
     * 
     * @param index Element index
     * @param[out] value Reference to store the value
     * @return ContainerError SUCCESS if found, OUT_OF_BOUNDS if invalid index
     */
    ContainerError at(size_t index, T& value) const {
        if (index >= size_) {
            return ContainerError::OUT_OF_BOUNDS;
        }
        
        value = data_[index];
        return ContainerError::SUCCESS;
    }
    
    /**
     * @brief Array-like access (no bounds checking)
     * 
     * @param index Element index (must be valid)
     * @return Reference to the element
     */
    T& operator[](size_t index) {
        return data_[index];
    }
    
    /**
     * @brief Array-like access (no bounds checking)
     * 
     * @param index Element index (must be valid)
     * @return Const reference to the element
     */
    const T& operator[](size_t index) const {
        return data_[index];
    }
    
    /**
     * @brief Current number of elements
     * 
     * @return Element count
     */
    size_t size() const {
        return size_;
    }
    
    /**
     * @brief Maximum capacity
     * 
     * @return Container capacity
     */
    size_t capacity() const {
        return Capacity;
    }
    
    /**
     * @brief Remove all elements
     */
    void clear() {
        size_ = 0;
    }
    
    /**
     * @brief Check if container is empty
     * 
     * @return True if empty
     */
    bool empty() const {
        return size_ == 0;
    }
    
private:
    std::array<T, Capacity> data_;
    size_t size_;
};

/**
 * @brief Fixed-size key-value map replacement
 * 
 * This class implements a simple key-value store with fixed capacity
 * and linear search for lookup. Optimized for small maps.
 * 
 * @tparam Key Key type
 * @tparam Value Value type
 * @tparam Capacity Maximum number of elements
 */
template<typename Key, typename Value, size_t Capacity>
class FixedMap {
public:
    /**
     * Default constructor
     */
    FixedMap() : size_(0) {}
    
    /**
     * @brief Insert or update key-value pair
     * 
     * @param key Key to insert/update
     * @param value Value to store
     * @return ContainerError SUCCESS if added/updated, FULL if no space
     */
    ContainerError insert(const Key& key, const Value& value) {
        // First check if key exists
        for (size_t i = 0; i < size_; ++i) {
            if (keys_[i] == key) {
                values_[i] = value;
                return ContainerError::SUCCESS;
            }
        }
        
        // If not found, add new entry
        if (size_ >= Capacity) {
            return ContainerError::FULL;
        }
        
        keys_[size_] = key;
        values_[size_] = value;
        ++size_;
        
        return ContainerError::SUCCESS;
    }
    
    /**
     * @brief Find value by key
     * 
     * @param key Key to find
     * @param[out] value Reference to store the value
     * @return ContainerError SUCCESS if found, NOT_FOUND otherwise
     */
    ContainerError find(const Key& key, Value& value) const {
        for (size_t i = 0; i < size_; ++i) {
            if (keys_[i] == key) {
                value = values_[i];
                return ContainerError::SUCCESS;
            }
        }
        
        return ContainerError::NOT_FOUND;
    }
    
    /**
     * @brief Remove entry by key
     * 
     * @param key Key to remove
     * @return ContainerError SUCCESS if removed, NOT_FOUND if key not found
     */
    ContainerError erase(const Key& key) {
        for (size_t i = 0; i < size_; ++i) {
            if (keys_[i] == key) {
                // Move last element to fill the gap (unless removing last element)
                if (i < size_ - 1) {
                    keys_[i] = keys_[size_ - 1];
                    values_[i] = values_[size_ - 1];
                }
                
                --size_;
                return ContainerError::SUCCESS;
            }
        }
        
        return ContainerError::NOT_FOUND;
    }
    
    /**
     * @brief Current number of elements
     * 
     * @return Element count
     */
    size_t size() const {
        return size_;
    }
    
    /**
     * @brief Maximum capacity
     * 
     * @return Container capacity
     */
    size_t capacity() const {
        return Capacity;
    }
    
    /**
     * @brief Remove all elements
     */
    void clear() {
        size_ = 0;
    }
    
private:
    std::array<Key, Capacity> keys_;
    std::array<Value, Capacity> values_;
    size_t size_;
};

} // namespace memory
} // namespace core
} // namespace rad_ml

#endif // RAD_ML_FIXED_CONTAINERS_HPP 