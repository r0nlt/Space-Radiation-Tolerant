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
#include <array>
#include <type_traits>
#include <new>
#include <stdexcept>

namespace rad_ml {
namespace core {
namespace memory {

/**
 * @brief A static memory allocator that preallocates a fixed amount of memory
 * at compile time and manages allocations from that pool.
 * 
 * This allocator never uses dynamic memory allocation and is designed for
 * radiation-tolerant applications where predictable memory usage is critical.
 */
template <typename T, std::size_t Capacity>
class StaticAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template <typename U>
    struct rebind {
        using other = StaticAllocator<U, Capacity>;
    };
    
    // Constructor
    constexpr StaticAllocator() noexcept : next_free_(0) {}
    
    // Copy constructor
    template <typename U>
    constexpr StaticAllocator(const StaticAllocator<U, Capacity>&) noexcept {}
    
    // Allocate memory
    [[nodiscard]] pointer allocate(size_type n) {
        const size_type alignment = alignof(T);
        const size_type aligned_next = (next_free_ + alignment - 1) & ~(alignment - 1);
        
        if (aligned_next + n * sizeof(T) > Capacity) {
            throw std::bad_alloc();
        }
        
        pointer result = reinterpret_cast<pointer>(&storage_[aligned_next]);
        next_free_ = aligned_next + n * sizeof(T);
        return result;
    }
    
    // Deallocate memory (no-op in this implementation for simplicity)
    void deallocate([[maybe_unused]] pointer p, [[maybe_unused]] size_type n) noexcept {
        // In a real implementation, we would track and reuse freed memory
        // For simplicity, this example doesn't reclaim memory
    }
    
    // Reset the allocator to its initial state
    void reset() noexcept {
        next_free_ = 0;
    }
    
    // Maximum number of objects that can be allocated
    constexpr size_type max_size() const noexcept {
        return Capacity / sizeof(T);
    }
    
    // Construct an object at the given address
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        new (p) U(std::forward<Args>(args)...);
    }
    
    // Destroy an object at the given address
    template <typename U>
    void destroy(U* p) {
        p->~U();
    }
    
private:
    // Storage for allocations
    alignas(alignof(std::max_align_t)) std::array<std::byte, Capacity> storage_;
    size_type next_free_;
};

// Equality comparison
template <typename T, std::size_t Capacity, typename U, std::size_t OtherCapacity>
bool operator==(const StaticAllocator<T, Capacity>&, const StaticAllocator<U, OtherCapacity>&) noexcept {
    return Capacity == OtherCapacity;
}

template <typename T, std::size_t Capacity, typename U, std::size_t OtherCapacity>
bool operator!=(const StaticAllocator<T, Capacity>&, const StaticAllocator<U, OtherCapacity>&) noexcept {
    return Capacity != OtherCapacity;
}

} // namespace memory
} // namespace core
} // namespace rad_ml 
