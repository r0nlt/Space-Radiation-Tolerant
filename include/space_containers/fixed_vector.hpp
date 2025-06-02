#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits>

namespace space_containers {

/**
 * @brief A fixed-size vector implementation for space flight software.
 *
 * This container provides deterministic behavior and zero dynamic allocation
 * guarantees required for radiation-tolerant space applications.
 *
 * @tparam T The type of elements stored in the vector
 * @tparam MaxSize The maximum number of elements that can be stored
 * @tparam Alignment Custom alignment value (defaults to alignof(T))
 */
template <typename T, std::size_t MaxSize, std::size_t Alignment = alignof(T)>
class alignas(Alignment) FixedVector {
   public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    // Telemetry counters
    mutable struct {
        std::size_t access_count{0};
        std::size_t error_count{0};
    } telemetry;

    // Constructors
    FixedVector() noexcept : size_(0) {}

    /**
     * @brief Copy constructor with validation
     */
    FixedVector(const FixedVector& other) noexcept(std::is_nothrow_copy_constructible_v<T>)
        : size_(other.size_)
    {
        std::memcpy(data_, other.data_, size_ * sizeof(T));
    }

    /**
     * @brief Move constructor with validation
     */
    FixedVector(FixedVector&& other) noexcept : size_(other.size_)
    {
        std::memcpy(data_, other.data_, size_ * sizeof(T));
        other.size_ = 0;
    }

    // Assignment operators
    FixedVector& operator=(const FixedVector& other) noexcept(std::is_nothrow_copy_assignable_v<T>)
    {
        if (this != &other) {
            size_ = other.size_;
            std::memcpy(data_, other.data_, size_ * sizeof(T));
        }
        return *this;
    }

    FixedVector& operator=(FixedVector&& other) noexcept
    {
        if (this != &other) {
            size_ = other.size_;
            std::memcpy(data_, other.data_, size_ * sizeof(T));
            other.size_ = 0;
        }
        return *this;
    }

    // Element access
    /**
     * @brief Safe element access with bounds checking
     * @throws std::out_of_range if index is out of bounds
     */
    reference operator[](size_type index)
    {
        if (!validate_index(index)) {
            ++telemetry.error_count;
            // In space applications, we use assert instead of exceptions
            assert(false && "Index out of bounds");
        }
        ++telemetry.access_count;
        return data_[index];
    }

    const_reference operator[](size_type index) const
    {
        if (!validate_index(index)) {
            ++telemetry.error_count;
            assert(false && "Index out of bounds");
        }
        ++telemetry.access_count;
        return data_[index];
    }

    /**
     * @brief Unsafe but faster element access without bounds checking
     * @warning Use only when index is guaranteed to be valid
     */
    reference unsafe_access(size_type index) noexcept
    {
        ++telemetry.access_count;
        return data_[index];
    }

    const_reference unsafe_access(size_type index) const noexcept
    {
        ++telemetry.access_count;
        return data_[index];
    }

    // Capacity
    [[nodiscard]] constexpr size_type max_size() const noexcept { return MaxSize; }
    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] bool full() const noexcept { return size_ == MaxSize; }

    // Modifiers
    bool push_back(const T& value) noexcept(std::is_nothrow_copy_constructible_v<T>)
    {
        if (full()) {
            ++telemetry.error_count;
            return false;
        }
        data_[size_++] = value;
        return true;
    }

    bool push_back(T&& value) noexcept(std::is_nothrow_move_constructible_v<T>)
    {
        if (full()) {
            ++telemetry.error_count;
            return false;
        }
        data_[size_++] = std::move(value);
        return true;
    }

    /**
     * @brief Constructs element in-place at the end of the vector
     * @return True if successful, false if vector is full
     */
    template <typename... Args>
    bool emplace_back(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>)
    {
        if (full()) {
            ++telemetry.error_count;
            return false;
        }
        new (&data_[size_]) T(std::forward<Args>(args)...);
        ++size_;
        return true;
    }

    void pop_back() noexcept
    {
        if (!empty()) {
            --size_;
        }
        else {
            ++telemetry.error_count;
        }
    }

    void clear() noexcept { size_ = 0; }

    // Iterators
    iterator begin() noexcept { return data_; }
    const_iterator begin() const noexcept { return data_; }
    const_iterator cbegin() const noexcept { return data_; }

    iterator end() noexcept { return data_ + size_; }
    const_iterator end() const noexcept { return data_ + size_; }
    const_iterator cend() const noexcept { return data_ + size_; }

    // Data validation
    /**
     * @brief Validates the integrity of the container
     * @return True if the container is in a valid state
     */
    [[nodiscard]] bool validate() const noexcept { return size_ <= MaxSize; }

    // Telemetry access
    [[nodiscard]] std::size_t get_access_count() const noexcept { return telemetry.access_count; }

    [[nodiscard]] std::size_t get_error_count() const noexcept { return telemetry.error_count; }

    void reset_telemetry() noexcept
    {
        telemetry.access_count = 0;
        telemetry.error_count = 0;
    }

   private:
    alignas(Alignment) T data_[MaxSize];
    size_type size_;

    [[nodiscard]] bool validate_index(size_type index) const noexcept { return index < size_; }
};

}  // namespace space_containers
