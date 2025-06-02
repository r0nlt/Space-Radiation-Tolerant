#pragma once

#include <algorithm>
#include <optional>
#include <utility>

#include "fixed_vector.hpp"

namespace space_containers {

/**
 * @brief A fixed-size map implementation for space flight software
 *
 * This container provides a deterministic, order-preserving map implementation
 * with zero dynamic allocation for space applications.
 *
 * @tparam Key The key type
 * @tparam Value The value type
 * @tparam MaxSize Maximum number of key-value pairs
 * @tparam Alignment Custom alignment value (defaults to maximum of Key and Value alignment)
 */
template <typename Key, typename Value, std::size_t MaxSize,
          std::size_t Alignment = (alignof(Key) > alignof(Value) ? alignof(Key) : alignof(Value))>
class FixedMap {
   public:
    using key_type = Key;
    using mapped_type = Value;
    using value_type = std::pair<const key_type, mapped_type>;
    using size_type = std::size_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    // Telemetry counters
    mutable struct {
        std::size_t access_count{0};
        std::size_t error_count{0};
        std::size_t find_count{0};
    } telemetry;

    // Element access
    /**
     * @brief Access or insert element
     * @return Reference to the value that is mapped to the key
     */
    mapped_type& operator[](const key_type& key)
    {
        ++telemetry.access_count;
        auto it = find_key(key);
        if (it != data_.size()) {
            return data_[it].second;
        }

        if (data_.full()) {
            ++telemetry.error_count;
            // In space applications, we use assert instead of exceptions
            assert(false && "Map is full");
            return data_[0].second;  // Return reference to first element as fallback
        }

        data_.push_back({key, mapped_type()});
        return data_.back().second;
    }

    /**
     * @brief Find element by key
     * @return Optional containing the value if found, empty optional otherwise
     */
    std::optional<mapped_type> find(const key_type& key) const
    {
        ++telemetry.find_count;
        auto it = find_key(key);
        if (it != data_.size()) {
            return data_[it].second;
        }
        return std::nullopt;
    }

    /**
     * @brief Insert a key-value pair
     * @return True if insertion successful, false if map is full or key exists
     */
    bool insert(const key_type& key, const mapped_type& value)
    {
        if (contains(key) || data_.full()) {
            ++telemetry.error_count;
            return false;
        }
        return data_.push_back({key, value});
    }

    /**
     * @brief Insert a key-value pair with move semantics
     */
    bool insert(const key_type& key, mapped_type&& value)
    {
        if (contains(key) || data_.full()) {
            ++telemetry.error_count;
            return false;
        }
        return data_.push_back({key, std::move(value)});
    }

    /**
     * @brief Construct element in-place
     */
    template <typename... Args>
    bool emplace(const key_type& key, Args&&... args)
    {
        if (contains(key) || data_.full()) {
            ++telemetry.error_count;
            return false;
        }
        return data_.emplace_back(std::piecewise_construct, std::forward_as_tuple(key),
                                  std::forward_as_tuple(std::forward<Args>(args)...));
    }

    /**
     * @brief Remove element by key
     * @return True if element was removed
     */
    bool erase(const key_type& key)
    {
        auto it = find_key(key);
        if (it == data_.size()) {
            return false;
        }

        // Preserve order by shifting elements
        for (size_type i = it; i < data_.size() - 1; ++i) {
            data_[i] = std::move(data_[i + 1]);
        }
        data_.pop_back();
        return true;
    }

    // Capacity
    [[nodiscard]] size_type size() const noexcept { return data_.size(); }
    [[nodiscard]] constexpr size_type max_size() const noexcept { return MaxSize; }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }
    [[nodiscard]] bool full() const noexcept { return data_.full(); }

    // Lookup
    [[nodiscard]] bool contains(const key_type& key) const { return find_key(key) != data_.size(); }

    void clear() noexcept { data_.clear(); }

    // Iterators
    using iterator = typename FixedVector<std::pair<key_type, mapped_type>, MaxSize>::iterator;
    using const_iterator =
        typename FixedVector<std::pair<key_type, mapped_type>, MaxSize>::const_iterator;

    iterator begin() noexcept { return data_.begin(); }
    const_iterator begin() const noexcept { return data_.begin(); }
    const_iterator cbegin() const noexcept { return data_.cbegin(); }

    iterator end() noexcept { return data_.end(); }
    const_iterator end() const noexcept { return data_.end(); }
    const_iterator cend() const noexcept { return data_.cend(); }

    // Telemetry access
    [[nodiscard]] std::size_t get_access_count() const noexcept { return telemetry.access_count; }

    [[nodiscard]] std::size_t get_error_count() const noexcept { return telemetry.error_count; }

    [[nodiscard]] std::size_t get_find_count() const noexcept { return telemetry.find_count; }

    void reset_telemetry() noexcept
    {
        telemetry.access_count = 0;
        telemetry.error_count = 0;
        telemetry.find_count = 0;
    }

   private:
    FixedVector<std::pair<key_type, mapped_type>, MaxSize, Alignment> data_;

    [[nodiscard]] size_type find_key(const key_type& key) const
    {
        for (size_type i = 0; i < data_.size(); ++i) {
            if (data_[i].first == key) {
                return i;
            }
        }
        return data_.size();  // Not found
    }
};

}  // namespace space_containers
