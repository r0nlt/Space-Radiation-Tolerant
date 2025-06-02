#pragma once

#include <cstdint>
#include <functional>

#include "fixed_vector.hpp"

namespace space_containers {

/**
 * @brief CRC calculation function type
 * @tparam T The type to calculate CRC for
 */
template <typename T>
using CRCFunction = std::function<std::uint32_t(const T*, std::size_t)>;

/**
 * @brief A radiation-hardened vector with automatic memory scrubbing and error detection
 *
 * This container maintains redundant copies of data and performs periodic validation
 * to protect against radiation-induced memory corruption.
 *
 * @tparam T The type of elements stored in the vector
 * @tparam MaxSize The maximum number of elements that can be stored
 * @tparam Copies Number of redundant copies to maintain (default: 2)
 * @tparam Alignment Custom alignment value (defaults to alignof(T))
 */
template <typename T, std::size_t MaxSize, std::size_t Copies = 2,
          std::size_t Alignment = alignof(T)>
class RadiationHardenedVector {
    static_assert(Copies >= 2, "Must maintain at least 2 copies for radiation hardening");

   public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = value_type&;
    using const_reference = const value_type&;

    /**
     * @brief Constructor with custom CRC function
     * @param crc_func Custom CRC calculation function
     */
    explicit RadiationHardenedVector(CRCFunction<T> crc_func = nullptr)
        : crc_func_(crc_func ? crc_func : default_crc)
    {
        update_crc();
    }

    // Element access with automatic validation
    reference operator[](size_type index)
    {
        validate_and_correct();
        return primary_[index];
    }

    const_reference operator[](size_type index) const
    {
        validate_and_correct();
        return primary_[index];
    }

    // Modifiers that maintain redundancy
    bool push_back(const T& value)
    {
        if (primary_.full()) {
            return false;
        }

        // Update all copies
        for (auto& copy : backup_copies_) {
            if (!copy.push_back(value)) {
                return false;
            }
        }

        bool result = primary_.push_back(value);
        update_crc();
        return result;
    }

    bool push_back(T&& value)
    {
        if (primary_.full()) {
            return false;
        }

        // Update all copies
        T temp = value;  // Create copy for backups since value will be moved
        for (auto& copy : backup_copies_) {
            if (!copy.push_back(temp)) {
                return false;
            }
        }

        bool result = primary_.push_back(std::move(value));
        update_crc();
        return result;
    }

    template <typename... Args>
    bool emplace_back(Args&&... args)
    {
        if (primary_.full()) {
            return false;
        }

        // Construct temporary to copy to backups
        T temp(std::forward<Args>(args)...);

        // Update all copies
        for (auto& copy : backup_copies_) {
            if (!copy.push_back(temp)) {
                return false;
            }
        }

        bool result = primary_.emplace_back(std::forward<Args>(args)...);
        update_crc();
        return result;
    }

    void pop_back()
    {
        primary_.pop_back();
        for (auto& copy : backup_copies_) {
            copy.pop_back();
        }
        update_crc();
    }

    void clear()
    {
        primary_.clear();
        for (auto& copy : backup_copies_) {
            copy.clear();
        }
        update_crc();
    }

    // Capacity
    [[nodiscard]] size_type size() const noexcept { return primary_.size(); }
    [[nodiscard]] constexpr size_type max_size() const noexcept { return MaxSize; }
    [[nodiscard]] bool empty() const noexcept { return primary_.empty(); }
    [[nodiscard]] bool full() const noexcept { return primary_.full(); }

    // Validation and correction
    /**
     * @brief Validates data integrity and corrects any detected errors
     * @return True if data was valid or successfully corrected
     */
    bool validate_and_correct()
    {
        // Check CRC first for quick validation
        if (validate_crc()) {
            return true;
        }

        ++error_count_;

        // CRC failed, perform full comparison of copies
        std::size_t valid_copy_index = find_valid_copy();
        if (valid_copy_index < Copies) {
            // Valid copy found, restore from it
            restore_from_backup(valid_copy_index);
            update_crc();
            return true;
        }

        return false;  // No valid copy found
    }

    // Telemetry
    [[nodiscard]] std::size_t get_error_count() const noexcept { return error_count_; }

    [[nodiscard]] std::size_t get_correction_count() const noexcept { return correction_count_; }

    void reset_telemetry() noexcept
    {
        error_count_ = 0;
        correction_count_ = 0;
    }

   private:
    FixedVector<T, MaxSize, Alignment> primary_;
    std::array<FixedVector<T, MaxSize, Alignment>, Copies - 1> backup_copies_;
    CRCFunction<T> crc_func_;
    std::uint32_t stored_crc_{0};
    std::size_t error_count_{0};
    std::size_t correction_count_{0};

    // Default CRC implementation
    static std::uint32_t default_crc(const T* data, std::size_t size)
    {
        std::uint32_t crc = 0xFFFFFFFF;
        const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(data);
        std::size_t byte_size = size * sizeof(T);

        for (std::size_t i = 0; i < byte_size; ++i) {
            crc ^= bytes[i];
            for (int j = 0; j < 8; ++j) {
                crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            }
        }

        return ~crc;
    }

    void update_crc() { stored_crc_ = crc_func_(primary_.begin(), primary_.size()); }

    bool validate_crc() const
    {
        return stored_crc_ == crc_func_(primary_.begin(), primary_.size());
    }

    std::size_t find_valid_copy() const
    {
        for (std::size_t i = 0; i < backup_copies_.size(); ++i) {
            const auto& copy = backup_copies_[i];
            if (copy.size() == primary_.size()) {
                bool match = true;
                for (std::size_t j = 0; j < copy.size(); ++j) {
                    if (std::memcmp(&copy.unsafe_access(j), &primary_.unsafe_access(j),
                                    sizeof(T)) != 0) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    return i;
                }
            }
        }
        return Copies;  // No valid copy found
    }

    void restore_from_backup(std::size_t valid_copy_index)
    {
        ++correction_count_;
        const auto& valid_copy = backup_copies_[valid_copy_index];

        // Restore primary
        primary_.clear();
        for (std::size_t i = 0; i < valid_copy.size(); ++i) {
            primary_.push_back(valid_copy.unsafe_access(i));
        }

        // Restore other backups
        for (std::size_t i = 0; i < backup_copies_.size(); ++i) {
            if (i != valid_copy_index) {
                auto& copy = backup_copies_[i];
                copy.clear();
                for (std::size_t j = 0; j < valid_copy.size(); ++j) {
                    copy.push_back(valid_copy.unsafe_access(j));
                }
            }
        }
    }
};

}  // namespace space_containers
