/**
 * @file optimized_allocator.hpp
 * @brief Optimized memory allocator with cache-aware allocation and SIMD support
 *
 * This file provides optimized memory allocation strategies for radiation-tolerant
 * machine learning applications, including cache-optimized allocation patterns
 * and SIMD-friendly memory layouts.
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace rad_ml {
namespace memory {

/**
 * @brief Cache-optimized memory allocator
 *
 * Allocates memory in cache-line aligned blocks to minimize cache misses
 * and improve performance in memory-intensive operations.
 */
template <typename T, size_t CacheLineSize = 64>
class CacheOptimizedAllocator {
   public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    CacheOptimizedAllocator() = default;

    template <typename U>
    CacheOptimizedAllocator(const CacheOptimizedAllocator<U>&)
    {
    }

    /**
     * @brief Allocate cache-aligned memory
     */
    pointer allocate(size_type n)
    {
        if (n > max_size()) {
            throw std::bad_alloc();
        }

        // Calculate padded size for cache alignment
        size_type padded_size = calculate_padded_size(n * sizeof(T));

        // Allocate memory with cache alignment
        void* ptr = aligned_alloc(CacheLineSize, padded_size);
        if (!ptr) {
            throw std::bad_alloc();
        }

        // Initialize memory to avoid uninitialized data issues
        std::fill_n(static_cast<T*>(ptr), n, T{});

        return static_cast<pointer>(ptr);
    }

    /**
     * @brief Deallocate memory
     */
    void deallocate(pointer p, size_type n) { free(p); }

    /**
     * @brief Construct object at given location
     */
    template <typename... Args>
    void construct(pointer p, Args&&... args)
    {
        new (p) T(std::forward<Args>(args)...);
    }

    /**
     * @brief Destroy object at given location
     */
    void destroy(pointer p) { p->~T(); }

    /**
     * @brief Maximum allocatable size
     */
    size_type max_size() const { return std::numeric_limits<size_type>::max() / sizeof(T); }

   private:
    /**
     * @brief Calculate padded size for cache alignment
     */
    size_type calculate_padded_size(size_type requested_size) const
    {
        size_type alignment = CacheLineSize;
        return (requested_size + alignment - 1) & ~(alignment - 1);
    }
};

/**
 * @brief SIMD-optimized matrix allocator
 *
 * Allocates memory in a layout optimized for SIMD operations,
 * ensuring proper alignment for vectorized computations.
 */
template <typename T, size_t VectorSize = 8>
class SIMDMatrixAllocator {
   public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    /**
     * @brief Allocate SIMD-optimized memory for matrices
     */
    static pointer allocate_matrix(size_type rows, size_type cols)
    {
        size_type total_elements = rows * cols;
        size_type alignment = VectorSize * sizeof(T);

        // Ensure rows are multiples of vector size for optimal SIMD performance
        size_type padded_rows = (rows + VectorSize - 1) & ~(VectorSize - 1);

        void* ptr = aligned_alloc(alignment, padded_rows * cols * sizeof(T));
        if (!ptr) {
            throw std::bad_alloc();
        }

        return static_cast<pointer>(ptr);
    }

    /**
     * @brief Optimized matrix multiplication with SIMD
     */
    static void matrix_multiply(const T* A, const T* B, T* C, size_type rows_A, size_type cols_A,
                                size_type cols_B)
    {
#ifdef __AVX2__
        if constexpr (std::is_same_v<T, float>) {
            matrix_multiply_avx(A, B, C, rows_A, cols_A, cols_B);
        }
        else {
            matrix_multiply_fallback(A, B, C, rows_A, cols_A, cols_B);
        }
#else
        matrix_multiply_fallback(A, B, C, rows_A, cols_A, cols_B);
#endif
    }

   private:
#ifdef __AVX2__
    /**
     * @brief AVX2-optimized matrix multiplication
     */
    static void matrix_multiply_avx(const float* A, const float* B, float* C, size_type rows_A,
                                    size_type cols_A, size_type cols_B)
    {
        const size_type vector_size = 8;

        for (size_type i = 0; i < rows_A; ++i) {
            for (size_type j = 0; j < cols_B; ++j) {
                __m256 sum = _mm256_setzero_ps();

                for (size_type k = 0; k < cols_A; k += vector_size) {
                    __m256 a_vec = _mm256_loadu_ps(&A[i * cols_A + k]);
                    __m256 b_vec = _mm256_loadu_ps(&B[k * cols_B + j]);

                    sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                }

                // Store result
                _mm256_storeu_ps(&C[i * cols_B + j], sum);
            }
        }
    }
#endif

    /**
     * @brief Fallback matrix multiplication
     */
    static void matrix_multiply_fallback(const T* A, const T* B, T* C, size_type rows_A,
                                         size_type cols_A, size_type cols_B)
    {
        for (size_type i = 0; i < rows_A; ++i) {
            for (size_type j = 0; j < cols_B; ++j) {
                T sum = T(0);
                for (size_type k = 0; k < cols_A; ++k) {
                    sum += A[i * cols_A + k] * B[k * cols_B + j];
                }
                C[i * cols_B + j] = sum;
            }
        }
    }
};

/**
 * @brief Memory pool for frequently allocated small objects
 *
 * Reduces allocation overhead for small, frequently allocated objects
 */
template <typename T, std::size_t PoolSize = 1024>
class SmallObjectPool {
   public:
    using size_type = std::size_t;
    SmallObjectPool() : pool_(), free_list_()
    {
        // Pre-allocate memory blocks
        for (size_type i = 0; i < PoolSize; ++i) {
            free_list_.push_back(&pool_[i]);
        }
    }

    /**
     * @brief Allocate object from pool
     */
    T* allocate()
    {
        if (free_list_.empty()) {
            throw std::bad_alloc();
        }

        T* ptr = free_list_.back();
        free_list_.pop_back();
        return ptr;
    }

    /**
     * @brief Deallocate object to pool
     */
    void deallocate(T* ptr) { free_list_.push_back(ptr); }

    /**
     * @brief Get number of free objects in pool
     */
    size_type free_count() const { return free_list_.size(); }

   private:
    std::array<T, PoolSize> pool_;
    std::vector<T*> free_list_;
};

/**
 * @brief Thread-safe memory pool with lock-free operations
 */
template <typename T, std::size_t PoolSize = 1024>
class ThreadSafeMemoryPool {
   public:
    ThreadSafeMemoryPool() = default;

    /**
     * @brief Allocate object (thread-safe)
     */
    T* allocate()
    {
        T* ptr = nullptr;

        // Try to get from free list first
        if (free_list_.pop(ptr)) {
            return ptr;
        }

        // Allocate new memory if pool is exhausted
        return new T();
    }

    /**
     * @brief Deallocate object (thread-safe)
     */
    void deallocate(T* ptr)
    {
        if (!ptr) return;

        free_list_.push(ptr);
    }

   private:
    struct alignas(64) FreeList {
        struct Node {
            T* ptr;
            Node* next;
        };

        std::atomic<Node*> head{nullptr};
        std::atomic<size_t> size{0};

        bool pop(T*& result)
        {
            Node* current = head.load(std::memory_order_acquire);
            if (!current) return false;

            while (!head.compare_exchange_weak(current, current ? current->next : nullptr,
                                               std::memory_order_acquire)) {
                if (!current) return false;
            }

            result = current->ptr;
            delete current;
            size.fetch_sub(1, std::memory_order_relaxed);
            return true;
        }

        void push(T* ptr)
        {
            Node* node = new Node{ptr, nullptr};
            Node* current = head.load(std::memory_order_acquire);
            do {
                node->next = current;
            } while (!head.compare_exchange_weak(current, node, std::memory_order_release));

            size.fetch_add(1, std::memory_order_relaxed);
        }
    };

    FreeList free_list_;
};

}  // namespace memory
}  // namespace rad_ml
