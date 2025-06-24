#include <immintrin.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

/**
 * @brief SIMD-optimized matrix multiplication with bounds checking
 * This is the fixed version from cpu_optimized_training.cpp
 */
void simdMatrixMultiply(const std::vector<float>& a, const std::vector<std::vector<float>>& b,
                        std::vector<float>& result)
{
    // Safety checks for matrix dimensions
    if (a.empty() || b.empty() || result.empty()) {
        return;  // Handle empty matrices gracefully
    }

    // Ensure matrix dimensions are compatible
    if (b.size() != a.size()) {
        throw std::invalid_argument("Matrix dimension mismatch: b.size() != a.size()");
    }

    if (!b.empty() && b[0].size() != result.size()) {
        throw std::invalid_argument("Matrix dimension mismatch: b[0].size() != result.size()");
    }

    const size_t result_size = result.size();
    const size_t simd_size = 8;  // AVX processes 8 floats at once
    const size_t vectorized_end = (result_size / simd_size) * simd_size;

    // Vectorized computation using AVX (8 floats at once)
    for (size_t i = 0; i < vectorized_end; i += simd_size) {
        __m256 sum = _mm256_setzero_ps();

        for (size_t j = 0; j < a.size(); j++) {
            __m256 a_vec = _mm256_broadcast_ss(&a[j]);
            __m256 b_vec = _mm256_loadu_ps(&b[j][i]);
            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        }

        _mm256_storeu_ps(&result[i], sum);
    }

    // Handle remaining elements with scalar computation
    for (size_t i = vectorized_end; i < result_size; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < a.size(); j++) {
            sum += a[j] * b[j][i];
        }
        result[i] = sum;
    }
}

/**
 * @brief Reference scalar matrix multiplication for validation
 */
void scalarMatrixMultiply(const std::vector<float>& a, const std::vector<std::vector<float>>& b,
                          std::vector<float>& result)
{
    for (size_t i = 0; i < result.size(); ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < a.size(); j++) {
            sum += a[j] * b[j][i];
        }
        result[i] = sum;
    }
}

/**
 * @brief Test SIMD bounds checking with various sizes
 */
class SIMDBoundsTest {
   public:
    bool runTests()
    {
        std::cout << "🔬 SIMD Matrix Multiplication Bounds Test\n";
        std::cout << "=========================================\n\n";

        bool all_passed = true;

        // Test 1: Size exactly divisible by 8
        std::cout << "📊 Test 1: Size divisible by 8 (size=16)\n";
        all_passed &= testSize(16, 4);

        // Test 2: Size not divisible by 8 (remainder 1)
        std::cout << "\n📊 Test 2: Size with remainder 1 (size=17)\n";
        all_passed &= testSize(17, 4);

        // Test 3: Size not divisible by 8 (remainder 3)
        std::cout << "\n📊 Test 3: Size with remainder 3 (size=19)\n";
        all_passed &= testSize(19, 4);

        // Test 4: Size smaller than SIMD width
        std::cout << "\n📊 Test 4: Size smaller than SIMD (size=5)\n";
        all_passed &= testSize(5, 3);

        // Test 5: Size exactly SIMD width
        std::cout << "\n📊 Test 5: Size exactly SIMD width (size=8)\n";
        all_passed &= testSize(8, 3);

        // Test 6: Large size with remainder
        std::cout << "\n📊 Test 6: Large size with remainder (size=1000)\n";
        all_passed &= testSize(1000, 10);

        // Test 7: Error handling
        std::cout << "\n📊 Test 7: Error handling (dimension mismatches)\n";
        all_passed &= testErrorHandling();

        // Final result
        std::cout << "\n" << std::string(50, '=') << "\n";
        if (all_passed) {
            std::cout << "✅ ALL SIMD BOUNDS TESTS PASSED!\n";
            std::cout << "🚀 SIMD matrix multiplication is now safe!\n";
        }
        else {
            std::cout << "❌ SOME SIMD BOUNDS TESTS FAILED!\n";
            std::cout << "🔧 Please review the SIMD implementation.\n";
        }
        std::cout << std::string(50, '=') << "\n";

        return all_passed;
    }

   private:
    bool testSize(size_t result_size, size_t a_size)
    {
        // Create test matrices
        std::vector<float> a(a_size);
        std::vector<std::vector<float>> b(a_size, std::vector<float>(result_size));
        std::vector<float> result_simd(result_size);
        std::vector<float> result_scalar(result_size);

        // Fill with test data
        for (size_t i = 0; i < a_size; ++i) {
            a[i] = static_cast<float>(i + 1);
            for (size_t j = 0; j < result_size; ++j) {
                b[i][j] = static_cast<float>((i + 1) * (j + 1));
            }
        }

        try {
            // Compute with both methods
            simdMatrixMultiply(a, b, result_simd);
            scalarMatrixMultiply(a, b, result_scalar);

            // Compare results
            bool results_match = true;
            float max_error = 0.0f;

            for (size_t i = 0; i < result_size; ++i) {
                float error = std::abs(result_simd[i] - result_scalar[i]);
                max_error = std::max(max_error, error);

                if (error > 1e-5f) {  // Allow small floating-point errors
                    results_match = false;
                    std::cout << "   ❌ Mismatch at index " << i << ": SIMD=" << result_simd[i]
                              << ", Scalar=" << result_scalar[i] << "\n";
                    break;
                }
            }

            if (results_match) {
                std::cout << "   ✅ Results match (max error: " << max_error << ")\n";
                std::cout << "   📏 Size: " << result_size << ", A size: " << a_size
                          << ", Vectorized: " << (result_size / 8) * 8
                          << ", Scalar: " << result_size % 8 << "\n";
                return true;
            }
            else {
                std::cout << "   ❌ Results don't match!\n";
                return false;
            }
        }
        catch (const std::exception& e) {
            std::cout << "   ❌ Exception: " << e.what() << "\n";
            return false;
        }
    }

    bool testErrorHandling()
    {
        bool all_passed = true;

        // Test dimension mismatch: b.size() != a.size()
        try {
            std::vector<float> a(3);
            std::vector<std::vector<float>> b(4, std::vector<float>(5));  // Wrong size
            std::vector<float> result(5);

            simdMatrixMultiply(a, b, result);
            std::cout << "   ❌ Should have thrown dimension mismatch exception\n";
            all_passed = false;
        }
        catch (const std::invalid_argument& e) {
            std::cout << "   ✅ Correctly caught dimension mismatch: " << e.what() << "\n";
        }

        // Test dimension mismatch: b[0].size() != result.size()
        try {
            std::vector<float> a(3);
            std::vector<std::vector<float>> b(3, std::vector<float>(4));
            std::vector<float> result(5);  // Wrong size

            simdMatrixMultiply(a, b, result);
            std::cout << "   ❌ Should have thrown result size mismatch exception\n";
            all_passed = false;
        }
        catch (const std::invalid_argument& e) {
            std::cout << "   ✅ Correctly caught result size mismatch: " << e.what() << "\n";
        }

        // Test empty matrices (should handle gracefully)
        try {
            std::vector<float> a;
            std::vector<std::vector<float>> b;
            std::vector<float> result;

            simdMatrixMultiply(a, b, result);
            std::cout << "   ✅ Empty matrices handled gracefully\n";
        }
        catch (const std::exception& e) {
            std::cout << "   ❌ Empty matrices caused exception: " << e.what() << "\n";
            all_passed = false;
        }

        return all_passed;
    }
};

int main()
{
    SIMDBoundsTest test;
    bool success = test.runTests();
    return success ? 0 : 1;
}
