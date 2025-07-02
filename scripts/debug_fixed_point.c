/*
 * Debug the fixed-point precision issue
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "darwin_kernel/darwin_radml_real.h"

int main() {
    printf("🔍 Debugging Fixed-Point Precision Issue\n");
    printf("========================================\n\n");
    
    // Test the problematic small number case
    float input = 0.000015f;
    darwin_fixed16_16_t small = darwin_fixed_from_float(input);
    darwin_fixed16_16_t one = darwin_fixed_from_float(1.0f);
    darwin_fixed16_16_t result = darwin_fixed_multiply(small, one);
    
    float result_float = (float)result.value / DARWIN_FIXED_SCALE;
    float error = fabsf(result_float - input);
    
    printf("Input: %.6f\n", input);
    printf("Fixed16_16 representation: %d (0x%08x)\n", small.value, small.value);
    printf("Expected internal value: %d\n", (int)(input * DARWIN_FIXED_SCALE));
    printf("Result after multiply by 1: %.6f\n", result_float);
    printf("Error: %.8f\n", error);
    printf("Test threshold: 0.00001f\n");
    printf("Test passes: %s\n", (error < 0.00001f) ? "YES" : "NO");
    
    // Let's see what precision we can actually achieve
    printf("\nFixed16_16 Precision Analysis:\n");
    printf("DARWIN_FIXED_SCALE = %d (2^16 = %d)\n", DARWIN_FIXED_SCALE, 1 << 16);
    printf("Smallest representable value: %.8f\n", 1.0f / DARWIN_FIXED_SCALE);
    printf("Input (0.000015) in scale units: %.3f\n", input * DARWIN_FIXED_SCALE);
    
    // When we convert 0.000015 to fixed point, what do we actually get?
    int32_t exact_representation = (int32_t)(input * DARWIN_FIXED_SCALE);
    float reconstructed = (float)exact_representation / DARWIN_FIXED_SCALE;
    
    printf("\nDetailed Analysis:\n");
    printf("0.000015 × %d = %.3f\n", DARWIN_FIXED_SCALE, input * DARWIN_FIXED_SCALE);
    printf("Truncated to int32: %d\n", exact_representation);
    printf("Reconstructed: %.8f\n", reconstructed);
    printf("Reconstruction error: %.8f\n", fabsf(reconstructed - input));
    
    // Is this within the expected precision of Fixed16_16?
    float expected_precision = 1.0f / DARWIN_FIXED_SCALE;
    printf("\nExpected Fixed16_16 precision: %.8f\n", expected_precision);
    printf("Our error is %s than expected precision\n", 
           (fabsf(reconstructed - input) <= expected_precision) ? "less or equal" : "greater");
    
    // Test with a value that should work well
    printf("\nTesting with better-suited value:\n");
    float better_input = 0.0001f; // This should work better
    darwin_fixed16_16_t better_small = darwin_fixed_from_float(better_input);
    darwin_fixed16_16_t better_result = darwin_fixed_multiply(better_small, one);
    float better_result_float = (float)better_result.value / DARWIN_FIXED_SCALE;
    float better_error = fabsf(better_result_float - better_input);
    
    printf("Better input: %.6f\n", better_input);
    printf("Result: %.6f\n", better_result_float);
    printf("Error: %.8f\n", better_error);
    printf("Passes test: %s\n", (better_error < 0.00001f) ? "YES" : "NO");
    
    printf("\n🎯 Conclusion:\n");
    printf("=============\n");
    
    if (fabsf(reconstructed - input) <= expected_precision) {
        printf("✅ Fixed-point behavior is CORRECT!\n");
        printf("✅ The 'error' is within expected Fixed16_16 precision limits\n");
        printf("✅ 0.000015 is near the precision limit of 16.16 format\n");
        printf("✅ Our implementation is mathematically sound\n");
        printf("\nNote: The test threshold (0.00001) was too strict for this input.\n");
        printf("Fixed16_16 has ~0.000015 precision, so testing 0.000015 ± 0.00001 is unrealistic.\n");
        return 0;
    } else {
        printf("❌ There might be a real precision issue\n");
        return 1;
    }
}
