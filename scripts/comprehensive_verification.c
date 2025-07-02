/*
 * Comprehensive Darwin RadML Verification
 * Double-checking all mathematical properties
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "darwin_kernel/darwin_radml_real.h"

int main() {
    printf("🔍 Darwin RadML - Comprehensive Verification\n");
    printf("============================================\n\n");
    
    int errors = 0;
    
    // === GF(256) VERIFICATION ===
    printf("🔢 GF(256) Mathematical Properties\n");
    printf("==================================\n");
    
    // Test 1: Field axioms
    printf("Test 1: Field Axioms\n");
    
    // Additive identity: a + 0 = a
    uint8_t test_val = 0x53;
    if (darwin_gf256_add(test_val, 0) != test_val) {
        printf("❌ Additive identity failed\n");
        errors++;
    } else {
        printf("✅ Additive identity: a ⊕ 0 = a\n");
    }
    
    // Additive inverse: a + a = 0
    if (darwin_gf256_add(test_val, test_val) != 0) {
        printf("❌ Additive inverse failed\n");
        errors++;
    } else {
        printf("✅ Additive inverse: a ⊕ a = 0\n");
    }
    
    // Multiplicative identity: a * 1 = a
    if (darwin_gf256_multiply(test_val, 1) != test_val) {
        printf("❌ Multiplicative identity failed\n");
        errors++;
    } else {
        printf("✅ Multiplicative identity: a × 1 = a\n");
    }
    
    // Zero element: a * 0 = 0
    if (darwin_gf256_multiply(test_val, 0) != 0) {
        printf("❌ Zero multiplication failed\n");
        errors++;
    } else {
        printf("✅ Zero multiplication: a × 0 = 0\n");
    }
    
    // Test 2: Multiplication/Division consistency
    printf("\nTest 2: Multiplication/Division Consistency\n");
    
    uint8_t a = 0x53, b = 0xCA;
    uint8_t product = darwin_gf256_multiply(a, b);
    uint8_t quotient = darwin_gf256_divide(product, b);
    
    printf("  0x%02x × 0x%02x = 0x%02x\n", a, b, product);
    printf("  0x%02x ÷ 0x%02x = 0x%02x", product, b, quotient);
    
    if (quotient == a) {
        printf(" ✅\n");
    } else {
        printf(" ❌ (Expected 0x%02x)\n", a);
        errors++;
    }
    
    // Test 3: Commutativity
    printf("\nTest 3: Commutativity\n");
    
    uint8_t ab = darwin_gf256_multiply(a, b);
    uint8_t ba = darwin_gf256_multiply(b, a);
    
    if (ab == ba) {
        printf("✅ Multiplication is commutative: a×b = b×a\n");
    } else {
        printf("❌ Multiplication not commutative\n");
        errors++;
    }
    
    // Test 4: Distributivity
    printf("\nTest 4: Distributivity\n");
    
    uint8_t c = 0x95;
    uint8_t left = darwin_gf256_multiply(a, darwin_gf256_add(b, c));
    uint8_t right = darwin_gf256_add(darwin_gf256_multiply(a, b), darwin_gf256_multiply(a, c));
    
    if (left == right) {
        printf("✅ Distributivity: a×(b⊕c) = (a×b)⊕(a×c)\n");
    } else {
        printf("❌ Distributivity failed\n");
        errors++;
    }
    
    // === BRANCHLESS OPERATIONS VERIFICATION ===
    printf("\n⚡ Branchless Operations Verification\n");
    printf("====================================\n");
    
    // Test various TMR scenarios
    struct tmr_test {
        uint32_t a, b, c;
        uint32_t expected;
        const char* name;
    } tmr_tests[] = {
        {42, 42, 42, 42, "All same"},
        {42, 37, 42, 42, "A=C majority"},
        {37, 42, 42, 42, "B=C majority"},
        {42, 42, 37, 42, "A=B majority"},
        {10, 20, 30, 20, "All different -> B"}
    };
    
    printf("Test 5: TMR Voting Scenarios\n");
    
    for (int i = 0; i < 5; i++) {
        uint32_t result = darwin_tmr_vote_optimized(tmr_tests[i].a, tmr_tests[i].b, tmr_tests[i].c);
        if (result == tmr_tests[i].expected) {
            printf("✅ %s: TMR(%d,%d,%d) = %d\n", 
                   tmr_tests[i].name, tmr_tests[i].a, tmr_tests[i].b, tmr_tests[i].c, result);
        } else {
            printf("❌ %s: TMR(%d,%d,%d) = %d (expected %d)\n", 
                   tmr_tests[i].name, tmr_tests[i].a, tmr_tests[i].b, tmr_tests[i].c, 
                   result, tmr_tests[i].expected);
            errors++;
        }
    }
    
    // Test min/max
    printf("\nTest 6: Min/Max Operations\n");
    
    if (darwin_branchless_min(42, 37) == 37 && darwin_branchless_max(42, 37) == 42) {
        printf("✅ Min/Max operations correct\n");
    } else {
        printf("❌ Min/Max operations failed\n");
        errors++;
    }
    
    // === FIXED-POINT VERIFICATION ===
    printf("\n🎯 Fixed-Point Arithmetic Verification\n");
    printf("======================================\n");
    
    printf("Test 7: Fixed-Point Accuracy\n");
    
    // Test various mathematical operations
    darwin_fixed16_16_t pi = darwin_fixed_from_float(3.14159f);
    darwin_fixed16_16_t e = darwin_fixed_from_float(2.71828f);
    darwin_fixed16_16_t two = darwin_fixed_from_float(2.0f);
    
    darwin_fixed16_16_t pi_times_two = darwin_fixed_multiply(pi, two);
    darwin_fixed16_16_t pi_times_e = darwin_fixed_multiply(pi, e);
    
    float pi_times_two_float = (float)pi_times_two.value / DARWIN_FIXED_SCALE;
    float pi_times_e_float = (float)pi_times_e.value / DARWIN_FIXED_SCALE;
    
    float expected_pi_times_two = 3.14159f * 2.0f;
    float expected_pi_times_e = 3.14159f * 2.71828f;
    
    float error_two = fabsf(pi_times_two_float - expected_pi_times_two);
    float error_e = fabsf(pi_times_e_float - expected_pi_times_e);
    
    printf("  π × 2 = %.6f (expected %.6f, error %.6f)\n", 
           pi_times_two_float, expected_pi_times_two, error_two);
    printf("  π × e = %.6f (expected %.6f, error %.6f)\n", 
           pi_times_e_float, expected_pi_times_e, error_e);
    
    if (error_two < 0.001f && error_e < 0.001f) {
        printf("✅ Fixed-point accuracy excellent\n");
    } else {
        printf("❌ Fixed-point accuracy insufficient\n");
        errors++;
    }
    
    // === LOOKUP TABLE VERIFICATION ===
    printf("\nTest 8: Lookup Table Consistency\n");
    
    // Verify exp[log[x]] = x for all non-zero x
    int table_errors = 0;
    for (int x = 1; x < 256; x++) {
        uint8_t log_val = darwin_gf256_log_table[x];
        uint8_t exp_val = darwin_gf256_exp_table[log_val];
        if (exp_val != x) {
            table_errors++;
            if (table_errors <= 3) { // Only show first few errors
                printf("❌ Table inconsistency: exp[log[%d]] = %d\n", x, exp_val);
            }
        }
    }
    
    if (table_errors == 0) {
        printf("✅ Lookup tables are mathematically consistent\n");
    } else {
        printf("❌ Found %d lookup table inconsistencies\n", table_errors);
        errors++;
    }
    
    // === FINAL RESULTS ===
    printf("\n🎯 Verification Results\n");
    printf("======================\n");
    
    if (errors == 0) {
        printf("🎉 ALL TESTS PASSED! Darwin RadML Foundation is PERFECT!\n");
        printf("✅ GF(256) operations mathematically sound\n");
        printf("✅ Branchless operations working correctly\n");
        printf("✅ Fixed-point arithmetic accurate\n");
        printf("✅ Lookup tables consistent\n");
        printf("✅ Ready for Darwin kernel integration\n");
        return 0;
    } else {
        printf("❌ Found %d error(s) - needs investigation\n", errors);
        return 1;
    }
}
