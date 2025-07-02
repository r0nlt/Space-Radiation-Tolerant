/*
 * Darwin RadML Foundation – Realistic Micro-Benchmark
 * Measures:
 *   1) Branchless TMR voting
 *   2) GF-256 multiplication
 *   3) 16.16 fixed-point multiply
 *
 * Uses high-resolution mach_absolute_time on macOS and randomised
 * volatile operands to prevent compiler optimisation.
 */

#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "darwin_kernel/darwin_radml_real.h"

/* convert Δticks → nanoseconds */
static inline double ticks_to_ns(uint64_t dt)
{
    static mach_timebase_info_data_t tb = {0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)dt * tb.numer / tb.denom;
}

#define ITERS 1000000
#define RAND32() ((uint32_t)rand() ^ ((uint32_t)rand() << 16))
#define RAND8() ((uint8_t)(rand() & 0xFF))

/* generic timing macro */
#define TIME_BLOCK(NS_PER_OP, ...)                            \
    do {                                                      \
        uint64_t start = mach_absolute_time();                \
        for (int _i = 0; _i < ITERS; ++_i) {                  \
            __VA_ARGS__                                       \
        }                                                     \
        uint64_t end = mach_absolute_time();                  \
        NS_PER_OP = ticks_to_ns(end - start) / (double)ITERS; \
    } while (0)

/* --- new helper arrays for operands --- */
static uint32_t a_tmr[ITERS], b_tmr[ITERS], c_tmr[ITERS];
static uint8_t a_gf[ITERS], b_gf[ITERS];
static darwin_fixed16_16_t a_fix[ITERS], b_fix[ITERS];

static void prepare_operands(void)
{
    for (int i = 0; i < ITERS; ++i) {
        a_tmr[i] = RAND32();
        b_tmr[i] = RAND32();
        c_tmr[i] = RAND32();

        a_gf[i] = RAND8();
        b_gf[i] = RAND8();

        float fa = (float)rand() / RAND_MAX;
        float fb = (float)rand() / RAND_MAX;
        a_fix[i] = darwin_fixed_from_float(fa);
        b_fix[i] = darwin_fixed_from_float(fb);
    }
}

int main(void)
{
    srand(42);
    prepare_operands();

    double ns_tmr, ns_gf, ns_fix;

    TIME_BLOCK(ns_tmr, {
        volatile uint32_t r = darwin_tmr_vote_optimized(a_tmr[_i], b_tmr[_i], c_tmr[_i]);
        (void)r;
    });

    TIME_BLOCK(ns_gf, {
        volatile uint8_t r = darwin_gf256_multiply(a_gf[_i], b_gf[_i]);
        (void)r;
    });

    TIME_BLOCK(ns_fix, {
        volatile darwin_fixed16_16_t r = darwin_fixed_multiply(a_fix[_i], b_fix[_i]);
        (void)r;
    });

    puts("🍎 Darwin RadML Foundation – Realistic Benchmark (isolated maths)");
    puts("===============================================================");
    printf("Branchless TMR voting : %.3f ns/op\n", ns_tmr);
    printf("GF-256 multiply       : %.3f ns/op\n", ns_gf);
    printf("Fixed-point multiply  : %.3f ns/op\n", ns_fix);
    return 0;
}
