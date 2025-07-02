#ifndef DARWIN_BRANCHLESS_OPS_H
#define DARWIN_BRANCHLESS_OPS_H

#include <stdint.h>

// From your existing include/rad_ml/math/branchless_ops.hpp
static inline uint32_t darwin_branchless_min_u32(uint32_t a, uint32_t b) {
    uint32_t mask = -(a <= b);
    return (mask & a) | (~mask & b);
}

static inline uint32_t darwin_branchless_max_u32(uint32_t a, uint32_t b) {
    uint32_t mask = -(a >= b);
    return (mask & a) | (~mask & b);
}

// TMR voting using your branchless technique
static inline uint32_t darwin_tmr_vote_u32(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t ab_match = -(a == b);
    uint32_t ac_match = -(a == c);
    return (ab_match & a) | ((~ab_match & ac_match) & a) | ((~ab_match & ~ac_match) & b);
}

#endif
