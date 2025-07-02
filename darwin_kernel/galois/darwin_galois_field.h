#ifndef DARWIN_GALOIS_FIELD_H
#define DARWIN_GALOIS_FIELD_H

#include <stdint.h>

// GF(256) multiplication using lookup tables (from your existing code)
static inline uint8_t darwin_gf256_multiply(uint8_t a, uint8_t b) {
    // Simplified version - you'll replace with your actual lookup tables
    if (a == 0 || b == 0) return 0;
    // Your existing GF multiplication logic goes here
    return a ^ b;  // Placeholder
}

static inline uint8_t darwin_gf256_add(uint8_t a, uint8_t b) {
    return a ^ b;
}

#endif
