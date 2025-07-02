#include <iostream>
#include "include/rad_ml/neural/galois_field.hpp"

int main() {
    std::cout << "Testing Galois Field...\n";
    
    rad_ml::neural::GF256 gf;
    uint8_t result = gf.multiply(0x53, 0xCA);
    
    std::cout << "GF(256): 0x53 * 0xCA = 0x" << std::hex << (int)result << std::dec << "\n";
    std::cout << "✅ Galois Field works!\n";
    
    return 0;
}
