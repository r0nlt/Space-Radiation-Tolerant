#include <iostream>

// Test each include separately
#include "include/rad_ml/math/branchless_ops.hpp"

int main() {
    std::cout << "Testing branchless ops...\n";
    
    rad_ml::math::BranchlessOps ops;
    uint32_t result = ops.min(42u, 37u);
    
    std::cout << "Min(42, 37) = " << result << "\n";
    std::cout << "✅ Branchless ops work!\n";
    
    return 0;
}
