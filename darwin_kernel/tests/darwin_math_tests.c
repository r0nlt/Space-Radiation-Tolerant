#include <stdio.h>
#include <assert.h>
#include "../galois/darwin_galois_field.h"
#include "../branchless/darwin_branchless_ops.h"

int main() {
    printf("🧪 Darwin Kernel Math Tests\n");
    printf("===========================\n");
    
    // Test branchless operations
    uint32_t a = 42, b = 37, c = 42;
    uint32_t min_val = darwin_branchless_min_u32(a, b);
    uint32_t vote_result = darwin_tmr_vote_u32(a, b, c);
    
    printf("Min(%u, %u) = %u\n", a, b, min_val);
    printf("TMR Vote(%u, %u, %u) = %u\n", a, b, c, vote_result);
    
    assert(min_val == 37);
    assert(vote_result == 42);
    
    printf("✅ All tests passed!\n");
    return 0;
}
