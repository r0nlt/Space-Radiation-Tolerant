# Using Advanced Reed-Solomon Error Correction

```cpp
#include "rad_ml/neural/advanced_reed_solomon.hpp"

// Create Reed-Solomon codec with 8-bit symbols, 12 total symbols, 8 data symbols
neural::AdvancedReedSolomon<uint8_t, 8> rs_codec(12, 8);

// Encode a vector of data
std::vector<uint8_t> data = {1, 2, 3, 4, 5, 6, 7, 8};
auto encoded = rs_codec.encode(data);

// Simulate error (corrupt some data)
encoded[2] = 255;  // Corrupt a symbol

// Decode with error correction
auto decoded = rs_codec.decode(encoded);
if (decoded) {
    std::cout << "Successfully recovered data" << std::endl;
}
```
