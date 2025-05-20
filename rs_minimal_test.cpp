#include <array>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <vector>

class SimpleGF256 {
   public:
    using element_t = uint8_t;
    static constexpr uint16_t field_size = 256;

    SimpleGF256()
    {
        std::cout << "Initializing Galois Field tables..." << std::endl;
        for (int i = 0; i < 256; i++) {
            exp_table[i] = 0;
            log_table[i] = 0;
        }

        // Generate exponential table (simple version)
        uint8_t x = 1;
        for (int i = 0; i < 255; i++) {
            exp_table[i] = x;

            // Simplified multiplication by α in GF(2^8)
            uint16_t next = static_cast<uint16_t>(x) << 1;
            if (next & 0x100) {
                next ^= 0x11D;  // x^8 + x^4 + x^3 + x^2 + 1
            }
            x = next & 0xFF;
        }

        // Set the last element
        exp_table[255] = exp_table[0];

        // Generate log table
        log_table[0] = 0;  // log(0) is undefined
        for (int i = 0; i < 255; i++) {
            log_table[exp_table[i]] = i;
        }

        std::cout << "Galois Field tables initialized." << std::endl;
    }

    element_t add(element_t a, element_t b) const { return a ^ b; }

    element_t multiply(element_t a, element_t b) const
    {
        if (a == 0 || b == 0) return 0;

        return exp_table[(log_table[a] + log_table[b]) % 255];
    }

   private:
    std::array<element_t, field_size> exp_table;  // α^i lookup
    std::array<element_t, field_size> log_table;  // log_α(i) lookup
};

class SimpleReedSolomon {
   public:
    static constexpr size_t DATA_SIZE = 4;  // Size of float
    static constexpr size_t ECC_SIZE = 4;   // 4 ECC bytes
    static constexpr size_t TOTAL_SIZE = DATA_SIZE + ECC_SIZE;

    SimpleReedSolomon() : field_() { std::cout << "SimpleReedSolomon initialized." << std::endl; }

    std::vector<uint8_t> encode(float value) const
    {
        std::cout << "Encoding value: " << value << std::endl;

        // Convert float to bytes
        union {
            float f;
            uint8_t bytes[DATA_SIZE];
        } data;
        data.f = value;

        // Create the codeword (data + ecc)
        std::vector<uint8_t> codeword(TOTAL_SIZE, 0);

        // Copy data bytes
        for (size_t i = 0; i < DATA_SIZE; i++) {
            codeword[i] = data.bytes[i];
        }

        // Add simple checksum for ecc
        uint8_t checksum = 0;
        for (size_t i = 0; i < DATA_SIZE; i++) {
            checksum = field_.add(checksum, data.bytes[i]);
        }

        // Store checksum and some redundant data
        codeword[DATA_SIZE] = checksum;
        codeword[DATA_SIZE + 1] = ~checksum;  // Inverted checksum
        codeword[DATA_SIZE + 2] = data.bytes[0] ^ data.bytes[1];
        codeword[DATA_SIZE + 3] = data.bytes[2] ^ data.bytes[3];

        return codeword;
    }

    std::optional<float> decode(const std::vector<uint8_t>& codeword) const
    {
        std::cout << "Decoding codeword..." << std::endl;

        if (codeword.size() < TOTAL_SIZE) {
            std::cout << "Codeword too short" << std::endl;
            return std::nullopt;
        }

        // Check if the data is valid
        uint8_t checksum = 0;
        for (size_t i = 0; i < DATA_SIZE; i++) {
            checksum = field_.add(checksum, codeword[i]);
        }

        bool checksumValid = (checksum == codeword[DATA_SIZE]);
        bool invChecksumValid = (~checksum == codeword[DATA_SIZE + 1]);
        bool xorCheck1 = ((codeword[0] ^ codeword[1]) == codeword[DATA_SIZE + 2]);
        bool xorCheck2 = ((codeword[2] ^ codeword[3]) == codeword[DATA_SIZE + 3]);

        std::cout << "Checksum valid: " << checksumValid << std::endl;
        std::cout << "Inverted checksum valid: " << invChecksumValid << std::endl;
        std::cout << "XOR check 1 valid: " << xorCheck1 << std::endl;
        std::cout << "XOR check 2 valid: " << xorCheck2 << std::endl;

        // If all checks pass, data is likely good
        if (checksumValid && invChecksumValid && xorCheck1 && xorCheck2) {
            // Convert back to float
            union {
                float f;
                uint8_t bytes[DATA_SIZE];
            } data;

            for (size_t i = 0; i < DATA_SIZE; i++) {
                data.bytes[i] = codeword[i];
            }

            return data.f;
        }

        // Try basic error correction if possible
        if (!checksumValid && invChecksumValid) {
            std::cout << "Attempting to correct errors..." << std::endl;
            // This is an extremely simplified correction that only works for simple cases
            std::vector<uint8_t> corrected = codeword;

            // Just a simple example - not a real RS correction
            if (xorCheck1 && !xorCheck2) {
                // First two bytes are ok, last two might have an error
                uint8_t expected_xor = codeword[DATA_SIZE + 3];
                uint8_t actual_xor = codeword[2] ^ codeword[3];

                if (expected_xor != actual_xor) {
                    // Just a guess - set byte 3 to make XOR check pass
                    corrected[3] = corrected[2] ^ expected_xor;
                }
            }

            // Convert corrected data to float
            union {
                float f;
                uint8_t bytes[DATA_SIZE];
            } data;

            for (size_t i = 0; i < DATA_SIZE; i++) {
                data.bytes[i] = corrected[i];
            }

            return data.f;
        }

        return std::nullopt;
    }

    // Apply random bit errors for testing
    std::vector<uint8_t> apply_bit_errors(const std::vector<uint8_t>& data, double error_rate,
                                          uint64_t seed = 0) const
    {
        if (data.empty() || error_rate <= 0.0) return data;

        std::vector<uint8_t> result = data;
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // Apply bit errors based on probability
        for (size_t i = 0; i < result.size(); ++i) {
            for (int bit = 0; bit < 8; ++bit) {
                if (dist(rng) < error_rate) {
                    result[i] ^= (1 << bit);
                }
            }
        }

        return result;
    }

    // Get maximum number of errors that can be corrected
    size_t correction_capability() const
    {
        // This simple implementation can only correct 1 byte at most
        return 1;
    }

    // Calculate storage overhead
    double overhead_percent() const
    {
        return (static_cast<double>(TOTAL_SIZE) / DATA_SIZE - 1.0) * 100.0;
    }

   private:
    SimpleGF256 field_;
};

void print_hex(const std::vector<uint8_t>& data, const std::string& label)
{
    std::cout << label << ": ";
    for (uint8_t b : data) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b) << " ";
    }
    std::cout << std::dec << std::endl;
}

int main()
{
    std::cout << "Starting Reed-Solomon test with simplified implementation..." << std::endl;

    try {
        SimpleReedSolomon rs;

        // Original value
        float original_weight = 3.14159f;
        std::cout << "Original weight: " << original_weight << std::endl;

        // Encode
        auto encoded = rs.encode(original_weight);
        print_hex(encoded, "Encoded data");

        // Apply errors (10% bit error rate)
        double error_rate = 0.10;
        auto corrupted = rs.apply_bit_errors(encoded, error_rate, 42);
        print_hex(corrupted, "Corrupted data");

        // Count errors
        int bit_errors = 0;
        for (size_t i = 0; i < encoded.size(); i++) {
            uint8_t xor_result = encoded[i] ^ corrupted[i];
            bit_errors += std::bitset<8>(xor_result).count();
        }
        std::cout << "Number of bit errors: " << bit_errors << " out of " << (encoded.size() * 8)
                  << std::endl;

        // Decode
        auto decoded_opt = rs.decode(corrupted);

        if (decoded_opt) {
            float corrected_weight = *decoded_opt;
            std::cout << "Corrected weight: " << corrected_weight << std::endl;
            bool recovered = std::abs(corrected_weight - original_weight) < 1e-6;
            std::cout << "Original value recovered: " << (recovered ? "Yes" : "No") << std::endl;
        }
        else {
            std::cout << "Error correction failed: Unrecoverable data" << std::endl;
        }

        // Try with higher error rates
        std::cout << "\nTesting with increasing error rates:" << std::endl;
        std::vector<double> rates = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3};

        for (double rate : rates) {
            int success = 0;
            int trials = 10;

            for (int i = 0; i < trials; i++) {
                auto test_corrupted = rs.apply_bit_errors(encoded, rate, 42 + i);
                auto test_decoded = rs.decode(test_corrupted);

                if (test_decoded && std::abs(*test_decoded - original_weight) < 1e-6) {
                    success++;
                }
            }

            std::cout << "Error rate " << rate * 100 << "%: " << success << "/" << trials
                      << " successful corrections (" << (success * 100 / trials)
                      << "% success rate)" << std::endl;
        }

        // Print ECC information
        std::cout << "\nCorrection capability: up to " << rs.correction_capability()
                  << " byte errors" << std::endl;
        std::cout << "Storage overhead: " << rs.overhead_percent() << "%" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception" << std::endl;
    }

    return 0;
}
