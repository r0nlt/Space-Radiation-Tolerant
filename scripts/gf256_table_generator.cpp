/*
 * GF(256) Lookup Table Generator - Modern C++ Standards
 * Extracts tables for Darwin kernel integration
 */

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>

class GF256TableGenerator {
   private:
    static constexpr uint16_t PRIMITIVE_POLY = 0x11d;  // x^8 + x^4 + x^3 + x^2 + 1
    static constexpr size_t FIELD_SIZE = 256;

    std::array<uint8_t, FIELD_SIZE> exp_table_{};
    std::array<uint8_t, FIELD_SIZE> log_table_{};

   public:
    GF256TableGenerator() { generate_tables(); }

    void generate_tables()
    {
        std::cout << "🔢 Generating GF(256) lookup tables...\n";

        // Generate exp table: exp_table[i] = α^i
        uint16_t x = 1;

        for (size_t i = 0; i < FIELD_SIZE - 1; ++i) {
            exp_table_[i] = static_cast<uint8_t>(x);

            // Multiply by α (primitive element = 2)
            x <<= 1;
            if (x & 0x100) {  // If overflow occurred
                x ^= PRIMITIVE_POLY;
            }
        }
        exp_table_[FIELD_SIZE - 1] = exp_table_[0];  // Wrap around

        // Generate log table: log_table[α^i] = i
        log_table_[0] = 0;  // log(0) undefined, set to 0
        for (size_t i = 0; i < FIELD_SIZE - 1; ++i) {
            log_table_[exp_table_[i]] = static_cast<uint8_t>(i);
        }

        std::cout << "✅ Tables generated successfully!\n";
    }

    void validate_tables() const
    {
        std::cout << "\n🧪 Validating GF(256) properties...\n";

        // Test basic properties
        bool valid = true;

        // Test that exp[log[x]] = x for non-zero x
        for (uint16_t x = 1; x < FIELD_SIZE; ++x) {
            if (exp_table_[log_table_[x]] != x) {
                std::cout << "❌ Validation failed at x=" << x << "\n";
                valid = false;
                break;
            }
        }

        // Test multiplication via lookup
        uint8_t test_a = 0x53, test_b = 0xCA;
        uint8_t mult_result = (test_a == 0 || test_b == 0)
                                  ? 0
                                  : exp_table_[(log_table_[test_a] + log_table_[test_b]) % 255];

        std::cout << "  Sample multiplication: 0x" << std::hex << (int)test_a << " × 0x"
                  << (int)test_b << " = 0x" << (int)mult_result << std::dec << "\n";

        if (valid) {
            std::cout << "✅ All validations passed!\n";
        }
    }

    void generate_darwin_header() const
    {
        std::cout << "\n🔧 Generating Darwin kernel header...\n";

        std::ofstream out("darwin_kernel/gf256_tables.h");

        out << "/*\n";
        out << " * GF(256) Lookup Tables for Darwin Kernel\n";
        out << " * Generated using modern C++ standards\n";
        out << " * Polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11d)\n";
        out << " */\n\n";
        out << "#ifndef DARWIN_GF256_TABLES_H\n";
        out << "#define DARWIN_GF256_TABLES_H\n\n";

        // Generate exp table
        out << "/* Exponential table: exp_table[i] = α^i */\n";
        out << "static const uint8_t darwin_gf256_exp_table[256] = {\n";
        for (size_t i = 0; i < FIELD_SIZE; i += 16) {
            out << "    ";
            for (size_t j = 0; j < 16 && i + j < FIELD_SIZE; ++j) {
                out << "0x" << std::hex << std::setfill('0') << std::setw(2)
                    << (int)exp_table_[i + j];
                if (i + j < FIELD_SIZE - 1) out << ",";
                if (j < 15 && i + j < FIELD_SIZE - 1) out << " ";
            }
            out << std::dec << "\n";
        }
        out << "};\n\n";

        // Generate log table
        out << "/* Logarithm table: log_table[α^i] = i */\n";
        out << "static const uint8_t darwin_gf256_log_table[256] = {\n";
        for (size_t i = 0; i < FIELD_SIZE; i += 16) {
            out << "    ";
            for (size_t j = 0; j < 16 && i + j < FIELD_SIZE; ++j) {
                out << "0x" << std::hex << std::setfill('0') << std::setw(2)
                    << (int)log_table_[i + j];
                if (i + j < FIELD_SIZE - 1) out << ",";
                if (j < 15 && i + j < FIELD_SIZE - 1) out << " ";
            }
            out << std::dec << "\n";
        }
        out << "};\n\n";

        out << "#endif\n";
        out.close();

        std::cout << "✅ Generated darwin_kernel/gf256_tables.h\n";
    }

    void print_tables_for_insertion() const
    {
        std::cout << "\n📋 Tables ready for darwin_radml_real.h insertion:\n";
        std::cout << "================================================\n\n";

        // Print exp table for copy-paste
        std::cout << "/* Exponential table: exp_table[i] = α^i */\n";
        std::cout << "static const uint8_t darwin_gf256_exp_table[256] = {\n";
        for (size_t i = 0; i < FIELD_SIZE; i += 16) {
            std::cout << "    ";
            for (size_t j = 0; j < 16 && i + j < FIELD_SIZE; ++j) {
                std::cout << "0x" << std::hex << std::setfill('0') << std::setw(2)
                          << (int)exp_table_[i + j];
                if (i + j < FIELD_SIZE - 1) std::cout << ",";
                if (j < 15 && i + j < FIELD_SIZE - 1) std::cout << " ";
            }
            std::cout << std::dec << "\n";
        }
        std::cout << "};\n\n";

        // Print log table for copy-paste
        std::cout << "/* Logarithm table: log_table[α^i] = i */\n";
        std::cout << "static const uint8_t darwin_gf256_log_table[256] = {\n";
        for (size_t i = 0; i < FIELD_SIZE; i += 16) {
            std::cout << "    ";
            for (size_t j = 0; j < 16 && i + j < FIELD_SIZE; ++j) {
                std::cout << "0x" << std::hex << std::setfill('0') << std::setw(2)
                          << (int)log_table_[i + j];
                if (i + j < FIELD_SIZE - 1) std::cout << ",";
                if (j < 15 && i + j < FIELD_SIZE - 1) std::cout << " ";
            }
            std::cout << std::dec << "\n";
        }
        std::cout << "};\n\n";
    }
};

int main()
{
    std::cout << "🍎 Darwin GF(256) Table Generator\n";
    std::cout << "Modern C++ Implementation\n";
    std::cout << "=========================\n\n";

    try {
        GF256TableGenerator generator;
        generator.validate_tables();
        generator.generate_darwin_header();
        generator.print_tables_for_insertion();

        std::cout << "\n🎉 Ready to integrate into darwin_radml_real.h!\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
