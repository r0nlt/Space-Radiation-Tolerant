#include <gtest/gtest.h>

#include <cstring>
#include <space_containers/radiation_hardened_vector.hpp>
#include <string>

using namespace space_containers;

// Custom CRC function for testing
template <typename T>
std::uint32_t test_crc(const T* data, std::size_t size)
{
    std::uint32_t crc = 0;
    const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(data);
    std::size_t byte_size = size * sizeof(T);

    for (std::size_t i = 0; i < byte_size; ++i) {
        crc += bytes[i];
    }
    return crc;
}

TEST(RadiationHardenedVectorTest, BasicOperations)
{
    RadiationHardenedVector<int, 5> vec;

    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0);
    EXPECT_EQ(vec.max_size(), 5);

    EXPECT_TRUE(vec.push_back(1));
    EXPECT_TRUE(vec.push_back(2));
    EXPECT_TRUE(vec.push_back(3));

    EXPECT_EQ(vec.size(), 3);
    EXPECT_FALSE(vec.empty());

    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
    EXPECT_EQ(vec[2], 3);
}

TEST(RadiationHardenedVectorTest, CapacityLimits)
{
    RadiationHardenedVector<int, 3> vec;

    EXPECT_TRUE(vec.push_back(1));
    EXPECT_TRUE(vec.push_back(2));
    EXPECT_TRUE(vec.push_back(3));
    EXPECT_FALSE(vec.push_back(4));  // Should fail, vector is full

    EXPECT_TRUE(vec.full());
    EXPECT_EQ(vec.size(), 3);
}

TEST(RadiationHardenedVectorTest, MoveSemantics)
{
    RadiationHardenedVector<std::string, 3> vec;

    std::string str = "test";
    EXPECT_TRUE(vec.push_back(std::move(str)));
    EXPECT_TRUE(str.empty());  // Original string should be moved from

    EXPECT_EQ(vec[0], "test");
}

TEST(RadiationHardenedVectorTest, EmplaceBack)
{
    RadiationHardenedVector<std::string, 3> vec;

    EXPECT_TRUE(vec.emplace_back("test"));
    EXPECT_TRUE(vec.emplace_back(5, 'a'));  // Creates string "aaaaa"

    EXPECT_EQ(vec[0], "test");
    EXPECT_EQ(vec[1], "aaaaa");
}

TEST(RadiationHardenedVectorTest, CustomCRC)
{
    RadiationHardenedVector<int, 5> vec(test_crc<int>);

    EXPECT_TRUE(vec.push_back(1));
    EXPECT_TRUE(vec.push_back(2));

    // Data should be valid
    EXPECT_TRUE(vec.validate_and_correct());
}

// Helper function to corrupt memory
template <typename T>
void corrupt_memory(T* data)
{
    std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(data);
    bytes[0] ^= 0xFF;  // Flip all bits in first byte
}

TEST(RadiationHardenedVectorTest, ErrorDetectionAndCorrection)
{
    RadiationHardenedVector<int, 5> vec;

    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    // Get pointer to internal data through operator[]
    int& mutable_value = const_cast<int&>(vec[0]);

    // Simulate radiation-induced bit flip
    corrupt_memory(&mutable_value);

    // Validation should detect and correct the error
    EXPECT_TRUE(vec.validate_and_correct());

    // Value should be restored
    EXPECT_EQ(vec[0], 1);

    // Error should be counted
    EXPECT_GT(vec.get_error_count(), 0);
}

TEST(RadiationHardenedVectorTest, MultipleBackups)
{
    // Create vector with 3 copies
    RadiationHardenedVector<int, 5, 3> vec;

    vec.push_back(1);
    vec.push_back(2);

    // Corrupt primary copy
    int& mutable_value = const_cast<int&>(vec[0]);
    corrupt_memory(&mutable_value);

    // Validation should restore from backup
    EXPECT_TRUE(vec.validate_and_correct());
    EXPECT_EQ(vec[0], 1);
}

TEST(RadiationHardenedVectorTest, PopBack)
{
    RadiationHardenedVector<int, 3> vec;
    vec.push_back(1);
    vec.push_back(2);

    EXPECT_EQ(vec.size(), 2);
    vec.pop_back();
    EXPECT_EQ(vec.size(), 1);
    EXPECT_EQ(vec[0], 1);

    // Validate that all copies are consistent
    EXPECT_TRUE(vec.validate_and_correct());
}

TEST(RadiationHardenedVectorTest, Clear)
{
    RadiationHardenedVector<int, 3> vec;
    vec.push_back(1);
    vec.push_back(2);

    EXPECT_FALSE(vec.empty());
    vec.clear();
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0);

    // Validate that all copies are consistent
    EXPECT_TRUE(vec.validate_and_correct());
}

TEST(RadiationHardenedVectorTest, Telemetry)
{
    RadiationHardenedVector<int, 3> vec;

    EXPECT_EQ(vec.get_error_count(), 0);
    EXPECT_EQ(vec.get_correction_count(), 0);

    vec.push_back(1);

    // Corrupt data and validate
    int& mutable_value = const_cast<int&>(vec[0]);
    corrupt_memory(&mutable_value);
    vec.validate_and_correct();

    EXPECT_GT(vec.get_error_count(), 0);
    EXPECT_GT(vec.get_correction_count(), 0);

    // Reset telemetry
    vec.reset_telemetry();
    EXPECT_EQ(vec.get_error_count(), 0);
    EXPECT_EQ(vec.get_correction_count(), 0);
}
