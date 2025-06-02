#include <gtest/gtest.h>

#include <space_containers/fixed_vector.hpp>
#include <string>

using namespace space_containers;

TEST(FixedVectorTest, BasicOperations)
{
    FixedVector<int, 5> vec;

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

TEST(FixedVectorTest, CapacityLimits)
{
    FixedVector<int, 3> vec;

    EXPECT_TRUE(vec.push_back(1));
    EXPECT_TRUE(vec.push_back(2));
    EXPECT_TRUE(vec.push_back(3));
    EXPECT_FALSE(vec.push_back(4));  // Should fail, vector is full

    EXPECT_TRUE(vec.full());
    EXPECT_EQ(vec.size(), 3);
}

TEST(FixedVectorTest, MoveSemantics)
{
    FixedVector<std::string, 3> vec;

    std::string str = "test";
    EXPECT_TRUE(vec.push_back(std::move(str)));
    EXPECT_TRUE(str.empty());  // Original string should be moved from

    EXPECT_EQ(vec[0], "test");
}

TEST(FixedVectorTest, EmplaceBack)
{
    FixedVector<std::string, 3> vec;

    EXPECT_TRUE(vec.emplace_back("test"));
    EXPECT_TRUE(vec.emplace_back(5, 'a'));  // Creates string "aaaaa"

    EXPECT_EQ(vec[0], "test");
    EXPECT_EQ(vec[1], "aaaaa");
}

TEST(FixedVectorTest, CopyAndMove)
{
    FixedVector<int, 3> vec1;
    vec1.push_back(1);
    vec1.push_back(2);

    // Test copy constructor
    FixedVector<int, 3> vec2(vec1);
    EXPECT_EQ(vec2.size(), 2);
    EXPECT_EQ(vec2[0], 1);
    EXPECT_EQ(vec2[1], 2);

    // Test move constructor
    FixedVector<int, 3> vec3(std::move(vec2));
    EXPECT_EQ(vec3.size(), 2);
    EXPECT_EQ(vec3[0], 1);
    EXPECT_EQ(vec3[1], 2);
    EXPECT_EQ(vec2.size(), 0);  // Original vector should be empty after move
}

TEST(FixedVectorTest, Iterators)
{
    FixedVector<int, 5> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    int sum = 0;
    for (const auto& val : vec) {
        sum += val;
    }
    EXPECT_EQ(sum, 6);

    // Test const iterators
    const FixedVector<int, 5>& const_vec = vec;
    sum = 0;
    for (auto it = const_vec.cbegin(); it != const_vec.cend(); ++it) {
        sum += *it;
    }
    EXPECT_EQ(sum, 6);
}

TEST(FixedVectorTest, Telemetry)
{
    FixedVector<int, 3> vec;

    EXPECT_EQ(vec.get_access_count(), 0);
    EXPECT_EQ(vec.get_error_count(), 0);

    vec.push_back(1);
    vec[0];  // Access element

    EXPECT_GT(vec.get_access_count(), 0);

// Try to access out of bounds
#ifdef NDEBUG
    vec[5];  // This should increment error count in release mode
    EXPECT_GT(vec.get_error_count(), 0);
#endif
}

TEST(FixedVectorTest, Alignment)
{
    // Test custom alignment
    FixedVector<int, 5, 16> aligned_vec;
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned_vec.begin()) % 16, 0);
}

TEST(FixedVectorTest, UnsafeAccess)
{
    FixedVector<int, 3> vec;
    vec.push_back(1);

    // Test unsafe access (should be faster but no bounds checking)
    EXPECT_EQ(vec.unsafe_access(0), 1);
}

TEST(FixedVectorTest, PopBack)
{
    FixedVector<int, 3> vec;
    vec.push_back(1);
    vec.push_back(2);

    EXPECT_EQ(vec.size(), 2);
    vec.pop_back();
    EXPECT_EQ(vec.size(), 1);
    EXPECT_EQ(vec[0], 1);
}

TEST(FixedVectorTest, Clear)
{
    FixedVector<int, 3> vec;
    vec.push_back(1);
    vec.push_back(2);

    EXPECT_FALSE(vec.empty());
    vec.clear();
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0);
}
