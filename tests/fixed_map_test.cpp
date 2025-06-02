#include <gtest/gtest.h>

#include <space_containers/fixed_map.hpp>
#include <string>

using namespace space_containers;

TEST(FixedMapTest, BasicOperations)
{
    FixedMap<std::string, int, 5> map;

    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
    EXPECT_EQ(map.max_size(), 5);

    EXPECT_TRUE(map.insert("one", 1));
    EXPECT_TRUE(map.insert("two", 2));
    EXPECT_TRUE(map.insert("three", 3));

    EXPECT_EQ(map.size(), 3);
    EXPECT_FALSE(map.empty());

    EXPECT_EQ(map["one"], 1);
    EXPECT_EQ(map["two"], 2);
    EXPECT_EQ(map["three"], 3);
}

TEST(FixedMapTest, CapacityLimits)
{
    FixedMap<std::string, int, 3> map;

    EXPECT_TRUE(map.insert("one", 1));
    EXPECT_TRUE(map.insert("two", 2));
    EXPECT_TRUE(map.insert("three", 3));
    EXPECT_FALSE(map.insert("four", 4));  // Should fail, map is full

    EXPECT_TRUE(map.full());
    EXPECT_EQ(map.size(), 3);
}

TEST(FixedMapTest, MoveSemantics)
{
    FixedMap<std::string, std::string, 3> map;

    std::string value = "test_value";
    EXPECT_TRUE(map.insert("key", std::move(value)));
    EXPECT_TRUE(value.empty());  // Original string should be moved from

    EXPECT_EQ(map["key"], "test_value");
}

TEST(FixedMapTest, EmplaceAndFind)
{
    FixedMap<std::string, std::string, 3> map;

    EXPECT_TRUE(map.emplace("key", "test"));
    EXPECT_TRUE(map.emplace("key2", 5, 'a'));  // Creates string "aaaaa"

    auto value1 = map.find("key");
    EXPECT_TRUE(value1.has_value());
    EXPECT_EQ(*value1, "test");

    auto value2 = map.find("key2");
    EXPECT_TRUE(value2.has_value());
    EXPECT_EQ(*value2, "aaaaa");

    auto value3 = map.find("nonexistent");
    EXPECT_FALSE(value3.has_value());
}

TEST(FixedMapTest, OperatorAccess)
{
    FixedMap<std::string, int, 3> map;

    // Operator[] should create new element if key doesn't exist
    map["one"] = 1;
    EXPECT_EQ(map.size(), 1);
    EXPECT_EQ(map["one"], 1);

    // Modifying existing element
    map["one"] = 2;
    EXPECT_EQ(map.size(), 1);
    EXPECT_EQ(map["one"], 2);
}

TEST(FixedMapTest, EraseAndClear)
{
    FixedMap<std::string, int, 3> map;

    map.insert("one", 1);
    map.insert("two", 2);

    EXPECT_EQ(map.size(), 2);

    // Test erase
    EXPECT_TRUE(map.erase("one"));
    EXPECT_EQ(map.size(), 1);
    EXPECT_FALSE(map.contains("one"));
    EXPECT_TRUE(map.contains("two"));

    // Test clear
    map.clear();
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
}

TEST(FixedMapTest, Iterators)
{
    FixedMap<std::string, int, 5> map;
    map.insert("one", 1);
    map.insert("two", 2);
    map.insert("three", 3);

    int sum = 0;
    for (const auto& pair : map) {
        sum += pair.second;
    }
    EXPECT_EQ(sum, 6);

    // Test const iterators
    const FixedMap<std::string, int, 5>& const_map = map;
    sum = 0;
    for (auto it = const_map.cbegin(); it != const_map.cend(); ++it) {
        sum += it->second;
    }
    EXPECT_EQ(sum, 6);
}

TEST(FixedMapTest, Telemetry)
{
    FixedMap<std::string, int, 3> map;

    EXPECT_EQ(map.get_access_count(), 0);
    EXPECT_EQ(map.get_error_count(), 0);
    EXPECT_EQ(map.get_find_count(), 0);

    map.insert("one", 1);
    map["one"];       // Access element
    map.find("one");  // Find element

    EXPECT_GT(map.get_access_count(), 0);
    EXPECT_GT(map.get_find_count(), 0);

    // Try to insert when full
    map.insert("two", 2);
    map.insert("three", 3);
    map.insert("four", 4);  // Should fail and increment error count
    EXPECT_GT(map.get_error_count(), 0);
}

TEST(FixedMapTest, OrderPreservation)
{
    FixedMap<std::string, int, 5> map;

    map.insert("one", 1);
    map.insert("two", 2);
    map.insert("three", 3);

    // Erase middle element
    map.erase("two");

    // Check order preservation
    auto it = map.begin();
    EXPECT_EQ(it->first, "one");
    ++it;
    EXPECT_EQ(it->first, "three");
}

TEST(FixedMapTest, Alignment)
{
    // Test custom alignment
    FixedMap<int, int, 5, 16> aligned_map;
    aligned_map.insert(1, 1);

    // The actual test of alignment would require access to internal data structure
    // This test is more of a compilation test for alignment specification
    EXPECT_TRUE(true);
}
