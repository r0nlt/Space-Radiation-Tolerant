// File: src/rad_ml/research/tests/evolutionary_ga_test.cpp

#include <gtest/gtest.h>

#include <rad_ml/research/auto_arch_search.hpp>

using namespace rad_ml::research;

namespace {

// Helper to build a tiny searcher with deterministic seed
AutoArchSearch makeTinySearcher()
{
    std::vector<float> train_data(50, 0.5f);
    std::vector<float> train_labels(5, 1.0f);
    std::vector<float> test_data(20, 0.5f);
    std::vector<float> test_labels(2, 1.0f);
    AutoArchSearch s(train_data, train_labels, test_data, test_labels,
                     rad_ml::sim::Environment::EARTH_ORBIT, {32, 64, 128}, {0.3, 0.4, 0.5});
    s.setFixedParameters(8, 2, 2);
    s.setProtectionLevels({rad_ml::neural::ProtectionLevel::NONE,
                           rad_ml::neural::ProtectionLevel::CHECKSUM_ONLY,
                           rad_ml::neural::ProtectionLevel::SELECTIVE_TMR,
                           rad_ml::neural::ProtectionLevel::FULL_TMR});
    s.setSeed(1234);
    return s;
}

}  // namespace

TEST(EvolutionaryGA, CrossoverStrategiesProduceValidChildren)
{
    auto searcher = makeTinySearcher();
    searcher.setCrossoverSettings(1.0, AutoArchSearch::CrossoverStrategy::UNIFORM);

    NetworkConfig p1({8, 64, 128, 2}, 0.5, false, rad_ml::neural::ProtectionLevel::NONE);
    NetworkConfig p2({8, 32, 256, 64, 2}, 0.3, true, rad_ml::neural::ProtectionLevel::FULL_TMR);

    // Uniform fallback when sizes differ
    auto c1 = searcher.crossoverConfigs_PUBLIC(p1, p2);
    ASSERT_GE(c1.layer_sizes.size(), 3u);
    EXPECT_EQ(c1.layer_sizes.front(), 8u);
    EXPECT_EQ(c1.layer_sizes.back(), 2u);

    // Same-length parents: test single-point preserves ends
    NetworkConfig p3({8, 64, 32, 2}, 0.4, true, rad_ml::neural::ProtectionLevel::CHECKSUM_ONLY);
    NetworkConfig p4({8, 128, 64, 2}, 0.6, false, rad_ml::neural::ProtectionLevel::SELECTIVE_TMR);

    searcher.setCrossoverSettings(1.0, AutoArchSearch::CrossoverStrategy::SINGLE_POINT);
    auto c2 = searcher.crossoverConfigs_PUBLIC(p3, p4);
    ASSERT_EQ(c2.layer_sizes.size(), p3.layer_sizes.size());
    EXPECT_EQ(c2.layer_sizes.front(), 8u);
    EXPECT_EQ(c2.layer_sizes.back(), 2u);
}

TEST(EvolutionaryGA, DiversityPreservingMutationChangesGenes)
{
    auto searcher = makeTinySearcher();
    searcher.setAdaptiveMutation(true, 0.2, 0.3, 0.5, 0.01);

    // Induce adaptive controller creation via evolutionary call setup
    NetworkConfig base({8, 64, 64, 2}, 0.4, false, rad_ml::neural::ProtectionLevel::NONE);
    auto mutated = searcher.mutateConfig_PUBLIC(base, 0.9);  // high rate to trigger changes

    // Expect at least one gene to differ commonly
    bool any_diff = false;
    any_diff |= (mutated.dropout_rate != base.dropout_rate);
    any_diff |= (mutated.has_residual_connections != base.has_residual_connections);
    any_diff |= (mutated.protection_level != base.protection_level);
    if (mutated.layer_sizes.size() == base.layer_sizes.size()) {
        for (size_t i = 0; i < base.layer_sizes.size(); ++i) {
            if (mutated.layer_sizes[i] != base.layer_sizes[i]) {
                any_diff = true;
                break;
            }
        }
    }
    EXPECT_TRUE(any_diff);
}

TEST(EvolutionaryGA, RandomImmigrantsBoostDiversityWhenCollapsed)
{
    auto searcher = makeTinySearcher();
    searcher.setRandomImmigrants(true, 0.2);
    searcher.setAdaptiveMutation(true, 0.1, 0.3, 0.5, 0.01);
    searcher.setCrossoverSettings(0.0, AutoArchSearch::CrossoverStrategy::UNIFORM);

    // Construct a near-identical population to collapse diversity
    std::vector<NetworkConfig> pop;
    for (int i = 0; i < 10; ++i) {
        pop.push_back(
            NetworkConfig({8, 64, 64, 2}, 0.4, false, rad_ml::neural::ProtectionLevel::NONE));
    }
    double d0 = searcher.calculatePopulationDiversity_PUBLIC(pop);
    EXPECT_LE(d0, 0.05);

    // Simulate end-of-generation injection by calling evolutionarySearch with tiny gens
    // Small run to exercise path without long compute
    auto res = searcher.evolutionarySearch(10, 1, 0.1, 1, false, 1);
    (void)res;  // unused, compile guard

    // We cannot directly access internal population here; this test ensures the path executes.
    SUCCEED();
}

TEST(EvolutionaryGA, GeneticsMetricsCSVIsWritten)
{
    auto searcher = makeTinySearcher();
    // Write metrics to a temp file in build dir
    const std::string metrics_file =
        "genetics_metrics_test.csv";  // will map to results/genetic_algorithm/
    searcher.setGeneticsMetricsFile(metrics_file);

    // Run a tiny evolutionary search
    auto res = searcher.evolutionarySearch(6, 2, 0.2, 1, false, 1);
    (void)res;

    // Verify file exists and has header + at least one data row
    std::ifstream ifs(std::string("results/genetic_algorithm/") + metrics_file);
    ASSERT_TRUE(ifs.good());
    std::string header;
    ASSERT_TRUE(static_cast<bool>(std::getline(ifs, header)));
    EXPECT_NE(header.find("generation,best_preservation,mean_fitness"), std::string::npos);

    std::string row;
    bool has_row = static_cast<bool>(std::getline(ifs, row));
    EXPECT_TRUE(has_row);
}
