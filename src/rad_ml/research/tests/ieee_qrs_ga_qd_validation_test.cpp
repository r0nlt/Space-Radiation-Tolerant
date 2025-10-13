// File: src/rad_ml/research/tests/ieee_qrs_ga_qd_validation_test.cpp

#include <gtest/gtest.h>

#include <fstream>
#include <rad_ml/research/auto_arch_search.hpp>
#include <sstream>

using namespace rad_ml::research;

namespace {

// Helper to build a small, deterministic searcher configured for QRS-style validation
AutoArchSearch makeQRSValidationSearcher()
{
    // Minimal synthetic dataset
    std::vector<float> train_data(80, 0.25f);
    std::vector<float> train_labels(8, 1.0f);
    std::vector<float> test_data(24, 0.25f);
    std::vector<float> test_labels(3, 1.0f);

    AutoArchSearch s(train_data, train_labels, test_data, test_labels,
                     rad_ml::sim::Environment::EARTH_ORBIT, {32, 64, 128}, {0.3, 0.4, 0.5});

    s.setFixedParameters(8, 2, 2);
    s.setProtectionLevels(
        {rad_ml::neural::ProtectionLevel::NONE, rad_ml::neural::ProtectionLevel::CHECKSUM_ONLY,
         rad_ml::neural::ProtectionLevel::SELECTIVE_TMR, rad_ml::neural::ProtectionLevel::FULL_TMR,
         rad_ml::neural::ProtectionLevel::ADAPTIVE_TMR,
         rad_ml::neural::ProtectionLevel::SPACE_OPTIMIZED});
    s.setSeed(12345);
    s.setAdaptiveMutation(true, 0.1, 0.3, 0.5, 0.01);
    // Enable QD and Advanced QD for robustness search
    s.enableQualityDiversity(true);
    s.enableAdvancedQualityDiversity(true);
    return s;
}

}  // namespace

// IEEE QRS-style robustness validation: smoke test
TEST(IEEE_QRS_GA_QD, RobustnessSmoke)
{
    auto searcher = makeQRSValidationSearcher();

    const size_t pop = 6;
    const size_t gens = 2;

    auto result = searcher.evolutionarySearch(pop, gens, 0.2, 1, /*use_monte_carlo=*/true,
                                              /*monte_carlo_trials=*/5);

    // Basic sanity: values within expected bounds
    EXPECT_GE(result.baseline_accuracy, 0.0);
    EXPECT_LE(result.baseline_accuracy, 100.0);
    EXPECT_GE(result.radiation_accuracy, 0.0);
    EXPECT_LE(result.radiation_accuracy, 100.0);
    EXPECT_GE(result.accuracy_preservation, 0.0);
    EXPECT_LE(result.accuracy_preservation, 100.0);

    // Iteration count should equal gens * pop
    EXPECT_EQ(result.iterations, gens * pop);
}

// Verify per-generation genetics metrics are written (header + data rows)
TEST(IEEE_QRS_GA_QD, GeneticsMetricsCSVWritten)
{
    auto searcher = makeQRSValidationSearcher();

    const std::string metrics_file = "ieee_qrs_ga_qd_metrics.csv";
    searcher.setGeneticsMetricsFile(metrics_file);

    auto res = searcher.evolutionarySearch(6, 2, 0.2, 1, /*use_monte_carlo=*/false,
                                           /*monte_carlo_trials=*/1);
    (void)res;

    std::ifstream ifs(std::string("results/genetic_algorithm/") + metrics_file);
    ASSERT_TRUE(ifs.good()) << "Expected genetics metrics file at results/genetic_algorithm/"
                            << metrics_file;

    std::string header;
    ASSERT_TRUE(static_cast<bool>(std::getline(ifs, header)));
    EXPECT_NE(header.find("generation,best_preservation,mean_fitness,fitness_variance,diversity"),
              std::string::npos);

    std::string first_row;
    EXPECT_TRUE(static_cast<bool>(std::getline(ifs, first_row)));
}

// Reproducibility check under fixed seed (IEEE robustness requirement)
TEST(IEEE_QRS_GA_QD, ReproducibilityWithFixedSeed)
{
    auto s1 = makeQRSValidationSearcher();
    auto s2 = makeQRSValidationSearcher();

    auto r1 = s1.evolutionarySearch(6, 2, 0.2, 1, /*use_monte_carlo=*/true,
                                    /*monte_carlo_trials=*/5);
    auto r2 = s2.evolutionarySearch(6, 2, 0.2, 1, /*use_monte_carlo=*/true,
                                    /*monte_carlo_trials=*/5);

    // Allow tiny numerical variations, but expect equal results in this deterministic setup
    EXPECT_NEAR(r1.accuracy_preservation, r2.accuracy_preservation, 1e-6);
    EXPECT_NEAR(r1.baseline_accuracy, r2.baseline_accuracy, 1e-6);
    EXPECT_NEAR(r1.radiation_accuracy, r2.radiation_accuracy, 1e-6);
}

// Diversity clamping: ensure normalized diversity is in [0,1]
TEST(IEEE_QRS_GA_QD, DiversityIsClamped)
{
    auto s = makeQRSValidationSearcher();

    std::vector<NetworkConfig> pop;
    pop.push_back(NetworkConfig({8, 32, 2}, 0.3, false, rad_ml::neural::ProtectionLevel::NONE));
    pop.push_back(NetworkConfig({8, 128, 2}, 0.7, true, rad_ml::neural::ProtectionLevel::FULL_TMR));
    pop.push_back(
        NetworkConfig({8, 64, 64, 2}, 0.5, false, rad_ml::neural::ProtectionLevel::CHECKSUM_ONLY));

    double d = s.calculatePopulationDiversity_PUBLIC(pop);
    EXPECT_GE(d, 0.0);
    EXPECT_LE(d, 1.0);
}

// Adaptive mutation bounds: returned rate must be within configured [min, max]
TEST(IEEE_QRS_GA_QD, AdaptiveMutationRateWithinBounds)
{
    auto s = makeQRSValidationSearcher();
    // Ensure adaptive settings are known
    s.setAdaptiveMutation(true, /*base=*/0.1, /*threshold=*/0.3, /*max=*/0.5, /*min=*/0.01);

    // Craft small population and dummy fitness
    std::vector<NetworkConfig> pop(
        6, NetworkConfig({8, 64, 2}, 0.4, false, rad_ml::neural::ProtectionLevel::NONE));
    std::vector<double> fitness(6, 90.0);

    double rate = s.calculateAdaptiveMutationRate_PUBLIC(pop, fitness, /*generation=*/1,
                                                         /*total_generations=*/10);
    EXPECT_GE(rate, 0.01);
    EXPECT_LE(rate, 0.5);
}

// Config distance properties: identity and symmetry
TEST(IEEE_QRS_GA_QD, ConfigDistanceIdentityAndSymmetry)
{
    auto s = makeQRSValidationSearcher();

    NetworkConfig a({8, 64, 2}, 0.4, false, rad_ml::neural::ProtectionLevel::NONE);
    NetworkConfig b({8, 32, 2}, 0.6, true, rad_ml::neural::ProtectionLevel::FULL_TMR);

    double daa = s.calculateConfigDistance_PUBLIC(a, a);
    double dab = s.calculateConfigDistance_PUBLIC(a, b);
    double dba = s.calculateConfigDistance_PUBLIC(b, a);

    EXPECT_NEAR(daa, 0.0, 1e-12);
    EXPECT_NEAR(dab, dba, 1e-12);
}

// Diversity increases when configurations are more different
TEST(IEEE_QRS_GA_QD, DiversityRespondsToDifference)
{
    auto s = makeQRSValidationSearcher();

    std::vector<NetworkConfig> pop1 = {
        NetworkConfig({8, 64, 2}, 0.4, false, rad_ml::neural::ProtectionLevel::NONE),
        NetworkConfig({8, 64, 2}, 0.4, false, rad_ml::neural::ProtectionLevel::NONE),
    };
    std::vector<NetworkConfig> pop2 = {
        NetworkConfig({8, 32, 2}, 0.1, false, rad_ml::neural::ProtectionLevel::CHECKSUM_ONLY),
        NetworkConfig({8, 256, 2}, 0.9, true, rad_ml::neural::ProtectionLevel::FULL_TMR),
    };

    double d1 = s.calculatePopulationDiversity_PUBLIC(pop1);
    double d2 = s.calculatePopulationDiversity_PUBLIC(pop2);
    EXPECT_LT(d1, d2);
}

// Capture stdout to assert QD coverage logging appears during evolutionary search
TEST(IEEE_QRS_GA_QD, QDCoverageLoggingAppears)
{
    auto searcher = makeQRSValidationSearcher();
    // Ensure small but nontrivial run
    const size_t pop = 6;
    const size_t gens = 2;

    // Capture std::cout
    std::ostringstream capture;
    auto* old_buf = std::cout.rdbuf(capture.rdbuf());
    (void)old_buf;

    auto result = searcher.evolutionarySearch(pop, gens, 0.2, 1, /*use_monte_carlo=*/false,
                                              /*monte_carlo_trials=*/1);
    (void)result;

    // Restore cout
    std::cout.rdbuf(old_buf);

    std::string out = capture.str();
    // Expect presence of QD coverage logging from evolutionary loop
    // Example: "QD coverage: 0.1234% (occupied 3), elites injected: 1"
    bool has_qd_log = (out.find("QD coverage:") != std::string::npos);
    EXPECT_TRUE(has_qd_log);
}

// Monte Carlo trials propagate to SearchResult
TEST(IEEE_QRS_GA_QD, MonteCarloTrialsPropagate)
{
    auto searcher = makeQRSValidationSearcher();
    const size_t trials = 5;
    auto result = searcher.evolutionarySearch(6, 2, 0.2, 1, /*use_monte_carlo=*/true, trials);
    EXPECT_EQ(result.monte_carlo_trials, trials);
    EXPECT_GE(result.accuracy_preservation_stddev, 0.0);
}
