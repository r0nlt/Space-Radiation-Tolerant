/**
 * @file auto_arch_search_example.cpp
 * @brief Example demonstrating the automatic architecture search functionality
 *
 * This example shows how to use the AutoArchSearch class to find optimal
 * neural network architectures for radiation environments.
 */

#include <rad_ml/research/auto_arch_search.hpp>
// Removing logger include that's causing build errors
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Helper function to create a synthetic dataset for testing
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>>
createSyntheticDataset(size_t train_size, size_t test_size, size_t input_size, size_t num_classes)
{
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create vectors to hold data
    std::vector<float> train_data(train_size * input_size);
    std::vector<float> train_labels(train_size * num_classes, 0.0f);
    std::vector<float> test_data(test_size * input_size);
    std::vector<float> test_labels(test_size * num_classes, 0.0f);

    // Generate training data
    for (size_t i = 0; i < train_size; ++i) {
        // Generate input features
        for (size_t j = 0; j < input_size; ++j) {
            train_data[i * input_size + j] = dist(gen);
        }

        // Generate one-hot encoded label
        size_t class_idx = i % num_classes;
        train_labels[i * num_classes + class_idx] = 1.0f;
    }

    // Generate test data
    for (size_t i = 0; i < test_size; ++i) {
        // Generate input features
        for (size_t j = 0; j < input_size; ++j) {
            test_data[i * input_size + j] = dist(gen);
        }

        // Generate one-hot encoded label
        size_t class_idx = i % num_classes;
        test_labels[i * num_classes + class_idx] = 1.0f;
    }

    return {train_data, train_labels, test_data, test_labels};
}

int main(int argc, char** argv)
{
    try {
        std::cout << "🚀 RadML Auto Architecture Search Example\n";
        std::cout << "==========================================\n\n";

        // Defaults
        size_t trials = 10;
        size_t schedule_interval = 2;  // 0 = every gen
        size_t freeze_after_gen = 4;   // SIZE_MAX like behavior not needed here

        // Parse super simple CLI: --trials N --schedule K --freeze G
        for (int i = 1; i + 1 < argc; ++i) {
            if (std::strcmp(argv[i], "--trials") == 0) {
                trials = static_cast<size_t>(std::stoul(argv[i + 1]));
            }
            else if (std::strcmp(argv[i], "--schedule") == 0) {
                schedule_interval = static_cast<size_t>(std::stoul(argv[i + 1]));
            }
            else if (std::strcmp(argv[i], "--freeze") == 0) {
                freeze_after_gen = static_cast<size_t>(std::stoul(argv[i + 1]));
            }
        }

        // Create synthetic dataset with consistent dimensions
        auto [train_data, train_labels, test_data, test_labels] =
            createSyntheticDataset(100, 30, 8, 4);  // 8 inputs, 4 outputs

        std::cout << "📊 Dataset created:\n";
        std::cout << "   Training samples: " << train_data.size() / 8 << "\n";
        std::cout << "   Test samples: " << test_data.size() / 8 << "\n";
        std::cout << "   Input features: 8\n";
        std::cout << "   Output classes: 4\n\n";

        // Set up AutoArchSearch with consistent parameters
        rad_ml::research::AutoArchSearch searcher(train_data, train_labels, test_data, test_labels,
                                                  rad_ml::sim::Environment::EARTH_ORBIT,
                                                  {32, 64, 128, 256},               // Width options
                                                  {0.3, 0.4, 0.5, 0.6},             // Dropout rates
                                                  "auto_arch_search_results.csv");  // Results file

        // Set protection levels using correct namespace and enum values
        searcher.setProtectionLevels({rad_ml::neural::ProtectionLevel::NONE,
                                      rad_ml::neural::ProtectionLevel::CHECKSUM_ONLY,
                                      rad_ml::neural::ProtectionLevel::SELECTIVE_TMR,
                                      rad_ml::neural::ProtectionLevel::FULL_TMR,
                                      rad_ml::neural::ProtectionLevel::ADAPTIVE_TMR,
                                      rad_ml::neural::ProtectionLevel::SPACE_OPTIMIZED});

        // Set deterministic seed for reproducible results
        searcher.setSeed(42);

        // Enable adaptive mutation based on population diversity
        searcher.setAdaptiveMutation(true, 0.1, 0.3, 0.5, 0.01);
        // Decouple policy: configurable schedule and freeze
        searcher.setMutationRateSchedule(schedule_interval);
        searcher.setMutationRateFreezeGeneration(freeze_after_gen);

        std::cout << "🧬 Starting evolutionary architecture search...\n";
        std::cout << "   Population size: 10\n";
        std::cout << "   Generations: 5\n";
        std::cout << "   Monte Carlo trials: 10\n";
        std::cout << "   Adaptive mutation: ENABLED\n\n";

        std::cout << "📊 Adaptive Mutation Parameters:\n";
        std::cout << "   Base rate: 0.1\n";
        std::cout << "   Diversity threshold: 0.3\n";
        std::cout << "   Max rate: 0.5\n";
        std::cout << "   Min rate: 0.01\n";
        std::cout << "   Schedule interval: " << schedule_interval << " generations\n";
        std::cout << "   Freeze after generation: " << freeze_after_gen << "\n\n";

        // Run ONLY evolutionary search (remove random search)
        auto result = searcher.evolutionarySearch(10, 5, 0.1, 5, true, trials);

        // Save a per-run summary CSV (append)
        {
            std::ofstream summary("run_summaries.csv", std::ios::app);
            if (summary.tellp() == 0) {
                summary << "trials,schedule,freeze,baseline_acc,radiation_acc,preservation,"
                        << "preservation_stddev,iterations\n";
            }
            summary << trials << "," << schedule_interval << "," << freeze_after_gen << ","
                    << std::fixed << std::setprecision(2) << result.baseline_accuracy << ","
                    << result.radiation_accuracy << "," << result.accuracy_preservation << ","
                    << std::setprecision(3) << result.accuracy_preservation_stddev << ","
                    << result.iterations << "\n";
        }

        // Rename operator_stats.csv to include parameters for comparison
        // (the exporter writes to CWD; move if present)
        {
            std::ifstream test("operator_stats.csv");
            if (test.good()) {
                std::ostringstream name;
                name << "operator_stats_trials" << trials << "_sched" << schedule_interval
                     << "_freeze" << freeze_after_gen << ".csv";
                std::string out_name = name.str();
                std::ifstream src("operator_stats.csv", std::ios::binary);
                std::ofstream dst(out_name, std::ios::binary);
                dst << src.rdbuf();
            }
        }

        std::cout << "\n🎯 Search Results:\n";
        std::cout << "==================\n";
        std::cout << "Best LEO-optimized architecture found:\n";
        std::cout << "Layer sizes: ";
        for (size_t size : result.config.layer_sizes) {
            std::cout << size << "-";
        }
        std::cout << "\n";
        std::cout << "Dropout rate: " << std::fixed << std::setprecision(2)
                  << result.config.dropout_rate << "\n";
        std::cout << "Protection level: " << static_cast<int>(result.config.protection_level)
                  << "\n";
        std::cout << "Baseline accuracy: " << std::fixed << std::setprecision(2)
                  << result.baseline_accuracy << "%\n";
        std::cout << "Radiation accuracy: " << std::fixed << std::setprecision(2)
                  << result.radiation_accuracy << "%\n";
        std::cout << "Accuracy preservation: " << std::fixed << std::setprecision(2)
                  << result.accuracy_preservation << "%\n";
        std::cout << "Confidence interval: ±" << std::fixed << std::setprecision(1)
                  << result.accuracy_preservation_stddev << "%\n";
        std::cout << "Total iterations: " << result.iterations << "\n";

        // Export results
        std::cout << "\n📁 Exporting results to CSV...\n";
        searcher.exportResults("auto_arch_search_results.csv");
        std::cout << "Results saved to: auto_arch_search_results.csv\n";

        std::cout << "\n✅ Auto architecture search completed successfully!\n";
        std::cout << "\n🔬 Adaptive Mutation Analysis:\n";
        std::cout << "============================\n";
        std::cout << "The adaptive mutation system dynamically adjusted the mutation rate\n";
        std::cout << "based on population diversity during the evolutionary search.\n";
        std::cout << "This improves the genetic algorithm's ability to:\n";
        std::cout << "• Escape local optima when diversity is low\n";
        std::cout << "• Focus on exploitation when diversity is high\n";
        std::cout << "• Respond to population convergence\n";
        std::cout << "• Maintain exploration in later generations\n\n";
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
