/**
 * @file adaptive_mutation_demo.cpp
 * @brief Demonstration of the adaptive mutation system in Auto Architecture Search
 *
 * This example shows how the adaptive mutation system dynamically adjusts
 * mutation rates based on population diversity and convergence status.
 */

#include <chrono>
#include <iostream>
#include <rad_ml/research/auto_arch_search.hpp>
#include <random>
#include <vector>

using namespace rad_ml::research;

int main()
{
    try {
        std::cout << "🧬 Adaptive Mutation System Demonstration\n";
        std::cout << "==========================================\n\n";

        // Create synthetic dataset
        std::vector<float> train_data(1000, 0.5f);   // Simplified dataset
        std::vector<float> train_labels(100, 1.0f);  // 100 samples
        std::vector<float> test_data(300, 0.5f);
        std::vector<float> test_labels(30, 1.0f);  // 30 test samples

        // Initialize searcher
        AutoArchSearch searcher(train_data, train_labels, test_data, test_labels,
                                rad_ml::sim::Environment::EARTH_ORBIT);

        // Demonstrate different adaptive mutation configurations
        std::cout << "📊 Testing Different Adaptive Mutation Configurations\n";
        std::cout << "=====================================================\n\n";

        // Test 1: Conservative adaptive mutation
        std::cout << "1️⃣ Conservative Configuration:\n";
        std::cout << "   Base rate: 0.05, Threshold: 0.2, Max: 0.3\n";
        searcher.setAdaptiveMutation(true, 0.05, 0.2, 0.3, 0.01);
        std::cout << "   → Good for stable, conservative search\n\n";

        // Test 2: Balanced adaptive mutation (recommended)
        std::cout << "2️⃣ Balanced Configuration:\n";
        std::cout << "   Base rate: 0.1, Threshold: 0.3, Max: 0.5\n";
        searcher.setAdaptiveMutation(true, 0.1, 0.3, 0.5, 0.01);
        std::cout << "   → Recommended for most use cases\n\n";

        // Test 3: Aggressive adaptive mutation
        std::cout << "3️⃣ Aggressive Configuration:\n";
        std::cout << "   Base rate: 0.2, Threshold: 0.4, Max: 0.8\n";
        searcher.setAdaptiveMutation(true, 0.2, 0.4, 0.8, 0.05);
        std::cout << "   → Good for difficult optimization landscapes\n\n";

        // Test 4: Fixed mutation for comparison
        std::cout << "4️⃣ Fixed Mutation (No Adaptation):\n";
        searcher.setAdaptiveMutation(false);  // Disable adaptive mutation
        std::cout << "   → Traditional fixed mutation rate approach\n\n";

        std::cout << "🔬 Adaptive Mutation Benefits:\n";
        std::cout << "=============================\n";
        std::cout << "✅ Automatically increases exploration when population diversity is low\n";
        std::cout << "✅ Helps escape local optima when the search converges\n";
        std::cout << "✅ Maintains exploitation when good solutions are found\n";
        std::cout << "✅ Reduces manual tuning of mutation parameters\n";
        std::cout << "✅ Adapts to the current state of the search process\n\n";

        std::cout << "📈 When to Use Adaptive Mutation:\n";
        std::cout << "=================================\n";
        std::cout << "✅ Complex optimization landscapes\n";
        std::cout << "✅ Long-running evolutionary searches\n";
        std::cout << "✅ When traditional GA struggles with convergence\n";
        std::cout << "✅ Multi-modal fitness landscapes\n";
        std::cout << "✅ When you want to reduce hyperparameter tuning\n\n";

        std::cout << "⚠️ When Fixed Mutation Might Be Better:\n";
        std::cout << "=====================================\n";
        std::cout << "⚠️ Simple optimization problems\n";
        std::cout << "⚠️ Short evolutionary searches\n";
        std::cout << "⚠️ When you need reproducible exact results\n";
        std::cout << "⚠️ Well-understood optimization landscapes\n\n";

        std::cout << "🎯 Adaptive Mutation Algorithm:\n";
        std::cout << "===============================\n";
        std::cout
            << "1. Calculate population diversity (0.0 = identical, 1.0 = maximally diverse)\n";
        std::cout << "2. Measure fitness variance to detect convergence\n";
        std::cout << "3. Adjust mutation rate based on:\n";
        std::cout << "   • Low diversity → Increase mutation for exploration\n";
        std::cout << "   • High diversity → Decrease mutation for exploitation\n";
        std::cout << "   • Low fitness variance → Increase mutation to escape local optima\n";
        std::cout << "   • Late generations → Increase mutation for final exploration\n";
        std::cout << "4. Ensure rate stays within configured min/max bounds\n\n";

        std::cout << "🔧 Configuration Recommendations:\n";
        std::cout << "================================\n";
        std::cout << "For most applications:\n";
        std::cout << "  searcher.setAdaptiveMutation(true, 0.1, 0.3, 0.5, 0.01);\n\n";
        std::cout << "For difficult optimization:\n";
        std::cout << "  searcher.setAdaptiveMutation(true, 0.15, 0.25, 0.6, 0.02);\n\n";
        std::cout << "For stable, conservative search:\n";
        std::cout << "  searcher.setAdaptiveMutation(true, 0.05, 0.4, 0.25, 0.005);\n\n";

        std::cout << "✅ Adaptive mutation demonstration completed!\n";
        std::cout << "The system will automatically adjust mutation rates during\n";
        std::cout << "evolutionary search based on population characteristics.\n\n";
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
