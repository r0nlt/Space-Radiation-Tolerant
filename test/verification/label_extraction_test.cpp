#include <cassert>
#include <iostream>
#include <vector>

/**
 * @brief Extract the class label from one-hot encoded vector
 */
int extractClassFromOneHot(const std::vector<float>& labels, int sample_index, int num_classes)
{
    for (int i = 0; i < num_classes; ++i) {
        if (labels[sample_index * num_classes + i] == 1.0f) {
            return i;
        }
    }
    return -1;  // Error case
}

/**
 * @brief Test the label extraction functionality
 */
void testLabelExtraction()
{
    std::cout << "🔬 Testing One-Hot Label Extraction\n";
    std::cout << "===================================\n\n";

    // Create test data: 3 samples, 4 classes each
    // Sample 0: class 2, Sample 1: class 0, Sample 2: class 3
    std::vector<float> test_labels = {// Sample 0: class 2 (one-hot: [0, 0, 1, 0])
                                      0.0f, 0.0f, 1.0f, 0.0f,
                                      // Sample 1: class 0 (one-hot: [1, 0, 0, 0])
                                      1.0f, 0.0f, 0.0f, 0.0f,
                                      // Sample 2: class 3 (one-hot: [0, 0, 0, 1])
                                      0.0f, 0.0f, 0.0f, 1.0f};

    const int num_classes = 4;
    const int num_samples = 3;

    std::cout << "📊 Test Data Layout:\n";
    std::cout << "   Sample 0: [0, 0, 1, 0] -> Expected class: 2\n";
    std::cout << "   Sample 1: [1, 0, 0, 0] -> Expected class: 0\n";
    std::cout << "   Sample 2: [0, 0, 0, 1] -> Expected class: 3\n\n";

    bool all_passed = true;

    // Test each sample
    for (int sample = 0; sample < num_samples; ++sample) {
        int extracted_class = extractClassFromOneHot(test_labels, sample, num_classes);

        // Expected classes
        int expected_classes[] = {2, 0, 3};
        int expected = expected_classes[sample];

        bool passed = (extracted_class == expected);
        all_passed &= passed;

        std::cout << "🧪 Sample " << sample << ": ";
        std::cout << "Extracted=" << extracted_class << ", Expected=" << expected;
        std::cout << " " << (passed ? "✅ PASS" : "❌ FAIL") << "\n";
    }

    // Test error case (invalid sample index)
    int invalid_result = extractClassFromOneHot(test_labels, 999, num_classes);
    bool error_handled = (invalid_result == -1);
    all_passed &= error_handled;

    std::cout << "🧪 Invalid sample index: ";
    std::cout << "Result=" << invalid_result << " " << (error_handled ? "✅ PASS" : "❌ FAIL")
              << "\n";

    // Test MNIST-like scenario (10 classes)
    std::cout << "\n📊 MNIST-like Test (10 classes):\n";
    std::vector<float> mnist_labels = {// Sample 0: digit 7
                                       0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                       // Sample 1: digit 3
                                       0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                       // Sample 2: digit 0
                                       1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    int mnist_expected[] = {7, 3, 0};
    for (int sample = 0; sample < 3; ++sample) {
        int extracted = extractClassFromOneHot(mnist_labels, sample, 10);
        int expected = mnist_expected[sample];
        bool passed = (extracted == expected);
        all_passed &= passed;

        std::cout << "   Sample " << sample << " (digit " << expected << "): ";
        std::cout << "Extracted=" << extracted << " " << (passed ? "✅ PASS" : "❌ FAIL") << "\n";
    }

    std::cout << "\n" << std::string(40, '=') << "\n";
    if (all_passed) {
        std::cout << "✅ ALL LABEL EXTRACTION TESTS PASSED!\n";
        std::cout << "🚀 MNIST label extraction is now correct!\n";
    }
    else {
        std::cout << "❌ SOME LABEL EXTRACTION TESTS FAILED!\n";
        std::cout << "🔧 Please review the implementation.\n";
    }
    std::cout << std::string(40, '=') << "\n";
}

int main()
{
    testLabelExtraction();
    return 0;
}
