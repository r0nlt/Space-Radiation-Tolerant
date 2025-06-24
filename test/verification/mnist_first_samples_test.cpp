#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

/**
 * @brief Reverse bytes for big-endian to little-endian conversion
 */
uint32_t reverseBytes(uint32_t value)
{
    return ((value & 0xFF000000) >> 24) | ((value & 0x00FF0000) >> 8) |
           ((value & 0x0000FF00) << 8) | ((value & 0x000000FF) << 24);
}

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
 * @brief Load MNIST labels (simplified version)
 */
std::vector<float> loadMNISTLabels(const std::string& filename, int max_samples = 20)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open MNIST label file: " + filename);
    }

    // Read header
    uint32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    // Convert from big-endian
    magic = reverseBytes(magic);
    num_labels = reverseBytes(num_labels);

    if (magic != 0x00000801) {
        throw std::runtime_error("Invalid MNIST label file magic number");
    }

    // Limit number of samples
    if (max_samples > 0 && max_samples < static_cast<int>(num_labels)) {
        num_labels = max_samples;
    }

    // Load labels and convert to one-hot encoding (flattened)
    std::vector<float> labels;
    labels.reserve(num_labels * 10);

    for (uint32_t i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);

        // Add one-hot encoded label (10 values per label)
        for (int j = 0; j < 10; ++j) {
            labels.push_back(j == label ? 1.0f : 0.0f);
        }
    }

    return labels;
}

int main()
{
    std::cout << "🔍 MNIST First Samples Investigation\n";
    std::cout << "====================================\n\n";

    try {
        // Load first 20 training labels
        std::cout << "📚 Loading first 20 MNIST training labels...\n";
        auto train_labels = loadMNISTLabels("data/MNIST/raw/train-labels-idx1-ubyte", 20);

        std::cout << "📊 First 20 MNIST Training Samples:\n";
        std::cout << "Sample Index | Digit Label\n";
        std::cout << "-------------|------------\n";

        for (int i = 0; i < 20; ++i) {
            int digit = extractClassFromOneHot(train_labels, i, 10);
            std::cout << "     " << std::setw(2) << i << "      |     " << digit << "\n";
        }

        // Load first 20 test labels
        std::cout << "\n📚 Loading first 20 MNIST test labels...\n";
        auto test_labels = loadMNISTLabels("data/MNIST/raw/t10k-labels-idx1-ubyte", 20);

        std::cout << "📊 First 20 MNIST Test Samples:\n";
        std::cout << "Sample Index | Digit Label\n";
        std::cout << "-------------|------------\n";

        for (int i = 0; i < 20; ++i) {
            int digit = extractClassFromOneHot(test_labels, i, 10);
            std::cout << "     " << std::setw(2) << i << "      |     " << digit << "\n";
        }

        std::cout << "\n🎯 Conclusion:\n";
        std::cout << "The reason you always see digit '5' is because:\n";
        std::cout << "- The code always uses sample_index = 0 (first sample)\n";
        std::cout << "- The first training sample in MNIST happens to be digit '"
                  << extractClassFromOneHot(train_labels, 0, 10) << "'\n";
        std::cout << "- To see variety, we should show different sample indices!\n";
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
