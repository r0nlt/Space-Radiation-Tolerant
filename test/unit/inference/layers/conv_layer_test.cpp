#include <rad_ml/inference/layers/conv_layer.hpp>
#include <rad_ml/math/fixed_point.hpp>
#include <rad_ml/testing/fault_injector.hpp>
#include <cassert>
#include <iostream>
#include <cmath>

using namespace rad_ml::inference::layers;
using namespace rad_ml::math;
using namespace rad_ml::testing;

// Simple test framework
#define TEST(name) void name()
#define ASSERT(condition) assert(condition)
#define ASSERT_NEAR(a, b, epsilon) assert(std::abs((a) - (b)) < (epsilon))

// Test convolution with identity kernel
TEST(test_identity_convolution) {
    // Define fixed-point type for this test
    using fixed_t = Fixed8_8;
    
    // Create input tensor filled with 1.0
    fixed_t input[1][4][4];
    for (size_t h = 0; h < 4; ++h) {
        for (size_t w = 0; w < 4; ++w) {
            input[0][h][w] = fixed_t(1.0);
        }
    }
    
    // Define the convolutional layer type
    using ConvLayer = ConvolutionalLayer<fixed_t, 1, 1, 4, 4, 3, 2, 0>;
    
    // Calculate output dimensions
    constexpr size_t OutputHeight = ConvLayer::OutputHeight; // Should be 2
    constexpr size_t OutputWidth = ConvLayer::OutputWidth;   // Should be 2
    
    // Create output tensor with correct dimensions
    fixed_t output[1][OutputHeight][OutputWidth];
    
    // Create convolutional layer
    ConvLayer conv_layer;
    
    // Set weights to create an identity filter (center = 1, rest = 0)
    for (size_t kh = 0; kh < 3; ++kh) {
        for (size_t kw = 0; kw < 3; ++kw) {
            if (kh == 1 && kw == 1) {
                conv_layer.setWeight(0, 0, kh, kw, fixed_t(1.0));
            } else {
                conv_layer.setWeight(0, 0, kh, kw, fixed_t(0.0));
            }
        }
    }
    
    // Set bias to 0
    conv_layer.setBias(0, fixed_t(0.0));
    
    // Forward pass
    conv_layer.forward(input, output);
    
    // Check output - should be all 1.0
    for (size_t h = 0; h < OutputHeight; ++h) {
        for (size_t w = 0; w < OutputWidth; ++w) {
            ASSERT_NEAR(output[0][h][w].to_float(), 1.0f, 0.01f);
        }
    }
}

// Test convolution with a simple edge detection kernel
TEST(test_edge_detection) {
    // Define fixed-point type for this test
    using fixed_t = Fixed8_8;
    
    // Create input tensor - simple 3x3 image with an edge
    fixed_t input[1][3][3] = {
        {
            {fixed_t(1.0), fixed_t(1.0), fixed_t(0.0)},
            {fixed_t(1.0), fixed_t(1.0), fixed_t(0.0)},
            {fixed_t(1.0), fixed_t(1.0), fixed_t(0.0)}
        }
    };
    
    // Define the convolutional layer type
    using ConvLayer = ConvolutionalLayer<fixed_t, 1, 1, 3, 3, 3, 1, 0>;
    
    // Calculate output dimensions
    constexpr size_t OutputHeight = ConvLayer::OutputHeight; // Should be 1
    constexpr size_t OutputWidth = ConvLayer::OutputWidth;   // Should be 1
    
    // Create output tensor with correct dimensions
    fixed_t output[1][OutputHeight][OutputWidth];
    
    // Create convolutional layer
    ConvLayer conv_layer;
    
    // Set weights to create a horizontal edge detection filter
    // -1 -1 -1
    //  0  0  0
    //  1  1  1
    for (size_t kh = 0; kh < 3; ++kh) {
        for (size_t kw = 0; kw < 3; ++kw) {
            if (kh == 0) {
                conv_layer.setWeight(0, 0, kh, kw, fixed_t(-1.0));
            } else if (kh == 1) {
                conv_layer.setWeight(0, 0, kh, kw, fixed_t(0.0));
            } else { // kh == 2
                conv_layer.setWeight(0, 0, kh, kw, fixed_t(1.0));
            }
        }
    }
    
    // Set bias to 0
    conv_layer.setBias(0, fixed_t(0.0));
    
    // Forward pass
    conv_layer.forward(input, output);
    
    // Check output - should detect no horizontal edge (0.0)
    ASSERT_NEAR(output[0][0][0].to_float(), 0.0f, 0.01f);
    
    // Now set weights to create a vertical edge detection filter
    // -1  0  1
    // -1  0  1
    // -1  0  1
    for (size_t kh = 0; kh < 3; ++kh) {
        for (size_t kw = 0; kw < 3; ++kw) {
            if (kw == 0) {
                conv_layer.setWeight(0, 0, kh, kw, fixed_t(-1.0));
            } else if (kw == 1) {
                conv_layer.setWeight(0, 0, kh, kw, fixed_t(0.0));
            } else { // kw == 2
                conv_layer.setWeight(0, 0, kh, kw, fixed_t(1.0));
            }
        }
    }
    
    // Forward pass again
    conv_layer.forward(input, output);
    
    // Check output - should detect a vertical edge (-3.0)
    // The vertical edge kernel is:
    //   -1  0  1
    //   -1  0  1
    //   -1  0  1
    // When applied to the input:
    //    1  1  0
    //    1  1  0
    //    1  1  0
    // The result is:
    //   (-1*1 + 0*1 + 1*0) + (-1*1 + 0*1 + 1*0) + (-1*1 + 0*1 + 1*0) = -3
    ASSERT_NEAR(output[0][0][0].to_float(), -3.0f, 0.01f);
}

// Test resilience to faults
TEST(test_convolution_resilience) {
    // Define fixed-point type for this test
    using fixed_t = Fixed8_8;
    
    // Define the convolutional layer type and its output dimensions
    using ConvLayer = ConvolutionalLayer<fixed_t, 1, 1, 3, 3, 3, 1, 0>;
    constexpr size_t OutputHeight = ConvLayer::OutputHeight; // Should be 1
    constexpr size_t OutputWidth = ConvLayer::OutputWidth;   // Should be 1
    
    // Create convolutional layer
    ConvLayer conv_layer;
    
    // Set weights to some non-zero value
    for (size_t kh = 0; kh < 3; ++kh) {
        for (size_t kw = 0; kw < 3; ++kw) {
            conv_layer.setWeight(0, 0, kh, kw, fixed_t(0.1 * (kh * 3 + kw)));
        }
    }
    
    // Set bias
    conv_layer.setBias(0, fixed_t(0.5));
    
    // Create a fault injector
    FaultInjector injector;
    
    // Inject a fault into the weights (using direct memory access for simplicity)
    // In a real system, this would be done through a more controlled interface
    void* layer_ptr = &conv_layer;
    injector.injectFault(layer_ptr, sizeof(conv_layer), FaultType::SingleBitFlip);
    
    // Repair the layer
    conv_layer.repair();
    
    // Create input and output tensors with correct dimensions
    fixed_t input[1][3][3];
    fixed_t output[1][OutputHeight][OutputWidth];
    
    // Fill input with 1.0
    for (size_t h = 0; h < 3; ++h) {
        for (size_t w = 0; w < 3; ++w) {
            input[0][h][w] = fixed_t(1.0);
        }
    }
    
    // Forward pass
    conv_layer.forward(input, output);
    
    // The test passes if we get here without crashing
    std::cout << "Convolutional layer test with fault injection completed" << std::endl;
}

int main() {
    std::cout << "Running convolutional layer tests..." << std::endl;
    
    test_identity_convolution();
    test_edge_detection();
    test_convolution_resilience();
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
} 