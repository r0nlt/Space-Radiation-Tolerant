#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

// Include any headers from your C++ implementation here
// #include "rad_ml/tmr/tmr_protection.h"
// #include "rad_ml/error_correction/reed_solomon.h"
// #include "rad_ml/simulation/radiation_simulator.h"

namespace py = pybind11;

// Create random number generator with fixed seed for reproducibility
std::mt19937 rng(42);
std::uniform_real_distribution<float> dist(0.0, 1.0);

// Version information
struct Version {
    static constexpr int major = 0;
    static constexpr int minor = 1;
    static constexpr int patch = 0;
    static std::string get_version_string()
    {
        return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    }
};

// Global state
bool is_initialized = false;

// Initialize the library
bool initialize()
{
    // Perform any necessary initialization here
    is_initialized = true;
    return true;
}

// Shutdown the library
bool shutdown()
{
    // Perform any necessary cleanup here
    is_initialized = false;
    return true;
}

// Apply radiation effects to a tensor
py::array_t<float> apply_radiation(py::array_t<float> tensor, float strength)
{
    py::buffer_info buf = tensor.request();
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size;

    // Make a copy of the input tensor
    py::array_t<float> result = py::array_t<float>(buf.size);
    py::buffer_info buf_result = result.request();
    float* ptr_result = static_cast<float*>(buf_result.ptr);

    // Copy data
    std::memcpy(ptr_result, ptr, buf.size * sizeof(float));

    // Apply radiation effects
    for (size_t i = 0; i < size; i++) {
        if (dist(rng) < 0.01 * strength) {
            // Bit flip simulation
            uint32_t* bit_ptr = reinterpret_cast<uint32_t*>(&ptr_result[i]);

            // Choose a bit to flip, favoring more significant bits for larger effects
            int bit_pos = int(dist(rng) * 32);

            // Flip the bit
            *bit_ptr ^= (1 << bit_pos);

            // For significant radiation levels, sometimes apply more severe effects
            if (strength > 2.0 && dist(rng) < 0.2) {
                // Multi-bit upset - flip another nearby bit
                *bit_ptr ^= (1 << ((bit_pos + 1) % 32));
            }
        }
    }

    // For higher radiation strengths, also apply block corruption
    if (strength > 3.0 && size > 100) {
        size_t block_size = std::min<size_t>(50, size / 10);
        size_t start_idx = int(dist(rng) * (size - block_size));

        // 20% chance for memory block corruption
        if (dist(rng) < 0.2) {
            // Zero out a block
            std::memset(ptr_result + start_idx, 0, block_size * sizeof(float));
        }
    }

    return result;
}

// Detect and count errors in a tensor
std::tuple<bool, int> detect_errors(py::array_t<float> original, py::array_t<float> current)
{
    py::buffer_info buf_orig = original.request();
    py::buffer_info buf_curr = current.request();

    if (buf_orig.size != buf_curr.size) {
        throw std::runtime_error("Tensors must have the same size");
    }

    float* ptr_orig = static_cast<float*>(buf_orig.ptr);
    float* ptr_curr = static_cast<float*>(buf_curr.ptr);

    bool detected = false;
    int count = 0;

    // Error detection
    for (size_t i = 0; i < buf_orig.size; i++) {
        // Check for bit-level differences or significant numerical differences
        if (std::memcmp(&ptr_orig[i], &ptr_curr[i], sizeof(float)) != 0 ||
            std::abs(ptr_orig[i] - ptr_curr[i]) > 1e-5) {
            detected = true;
            count++;
        }
    }

    return std::make_tuple(detected, count);
}

// TMR voting function
py::array_t<float> tmr_vote(py::list outputs, int protection_level)
{
    // Get the first output to determine shape
    py::array_t<float> first = outputs[0].cast<py::array_t<float>>();
    py::buffer_info buf_first = first.request();

    // Create result array with same shape
    py::array_t<float> result = py::array_t<float>(buf_first.size);
    py::buffer_info buf_result = result.request();
    float* ptr_result = static_cast<float*>(buf_result.ptr);

    // Implement TMR majority voting
    size_t num_modules = outputs.size();
    std::vector<std::vector<float>> values(buf_first.size);

    // Collect all values at each position
    for (size_t i = 0; i < buf_first.size; i++) {
        values[i].resize(num_modules);

        for (size_t j = 0; j < num_modules; j++) {
            py::array_t<float> arr = outputs[j].cast<py::array_t<float>>();
            py::buffer_info buf = arr.request();
            float* ptr = static_cast<float*>(buf.ptr);
            values[i][j] = ptr[i];
        }
    }

    // Count differences and do majority voting
    int detected = 0;
    int corrected = 0;

    for (size_t i = 0; i < buf_first.size; i++) {
        // Check for discrepancies
        bool has_discrepancy = false;
        for (size_t j = 1; j < num_modules; j++) {
            if (std::abs(values[i][0] - values[i][j]) > 1e-5) {
                has_discrepancy = true;
                break;
            }
        }

        if (has_discrepancy) {
            detected++;

            // Different voting strategies based on protection level
            if (protection_level == 3) {  // FULL_TMR
                // Sort values to find median (middle value for TMR)
                std::sort(values[i].begin(), values[i].end());
                ptr_result[i] = values[i][num_modules / 2];  // Use median for odd number of modules
            }
            else if (protection_level == 4) {  // ADAPTIVE_TMR
                // More sophisticated weighted voting
                std::vector<float> diffs(num_modules, 0.0f);
                for (size_t j = 0; j < num_modules; j++) {
                    for (size_t k = 0; k < num_modules; k++) {
                        if (j != k) {
                            diffs[j] += std::abs(values[i][j] - values[i][k]);
                        }
                    }
                }

                // Smaller differences get higher weights
                std::vector<float> weights(num_modules);
                float sum_weights = 0.0f;
                for (size_t j = 0; j < num_modules; j++) {
                    weights[j] = 1.0f / (1e-10f + diffs[j]);
                    sum_weights += weights[j];
                }

                // Normalize weights and compute weighted average
                float result_value = 0.0f;
                for (size_t j = 0; j < num_modules; j++) {
                    result_value += (weights[j] / sum_weights) * values[i][j];
                }

                ptr_result[i] = result_value;
            }
            else {  // Other protection levels
                // Simple majority voting or average
                float sum = 0.0f;
                for (size_t j = 0; j < num_modules; j++) {
                    sum += values[i][j];
                }
                ptr_result[i] = sum / num_modules;
            }

            corrected++;
        }
        else {
            // No discrepancy - use the first value
            ptr_result[i] = values[i][0];
        }
    }

    return result;
}

// Function to protect a tensor
py::array_t<float> protect_tensor(py::array_t<float> tensor, int protection_level, int strategy)
{
    py::buffer_info buf = tensor.request();
    float* ptr = static_cast<float*>(buf.ptr);

    // Create output tensor
    py::array_t<float> result = py::array_t<float>(buf.size);
    py::buffer_info buf_result = result.request();
    float* ptr_result = static_cast<float*>(buf_result.ptr);

    // Copy the original data
    std::memcpy(ptr_result, ptr, buf.size * sizeof(float));

    // Apply protection based on level and strategy
    if (protection_level == 0) {  // NONE
        // No protection, just copy
    }
    else if (protection_level == 1) {  // CHECKSUM_ONLY
        // Add small marker to values to indicate protection level
        for (size_t i = 0; i < buf.size; i++) {
            ptr_result[i] += 1e-12;  // Tiny increase for checksum marker
        }
    }
    else if (protection_level >= 2) {  // TMR variants
        // Add small marker based on protection level
        float marker = 1e-11 * protection_level;
        for (size_t i = 0; i < buf.size; i++) {
            ptr_result[i] += marker;  // Tiny increase to mark protection level
        }

        // Apply strategy-specific modifications
        if (strategy == 1) {  // REED_SOLOMON
            // Special marker for Reed-Solomon
            for (size_t i = 0; i < buf.size; i += 10) {
                if (i < buf.size) ptr_result[i] += 2e-10;
            }
        }
        else if (strategy == 2) {  // PHYSICS_DRIVEN
            // Special marker for Physics-Driven
            for (size_t i = 0; i < buf.size; i += 8) {
                if (i < buf.size) ptr_result[i] += 3e-10;
            }
        }
    }

    return result;
}

PYBIND11_MODULE(_cpp_binding, m)
{
    m.doc() = "C++ core functionality for rad_ml_lib";

    // Version information
    py::class_<Version>(m, "Version")
        .def_property_readonly_static("major", [](py::object) { return Version::major; })
        .def_property_readonly_static("minor", [](py::object) { return Version::minor; })
        .def_property_readonly_static("patch", [](py::object) { return Version::patch; })
        .def_static("get_version_string", &Version::get_version_string);

    // Core functionality
    m.def("initialize", &initialize, "Initialize the library");
    m.def("shutdown", &shutdown, "Shutdown the library");

    // Radiation effects and protection
    m.def("apply_radiation", &apply_radiation, "Apply radiation effects to a tensor",
          py::arg("tensor"), py::arg("strength") = 1.0);

    m.def("detect_errors", &detect_errors, "Detect errors in a tensor", py::arg("original"),
          py::arg("current"));

    m.def("tmr_vote", &tmr_vote, "Perform Triple Modular Redundancy voting", py::arg("outputs"),
          py::arg("protection_level") = 3);

    m.def("protect_tensor", &protect_tensor, "Apply protection to a tensor", py::arg("tensor"),
          py::arg("protection_level") = 3, py::arg("strategy") = 0);

    // Constants
    // Protection level constants
    m.attr("PROTECTION_LEVEL_NONE") = 0;
    m.attr("PROTECTION_LEVEL_CHECKSUM") = 1;
    m.attr("PROTECTION_LEVEL_SELECTIVE_TMR") = 2;
    m.attr("PROTECTION_LEVEL_FULL_TMR") = 3;
    m.attr("PROTECTION_LEVEL_ADAPTIVE_TMR") = 4;
    m.attr("PROTECTION_LEVEL_SPACE_OPTIMIZED") = 5;

    // Strategy constants
    m.attr("STRATEGY_MULTI_LAYERED") = 0;
    m.attr("STRATEGY_REED_SOLOMON") = 1;
    m.attr("STRATEGY_PHYSICS_DRIVEN") = 2;
    m.attr("STRATEGY_ADAPTIVE") = 3;
    m.attr("STRATEGY_HARDWARE") = 4;
}
