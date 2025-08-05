/**
 * rad_ml_core.cpp - Python bindings for the rad_ml framework
 *
 * This file implements Python bindings for the C++ rad_ml framework
 * using pybind11.
 *
 * Author: Rishab Nuguru
 * Copyright: © 2025 Rishab Nuguru
 * License: AGPL v3 license
 */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Core headers from rad_ml
#include <rad_ml/api/rad_ml.hpp>
// Temporarily comment out pytorch integration to avoid redefinitions
// #include <rad_ml/pytorch/pytorch_integration.hpp>

namespace py = pybind11;
using namespace rad_ml;

// Shorthand for making properties
template <typename T, typename... Args>
void def_property_readonly(py::class_<T>& c, const char* name, Args&&... args)
{
    c.def_property_readonly(name, std::forward<Args>(args)...);
}

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Radiation-Tolerant Machine Learning Framework - Python Bindings";

    // Version information
    py::class_<Version> version(m, "Version");
    version.def_readonly_static("major", &Version::major)
        .def_readonly_static("minor", &Version::minor)
        .def_readonly_static("patch", &Version::patch)
        .def_static("as_string", &Version::asString);

    // Core functions
    m.def("initialize", &initialize, py::arg("enable_logging") = true,
          py::arg("memory_protection_level") = memory::MemoryProtectionLevel::NONE,
          "Initialize the rad_ml framework");

    m.def("shutdown", &shutdown, py::arg("check_for_leaks") = true,
          "Shutdown the rad_ml framework and perform cleanup");

    // Enum: MemoryProtectionLevel
    py::enum_<memory::MemoryProtectionLevel>(m, "MemoryProtectionLevel")
        .value("NONE", memory::MemoryProtectionLevel::NONE)
        .value("MINIMAL", memory::MemoryProtectionLevel::MINIMAL)
        .value("MODERATE", memory::MemoryProtectionLevel::MODERATE)
        .value("HIGH", memory::MemoryProtectionLevel::HIGH)
        .value("VERY_HIGH", memory::MemoryProtectionLevel::VERY_HIGH)
        .value("ADAPTIVE", memory::MemoryProtectionLevel::ADAPTIVE)
        .export_values();

    // Enum: ProtectionLevel
    py::enum_<neural::ProtectionLevel>(m, "ProtectionLevel")
        .value("NONE", neural::ProtectionLevel::NONE)
        .value("MINIMAL", neural::ProtectionLevel::MINIMAL)
        .value("MODERATE", neural::ProtectionLevel::MODERATE)
        .value("HIGH", neural::ProtectionLevel::HIGH)
        .value("VERY_HIGH", neural::ProtectionLevel::VERY_HIGH)
        .value("ADAPTIVE", neural::ProtectionLevel::ADAPTIVE)
        .export_values();

    // Enum: HardeningStrategy
    py::enum_<neural::HardeningStrategy>(m, "HardeningStrategy")
        .value("ALL_LAYERS", neural::HardeningStrategy::ALL_LAYERS)
        .value("CRITICAL_LAYERS", neural::HardeningStrategy::CRITICAL_LAYERS)
        .value("WEIGHT_THRESHOLD", neural::HardeningStrategy::WEIGHT_THRESHOLD)
        .value("GRADIENT_BASED", neural::HardeningStrategy::GRADIENT_BASED)
        .value("ADAPTIVE", neural::HardeningStrategy::ADAPTIVE)
        .export_values();

    // Radiation environment enum
    py::enum_<sim::RadiationEnvironment>(m, "RadiationEnvironment")
        .value("LEO", sim::RadiationEnvironment::LEO)
        .value("EARTH_ORBIT", sim::RadiationEnvironment::EARTH_ORBIT)
        .value("MEO", sim::RadiationEnvironment::MEO)
        .value("GEO", sim::RadiationEnvironment::GEO)
        .value("LUNAR", sim::RadiationEnvironment::LUNAR)
        .value("MARS", sim::RadiationEnvironment::MARS)
        .value("MARS_ORBIT", sim::RadiationEnvironment::MARS_ORBIT)
        .value("MARS_SURFACE", sim::RadiationEnvironment::MARS_SURFACE)
        .value("JUPITER", sim::RadiationEnvironment::JUPITER)
        .value("EUROPA", sim::RadiationEnvironment::EUROPA)
        .value("INTERPLANETARY", sim::RadiationEnvironment::INTERPLANETARY)
        .value("SOLAR_PROBE", sim::RadiationEnvironment::SOLAR_PROBE)
        .value("SOLAR_MINIMUM", sim::RadiationEnvironment::SOLAR_MINIMUM)
        .value("SOLAR_MAXIMUM", sim::RadiationEnvironment::SOLAR_MAXIMUM)
        .value("SOLAR_STORM", sim::RadiationEnvironment::SOLAR_STORM);

    // For now, skip MissionType enum binding
    // py::enum_<mission::MissionType>(m, "MissionType")
    //     .value("LEO_MISSION", mission::MissionType::LEO_MISSION)
    //     .value("MARS_MISSION", mission::MissionType::MARS_MISSION)
    //     .value("JUPITER_MISSION", mission::MissionType::JUPITER_MISSION);

    // Enum: ErrorSeverity
    py::enum_<error::ErrorSeverity>(m, "ErrorSeverity")
        .value("TRACE", error::ErrorSeverity::TRACE)
        .value("DEBUG", error::ErrorSeverity::DEBUG)
        .value("INFO", error::ErrorSeverity::INFO)
        .value("WARNING", error::ErrorSeverity::WARNING)
        .value("ERROR", error::ErrorSeverity::ERROR)
        .value("CRITICAL", error::ErrorSeverity::CRITICAL)
        .export_values();

    // TMR template classes (for common numeric types)
    // Since these are templates, we need to explicitly instantiate for Python-friendly types

    // Define the StandardTMR class for integers
    py::class_<tmr_types::StandardTMR<int>>(m, "StandardTMRInt")
        .def(py::init<>())
        .def(py::init<int>())
        .def("get_value", &tmr_types::StandardTMR<int>::getValue)
        .def("set_value", &tmr_types::StandardTMR<int>::setValue)
        .def("correct", &tmr_types::StandardTMR<int>::correct)
        .def("check_integrity", &tmr_types::StandardTMR<int>::checkIntegrity);

    // Define the StandardTMR class for floats
    py::class_<tmr_types::StandardTMR<float>>(m, "StandardTMRFloat")
        .def(py::init<>())
        .def(py::init<float>())
        .def("get_value", &tmr_types::StandardTMR<float>::getValue)
        .def("set_value", &tmr_types::StandardTMR<float>::setValue)
        .def("correct", &tmr_types::StandardTMR<float>::correct)
        .def("check_integrity", &tmr_types::StandardTMR<float>::checkIntegrity);

    // Define the StandardTMR class for doubles
    py::class_<tmr_types::StandardTMR<double>>(m, "StandardTMRDouble")
        .def(py::init<>())
        .def(py::init<double>())
        .def("get_value", &tmr_types::StandardTMR<double>::getValue)
        .def("set_value", &tmr_types::StandardTMR<double>::setValue)
        .def("correct", &tmr_types::StandardTMR<double>::correct)
        .def("check_integrity", &tmr_types::StandardTMR<double>::checkIntegrity);

    // TMR classes
    py::class_<tmr::TMR<int>, std::shared_ptr<tmr::TMR<int>>>(m, "TMRInt")
        .def(py::init<int>())
        .def("get_value", &tmr::TMR<int>::getValue)
        .def("set_value", &tmr::TMR<int>::setValue)
        .def("correct", &tmr::TMR<int>::correct)
        .def("check_integrity", &tmr::TMR<int>::checkIntegrity);

    py::class_<tmr::TMR<float>, std::shared_ptr<tmr::TMR<float>>>(m, "TMRFloat")
        .def(py::init<float>())
        .def("get_value", &tmr::TMR<float>::getValue)
        .def("set_value", &tmr::TMR<float>::setValue)
        .def("correct", &tmr::TMR<float>::correct)
        .def("check_integrity", &tmr::TMR<float>::checkIntegrity);

    py::class_<tmr::TMR<double>, std::shared_ptr<tmr::TMR<double>>>(m, "TMRDouble")
        .def(py::init<double>())
        .def("get_value", &tmr::TMR<double>::getValue)
        .def("set_value", &tmr::TMR<double>::setValue)
        .def("correct", &tmr::TMR<double>::correct)
        .def("check_integrity", &tmr::TMR<double>::checkIntegrity);

    // For now, skip TMRBase binding due to template issues
    // py::class_<tmr::TMRBase<int>, std::shared_ptr<tmr::TMRBase<int>>>(m, "TMRBaseInt")

    // TMR factory functions
    m.def("create_standard_tmr_int", &make_tmr::standard<int>, py::arg("initial_value") = 0);
    m.def("create_standard_tmr_float", &make_tmr::standard<float>, py::arg("initial_value") = 0.0f);
    m.def("create_standard_tmr_double", &make_tmr::standard<double>,
          py::arg("initial_value") = 0.0);

    m.def("create_enhanced_tmr_int", &make_tmr::enhanced<int>, py::arg("initial_value") = 0);
    m.def("create_enhanced_tmr_float", &make_tmr::enhanced<float>, py::arg("initial_value") = 0.0f);
    m.def("create_enhanced_tmr_double", &make_tmr::enhanced<double>,
          py::arg("initial_value") = 0.0);

    // Simulation classes
    // Physics radiation simulator
    py::class_<sim::PhysicsRadiationSimulator>(m, "PhysicsRadiationSimulator")
        .def(py::init<const sim::EnvironmentParams&>())
        .def("set_environment", &sim::PhysicsRadiationSimulator::setEnvironment)
        .def("set_solar_activity", &sim::PhysicsRadiationSimulator::set_solar_activity)
        .def("get_environment", &sim::PhysicsRadiationSimulator::getEnvironment)
        .def("get_intensity", &sim::PhysicsRadiationSimulator::getIntensity)
        .def("simulate", &sim::PhysicsRadiationSimulator::simulate);

    // Mission simulator (simplified)
    py::class_<testing::MissionSimulator>(m, "MissionSimulator")
        .def(py::init<const testing::MissionProfile&, const testing::AdaptiveProtectionConfig&>(),
             py::arg("profile"), py::arg("protection_config") = testing::AdaptiveProtectionConfig{})
        .def("configure_mission",
             [](testing::MissionSimulator& sim, const std::string& mission_type,
                size_t duration_days) {
                 // Simplified implementation
             })
        .def("get_mission_type",
             [](const testing::MissionSimulator& sim) {
                 return std::string("LEO");  // Default
             })
        .def("get_duration_days",
             [](const testing::MissionSimulator& sim) {
                 return size_t(30);  // Default
             })
        .def("get_results", [](const testing::MissionSimulator& sim) {
            return std::string("No results available");
        });

    // Fault injector (simplified)
    py::class_<testing::FaultInjector>(m, "FaultInjector")
        .def(py::init<>())
        .def("inject_fault",
             [](testing::FaultInjector& injector) {
                 // Simplified implementation
                 return true;
             })
        .def("set_fault_rate",
             [](testing::FaultInjector& injector, double rate) {
                 // Simplified implementation
             })
        .def("get_fault_rate",
             [](const testing::FaultInjector& injector) {
                 return 0.01;  // Default
             })
        .def("get_total_faults", [](const testing::FaultInjector& injector) {
            return size_t(0);  // Default
        });

    // Factory functions for simulators
    m.def("create_radiation_simulator", &simulation::createRadiationSimulator,
          py::arg("environment") = sim::RadiationEnvironment::EARTH_ORBIT,
          py::arg("intensity") = 0.5);

    m.def("create_mission_simulator", &simulation::createMissionSimulator, py::arg("mission_type"),
          py::arg("duration_days") = 30);

    m.def("create_fault_injector", &simulation::createFaultInjector, py::arg("fault_rate") = 0.01);

    // Neural network classes
    // Error predictor (with template parameter)
    py::class_<neural::ErrorPredictor<double>>(m, "ErrorPredictor")
        .def(py::init<>())
        .def("predict", [](const neural::ErrorPredictor<double>& predictor, double input) {
            return 0.0;  // Default prediction
        });

    // PyTorch Integration (temporarily disabled due to redefinition issues)
    /*
    py::class_<rad_ml::pytorch::PyTorchConfig>(m, "PyTorchConfig")
        .def(py::init<>())
        .def_readwrite("enable_tmr_protection",
    &rad_ml::pytorch::PyTorchConfig::enable_tmr_protection)
        .def_readwrite("enable_radiation_hardening",
                       &rad_ml::pytorch::PyTorchConfig::enable_radiation_hardening)
        .def_readwrite("protection_level", &rad_ml::pytorch::PyTorchConfig::protection_level)
        .def_readwrite("tmr_strategy", &rad_ml::pytorch::PyTorchConfig::tmr_strategy)
        .def_readwrite("use_cuda_if_available",
    &rad_ml::pytorch::PyTorchConfig::use_cuda_if_available)
        .def_readwrite("enable_gradient_protection",
                       &rad_ml::pytorch::PyTorchConfig::enable_gradient_protection)
        .def_readwrite("enable_weight_protection",
                       &rad_ml::pytorch::PyTorchConfig::enable_weight_protection);

    py::class_<rad_ml::pytorch::PyTorchIntegration>(m, "PyTorchIntegration")
        .def_static("get_instance", &rad_ml::pytorch::PyTorchIntegration::get_instance,
                   py::return_value_policy::reference)
        .def("initialize", &rad_ml::pytorch::PyTorchIntegration::initialize,
             py::arg("config") = rad_ml::pytorch::PyTorchConfig{})
        .def("shutdown", &rad_ml::pytorch::PyTorchIntegration::shutdown)
        .def("get_config", &rad_ml::pytorch::PyTorchIntegration::get_config,
             py::return_value_policy::reference)
        .def("update_config", &rad_ml::pytorch::PyTorchIntegration::update_config);
    */

    // PyTorch utility functions (temporarily disabled)
    /*
    m.def(
        "create_pytorch_integration",
        []() -> rad_ml::pytorch::PyTorchIntegration& {
            return rad_ml::pytorch::PyTorchIntegration::get_instance();
        },
        py::return_value_policy::reference,
        "Create a PyTorch integration instance");

    m.def("is_pytorch_enabled", []() { return true; }, "Check if PyTorch is enabled");
    */
}
