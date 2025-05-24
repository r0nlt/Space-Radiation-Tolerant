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

namespace py = pybind11;
using namespace rad_ml;

// Shorthand for making properties
template <typename T, typename... Args>
void def_property_readonly(py::class_<T> &c, const char *name, Args &&...args)
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

    // Enum: RadiationEnvironment
    py::enum_<sim::RadiationEnvironment>(m, "RadiationEnvironment")
        .value("EARTH_ORBIT", sim::RadiationEnvironment::EARTH_ORBIT)
        .value("LEO", sim::RadiationEnvironment::LEO)
        .value("GEO", sim::RadiationEnvironment::GEO)
        .value("LUNAR", sim::RadiationEnvironment::LUNAR)
        .value("MARS", sim::RadiationEnvironment::MARS)
        .value("JUPITER", sim::RadiationEnvironment::JUPITER)
        .value("SOLAR_PROBE", sim::RadiationEnvironment::SOLAR_PROBE)
        .export_values();

    // Enum: MissionType
    py::enum_<mission::MissionType>(m, "MissionType")
        .value("LEO_EARTH_OBSERVATION", mission::MissionType::LEO_EARTH_OBSERVATION)
        .value("MEDIUM_EARTH_ORBIT", mission::MissionType::MEDIUM_EARTH_ORBIT)
        .value("GEOSTATIONARY", mission::MissionType::GEOSTATIONARY)
        .value("LUNAR_ORBIT", mission::MissionType::LUNAR_ORBIT)
        .value("LUNAR_SURFACE", mission::MissionType::LUNAR_SURFACE)
        .value("MARS_TRANSIT", mission::MissionType::MARS_TRANSIT)
        .value("MARS_ORBIT", mission::MissionType::MARS_ORBIT)
        .value("MARS_SURFACE", mission::MissionType::MARS_SURFACE)
        .value("DEEP_SPACE", mission::MissionType::DEEP_SPACE)
        .export_values();

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

    // Base trampoline class for StandardTMR to enable Python subclassing
    py::class_<tmr::TMRBase<int>, std::shared_ptr<tmr::TMRBase<int>>>(m, "TMRBaseInt")
        .def(py::init<>());

    py::class_<tmr::TMRBase<float>, std::shared_ptr<tmr::TMRBase<float>>>(m, "TMRBaseFloat")
        .def(py::init<>());

    py::class_<tmr::TMRBase<double>, std::shared_ptr<tmr::TMRBase<double>>>(m, "TMRBaseDouble")
        .def(py::init<>());

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
    // PhysicsRadiationSimulator
    py::class_<sim::PhysicsRadiationSimulator>(m, "PhysicsRadiationSimulator")
        .def(py::init<sim::RadiationEnvironment, double>(),
             py::arg("environment") = sim::RadiationEnvironment::EARTH_ORBIT,
             py::arg("intensity") = 0.5)
        .def("set_environment", &sim::PhysicsRadiationSimulator::setEnvironment)
        .def("set_intensity", &sim::PhysicsRadiationSimulator::setIntensity)
        .def("get_environment", &sim::PhysicsRadiationSimulator::getEnvironment)
        .def("get_intensity", &sim::PhysicsRadiationSimulator::getIntensity)
        .def("simulate", &sim::PhysicsRadiationSimulator::simulate);

    // MissionSimulator
    py::class_<testing::MissionSimulator>(m, "MissionSimulator")
        .def(py::init<mission::MissionType, size_t>(), py::arg("mission_type"),
             py::arg("duration_days") = 30)
        .def("run_simulation", &testing::MissionSimulator::runSimulation)
        .def("get_mission_type", &testing::MissionSimulator::getMissionType)
        .def("get_duration_days", &testing::MissionSimulator::getDurationDays)
        .def("get_results", &testing::MissionSimulator::getResults);

    // FaultInjector
    py::class_<testing::FaultInjector>(m, "FaultInjector")
        .def(py::init<double>(), py::arg("fault_rate") = 0.01)
        .def("inject_fault", &testing::FaultInjector::injectFault)
        .def("set_fault_rate", &testing::FaultInjector::setFaultRate)
        .def("get_fault_rate", &testing::FaultInjector::getFaultRate)
        .def("get_total_faults", &testing::FaultInjector::getTotalFaults);

    // Factory functions for simulators
    m.def("create_radiation_simulator", &simulation::createRadiationSimulator,
          py::arg("environment") = sim::RadiationEnvironment::EARTH_ORBIT,
          py::arg("intensity") = 0.5);

    m.def("create_mission_simulator", &simulation::createMissionSimulator, py::arg("mission_type"),
          py::arg("duration_days") = 30);

    m.def("create_fault_injector", &simulation::createFaultInjector, py::arg("fault_rate") = 0.01);

    // Neural network classes
    // ErrorPredictor
    py::class_<neural::ErrorPredictor>(m, "ErrorPredictor")
        .def(py::init<const std::string &>(), py::arg("model_path") = "")
        .def("train", &neural::ErrorPredictor::train)
        .def("predict", &neural::ErrorPredictor::predict)
        .def("load_model", &neural::ErrorPredictor::loadModel)
        .def("save_model", &neural::ErrorPredictor::saveModel);
}
