#pragma once

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rad_ml {
namespace physics {

struct OmereLetPoint {
    double let_mev_cm2_mg = 0;
    double integral_flux_cm2_s = 0;
    double differential_flux_per_cm2_s_per_let = 0;
};

struct OmereLetSpectrum {
    std::string version;
    std::string model;
    std::string solar_activity;
    std::string cutoff_model;
    std::string orbit_name;
    double mission_duration_years = 0;
    double shielding_g_cm2 = 0;
    std::vector<OmereLetPoint> points;
};

struct OmereSeeResult {
    std::string version;
    std::string let_file;
    std::string cross_section_type;
    double cell_depth_um = 0;
    double let_threshold_mev_cm2_mg = 0;
    double saturation_cross_section_cm2_per_bit = 0;
    double weibull_width = 0;
    double weibull_shape = 0;
    double heavy_ion_rate_per_bit_day = 0;
};

namespace omere_detail {

inline std::string trim(const std::string& value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

inline bool startsWith(const std::string& value, const std::string& prefix)
{
    return value.compare(0, prefix.size(), prefix) == 0;
}

inline std::string textAfterColon(const std::string& line)
{
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
        throw std::runtime_error("Malformed OMERE metadata line: " + line);
    }
    return trim(line.substr(colon + 1));
}

inline double numberAfterColon(const std::string& line)
{
    std::istringstream stream(textAfterColon(line));
    double value = 0;
    if (!(stream >> value) || !std::isfinite(value)) {
        throw std::runtime_error("Malformed OMERE numeric metadata: " + line);
    }
    return value;
}

inline double numberAfterPrefix(const std::string& line, const std::string& prefix)
{
    std::istringstream stream(trim(line.substr(prefix.size())));
    double value = 0;
    if (!(stream >> value) || !std::isfinite(value)) {
        throw std::runtime_error("Malformed OMERE numeric result: " + line);
    }
    return value;
}

inline std::string parseVersion(const std::string& line)
{
    constexpr const char* marker = "# OMERE ";
    if (!startsWith(line, marker)) return {};
    const std::string remainder = line.substr(std::char_traits<char>::length(marker));
    return remainder.substr(0, remainder.find_first_of(" \t"));
}

}  // namespace omere_detail

inline OmereLetSpectrum loadOmereLetSpectrum(const std::string& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Unable to open OMERE LET file: " + path);

    OmereLetSpectrum spectrum;
    bool in_energy_spectrum = false;
    bool found_let_units = false;
    double let_scale_to_mg = 0;
    std::string line;
    while (std::getline(input, line)) {
        if (spectrum.version.empty()) {
            spectrum.version = omere_detail::parseVersion(line);
        }
        if (omere_detail::startsWith(line, "# Shielding thickness applied:")) {
            spectrum.shielding_g_cm2 = omere_detail::numberAfterColon(line);
        } else if (omere_detail::startsWith(line, "# Model :")) {
            spectrum.model = omere_detail::textAfterColon(line);
        } else if (omere_detail::startsWith(line, "# Solar activity :")) {
            spectrum.solar_activity = omere_detail::textAfterColon(line);
        } else if (omere_detail::startsWith(line, "# Cutoff model :")) {
            spectrum.cutoff_model = omere_detail::textAfterColon(line);
        } else if (omere_detail::startsWith(line, "# Name :")) {
            spectrum.orbit_name = omere_detail::textAfterColon(line);
        } else if (omere_detail::startsWith(line, "# Duration :")) {
            spectrum.mission_duration_years = omere_detail::numberAfterColon(line);
        } else if (omere_detail::startsWith(line, "# Energy spectrum :")) {
            in_energy_spectrum = true;
        } else if (in_energy_spectrum && omere_detail::startsWith(line, "#MeV.cm2.")) {
            if (line.find("mg-1") != std::string::npos) {
                let_scale_to_mg = 1.0;
            } else if (line.find("g-1") != std::string::npos) {
                // OMERE exports LET in MeV.cm^2/g; SEE cross-sections use
                // MeV.cm^2/mg, so the numerical LET is divided by 1000.
                let_scale_to_mg = 1.0e-3;
            } else {
                throw std::runtime_error("Unsupported OMERE LET unit header");
            }
            found_let_units = true;
        } else if (in_energy_spectrum && !line.empty() && line.front() != '#') {
            std::istringstream stream(line);
            double raw_let = 0;
            double integral_flux = 0;
            double raw_differential_flux = 0;
            if (stream >> raw_let >> integral_flux >> raw_differential_flux) {
                if (!found_let_units) {
                    throw std::runtime_error("OMERE LET data appeared before its unit header");
                }
                spectrum.points.push_back(
                    {raw_let * let_scale_to_mg, integral_flux,
                     raw_differential_flux / let_scale_to_mg});
            }
        }
    }

    if (spectrum.version.empty() || spectrum.model.empty() || spectrum.points.size() < 2) {
        throw std::runtime_error("OMERE LET file is missing required metadata or spectrum data");
    }
    for (std::size_t i = 0; i < spectrum.points.size(); ++i) {
        const auto& point = spectrum.points[i];
        if (!std::isfinite(point.let_mev_cm2_mg) || point.let_mev_cm2_mg < 0 ||
            !std::isfinite(point.integral_flux_cm2_s) || point.integral_flux_cm2_s < 0 ||
            !std::isfinite(point.differential_flux_per_cm2_s_per_let) ||
            point.differential_flux_per_cm2_s_per_let < 0 ||
            (i > 0 && point.let_mev_cm2_mg <= spectrum.points[i - 1].let_mev_cm2_mg)) {
            throw std::runtime_error("Invalid OMERE LET spectrum point ordering or value");
        }
    }
    return spectrum;
}

inline OmereSeeResult loadOmereSeeResult(const std::string& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Unable to open OMERE SEE file: " + path);

    OmereSeeResult result;
    std::string line;
    while (std::getline(input, line)) {
        if (result.version.empty()) result.version = omere_detail::parseVersion(line);
        if (omere_detail::startsWith(line, "# LET File:")) {
            result.let_file = omere_detail::textAfterColon(line);
        } else if (omere_detail::startsWith(line, "# Cell depth :")) {
            result.cell_depth_um = omere_detail::numberAfterColon(line);
        } else if (omere_detail::startsWith(line, "# Heavy ions let threshold :")) {
            result.let_threshold_mev_cm2_mg = omere_detail::numberAfterColon(line);
        } else if (omere_detail::startsWith(line, "# Heavy ions limit cross section :")) {
            result.saturation_cross_section_cm2_per_bit =
                omere_detail::numberAfterColon(line);
        } else if (omere_detail::startsWith(line, "# Weibull W :")) {
            result.weibull_width = omere_detail::numberAfterColon(line);
        } else if (omere_detail::startsWith(line, "# Weibull S :")) {
            result.weibull_shape = omere_detail::numberAfterColon(line);
        } else if (omere_detail::startsWith(line, "# Cross section type:")) {
            result.cross_section_type = omere_detail::textAfterColon(line);
        } else if (omere_detail::startsWith(line, "# Heavy ions rate ")) {
            result.heavy_ion_rate_per_bit_day =
                omere_detail::numberAfterPrefix(line, "# Heavy ions rate ");
        }
    }

    if (result.version.empty() || result.let_file.empty() || result.cell_depth_um <= 0 ||
        result.saturation_cross_section_cm2_per_bit <= 0 || result.weibull_width <= 0 ||
        result.weibull_shape <= 0 || result.heavy_ion_rate_per_bit_day <= 0) {
        throw std::runtime_error("OMERE SEE file is missing required metadata or result values");
    }
    return result;
}

}  // namespace physics
}  // namespace rad_ml
