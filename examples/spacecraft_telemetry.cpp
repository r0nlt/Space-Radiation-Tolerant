#include <chrono>
#include <cmath>
#include <iostream>
#include <space_containers/fixed_map.hpp>
#include <space_containers/fixed_vector.hpp>
#include <space_containers/radiation_hardened_vector.hpp>
#include <thread>

// Example spacecraft sensor data structure
struct SensorData {
    double temperature;
    double pressure;
    double radiation;
    std::chrono::system_clock::time_point timestamp;

    SensorData() = default;
    SensorData(double t, double p, double r)
        : temperature(t), pressure(p), radiation(r), timestamp(std::chrono::system_clock::now())
    {
    }
};

// Example spacecraft telemetry system
class SpacecraftTelemetry {
   public:
    // Fixed-size storage for different types of data
    static constexpr std::size_t MAX_SENSOR_READINGS = 1000;
    static constexpr std::size_t MAX_CRITICAL_READINGS = 100;
    static constexpr std::size_t MAX_CALIBRATION_POINTS = 50;

    using SensorBuffer = space_containers::FixedVector<SensorData, MAX_SENSOR_READINGS>;
    using CriticalBuffer =
        space_containers::RadiationHardenedVector<SensorData, MAX_CRITICAL_READINGS>;
    using CalibrationMap = space_containers::FixedMap<std::string, double, MAX_CALIBRATION_POINTS>;

    SpacecraftTelemetry()
    {
        // Initialize calibration data
        calibration_data_.insert("temp_offset", 0.5);
        calibration_data_.insert("pressure_scale", 1.02);
        calibration_data_.insert("radiation_factor", 1.15);
    }

    // Record regular sensor reading
    bool record_sensor_reading(double temp, double pressure, double radiation)
    {
        SensorData data(temp, pressure, radiation);

        // Apply calibration
        data.temperature += calibration_data_.find("temp_offset").value_or(0.0);
        data.pressure *= calibration_data_.find("pressure_scale").value_or(1.0);
        data.radiation *= calibration_data_.find("radiation_factor").value_or(1.0);

        return sensor_buffer_.push_back(data);
    }

    // Record critical reading with radiation protection
    bool record_critical_reading(double temp, double pressure, double radiation)
    {
        SensorData data(temp, pressure, radiation);
        return critical_buffer_.push_back(data);
    }

    // Update calibration value
    bool update_calibration(const std::string& sensor, double value)
    {
        return calibration_data_.insert(sensor, value);
    }

    // Get latest sensor reading
    std::optional<SensorData> get_latest_reading() const
    {
        if (sensor_buffer_.empty()) {
            return std::nullopt;
        }
        return sensor_buffer_[sensor_buffer_.size() - 1];
    }

    // Get latest critical reading with validation
    std::optional<SensorData> get_latest_critical_reading()
    {
        if (critical_buffer_.empty()) {
            return std::nullopt;
        }

        // Validate data before returning
        critical_buffer_.validate_and_correct();
        return critical_buffer_[critical_buffer_.size() - 1];
    }

    // Print telemetry statistics
    void print_statistics() const
    {
        std::cout << "\nTelemetry Statistics:\n";
        std::cout << "Regular readings: " << sensor_buffer_.size() << "/" << MAX_SENSOR_READINGS
                  << "\n";
        std::cout << "Critical readings: " << critical_buffer_.size() << "/"
                  << MAX_CRITICAL_READINGS << "\n";
        std::cout << "Calibration points: " << calibration_data_.size() << "/"
                  << MAX_CALIBRATION_POINTS << "\n";

        std::cout << "\nAccess Statistics:\n";
        std::cout << "Regular buffer accesses: " << sensor_buffer_.get_access_count() << "\n";
        std::cout << "Regular buffer errors: " << sensor_buffer_.get_error_count() << "\n";
        std::cout << "Critical buffer errors: " << critical_buffer_.get_error_count() << "\n";
        std::cout << "Critical buffer corrections: " << critical_buffer_.get_correction_count()
                  << "\n";
        std::cout << "Calibration map lookups: " << calibration_data_.get_find_count() << "\n";
    }

   private:
    SensorBuffer sensor_buffer_;
    CriticalBuffer critical_buffer_;
    CalibrationMap calibration_data_;
};

// Simulate sensor readings
class SensorSimulator {
   public:
    static SensorData generate_reading()
    {
        static double time = 0.0;
        time += 0.1;

        // Generate simulated sensor data with some noise
        double temp = 20.0 + 5.0 * std::sin(time) + (std::rand() % 100) / 100.0;
        double pressure = 1000.0 + 50.0 * std::cos(time) + (std::rand() % 100) / 100.0;
        double radiation = 0.5 + 0.2 * std::sin(2.0 * time) + (std::rand() % 100) / 1000.0;

        return SensorData(temp, pressure, radiation);
    }
};

int main()
{
    SpacecraftTelemetry telemetry;

    std::cout << "Starting spacecraft telemetry simulation...\n";

    // Simulate sensor readings for 10 seconds
    for (int i = 0; i < 100; ++i) {
        auto reading = SensorSimulator::generate_reading();

        // Record regular reading
        telemetry.record_sensor_reading(reading.temperature, reading.pressure, reading.radiation);

        // Record critical reading every 10th sample
        if (i % 10 == 0) {
            telemetry.record_critical_reading(reading.temperature, reading.pressure,
                                              reading.radiation);
        }

        // Update calibration occasionally
        if (i % 25 == 0) {
            telemetry.update_calibration("temp_offset", 0.5 + (std::rand() % 100) / 1000.0);
        }

        // Print latest readings periodically
        if (i % 20 == 0) {
            std::cout << "\nLatest Readings:\n";

            if (auto regular = telemetry.get_latest_reading()) {
                std::cout << "Regular - Temp: " << regular->temperature
                          << ", Pressure: " << regular->pressure
                          << ", Radiation: " << regular->radiation << "\n";
            }

            if (auto critical = telemetry.get_latest_critical_reading()) {
                std::cout << "Critical - Temp: " << critical->temperature
                          << ", Pressure: " << critical->pressure
                          << ", Radiation: " << critical->radiation << "\n";
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Print final statistics
    telemetry.print_statistics();

    return 0;
}
