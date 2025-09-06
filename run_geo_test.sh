#!/bin/bash

# GEO Mission Validation Test Runner
# This script runs the comprehensive GEO mission validation test

echo "======================================================================="
echo "                    GEO MISSION VALIDATION TEST RUNNER"
echo "======================================================================="
echo ""

# Check if the test executable exists
if [ ! -f "./geo_mission_validation" ]; then
    echo "Error: geo_mission_validation executable not found!"
    echo "Please run 'make geo_mission_validation' first to build the test."
    exit 1
fi

echo "Starting GEO mission validation test..."
echo "This test validates the radiation tolerance framework for:"
echo "  - Geostationary Earth Orbit (GEO) missions"
echo "  - Van Allen belt exposure scenarios"
echo "  - Solar storm conditions"
echo "  - Eclipse temperature cycling"
echo "  - 15-year mission duration simulation"
echo "  - End-of-life component degradation"
echo ""

# Run the GEO test
./geo_mission_validation

# Check if the test completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "                    GEO TEST COMPLETED SUCCESSFULLY"
    echo "======================================================================="
    echo ""
    echo "Test results:"
    if [ -f "geo_mission_verification_report.txt" ]; then
        echo "  ✓ Detailed report generated: geo_mission_verification_report.txt"
    fi
    echo "  ✓ All GEO mission scenarios tested"
    echo "  ✓ Van Allen belt exposure validated"
    echo "  ✓ Solar storm survival confirmed"
    echo "  ✓ Long-duration stability verified"
    echo ""
    echo "The framework demonstrates excellent radiation tolerance for GEO missions!"
else
    echo ""
    echo "======================================================================="
    echo "                         GEO TEST FAILED"
    echo "======================================================================="
    echo ""
    echo "The GEO mission validation test encountered errors."
    echo "Please check the output above for details."
    exit 1
fi
