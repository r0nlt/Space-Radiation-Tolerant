#!/bin/bash

# Run NASA/ESA Standard Radiation Test Script
# This script builds and runs the NASA/ESA standardized radiation test
# for the Radiation-Tolerant ML Framework

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Set working directory to the script location
cd "$(dirname "$0")"
WORKSPACE_DIR="$(pwd)"

echo -e "${BLUE}${BOLD}NASA/ESA Standard Radiation Test${NC}"
echo -e "-----------------------------------------------"
echo -e "Starting test procedure following standardized NASA and ESA protocols"
echo -e "for radiation-tolerant system validation."
echo

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo -e "${YELLOW}Creating build directory...${NC}"
    mkdir -p build
fi

# Enter build directory
cd build

# Configure and build the project
echo -e "${YELLOW}Configuring project with CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

echo -e "${YELLOW}Building the NASA/ESA standard test...${NC}"
make nasa_esa_standard_test

# Check if build was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}${BOLD}Error: Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Build successful${NC}"
echo

# Run the NASA/ESA standard test
echo -e "${YELLOW}Running NASA/ESA Standard Radiation Test...${NC}"
echo -e "This may take several minutes due to statistical validation with Monte Carlo simulations."
echo

# Set any needed environment variables
export RAD_ML_MONTE_CARLO_TRIALS=25000
export RAD_ML_ENVIRONMENT_MODEL=CREME96
export RAD_ML_CONFIDENCE_LEVEL=0.95

# Run the test
./nasa_esa_standard_test

# Check if test was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}${BOLD}Error: Test failed!${NC}"
    exit 1
fi

echo -e "${GREEN}${BOLD}NASA/ESA Standard Radiation Test complete.${NC}"
echo -e "See results in nasa_esa_test_results.csv and nasa_esa_test_report.html"
echo

# Display file locations
if [ -f "nasa_esa_test_report.html" ]; then
    echo -e "HTML Report: ${WORKSPACE_DIR}/build/nasa_esa_test_report.html"
fi

if [ -f "nasa_esa_test_results.csv" ]; then
    echo -e "CSV Results: ${WORKSPACE_DIR}/build/nasa_esa_test_results.csv"
fi

if [ -f "nasa_esa_comprehensive_report.html" ]; then
    echo -e "Comprehensive Report: ${WORKSPACE_DIR}/build/nasa_esa_comprehensive_report.html"
fi

echo
echo -e "${BLUE}${BOLD}Test procedure complete.${NC}"
exit 0 