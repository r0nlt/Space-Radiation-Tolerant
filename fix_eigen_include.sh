#!/bin/bash

# Script to fix Eigen includes in all header and source files
# This script replaces 'eigen3/Eigen/' with just 'Eigen/' to match standard include paths

echo "Fixing Eigen includes in all source and header files..."

# Find all header and source files
for file in $(find . -name "*.hpp" -o -name "*.cpp" -o -name "*.h" -o -name "*.cc"); do
    # Skip files in build directories and third-party code
    if [[ $file == *"/build/"* || $file == *"/third_party/"* ]]; then
        continue
    fi
    
    # Make sure we're using standard Eigen includes (without eigen3/ prefix)
    sed -i '' 's|#include <eigen3/Eigen/|#include <Eigen/|g' "$file"
    echo "Fixed $file"
done

echo "Done fixing Eigen includes." 