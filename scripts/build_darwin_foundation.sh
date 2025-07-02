#!/bin/bash
# Darwin Foundation - Easy Clean & Build Script
# Single developer workflow for macOS kernel optimization

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🍎 Darwin RadML Foundation - Build System"
echo "=========================================="
echo "Root: $ROOT_DIR"
echo "Scripts: $SCRIPT_DIR"
echo ""

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# Function to clean all builds
clean_all() {
    log_info "Cleaning all build artifacts..."

    cd "$SCRIPT_DIR"

    # Remove compiled binaries
    rm -f debug_fp debug_fixed_point ultra_check poly_check reference_check
    rm -f verify_all comprehensive_verification final_check ultra_thorough_check
    rm -f final_test validate_complete_foundation final_validation
    rm -f gf256_generator gf256_table_generator test_real_foundation
    rm -f working_foundation_demo safe_foundation_demo extract_foundation_demo
    rm -f final_darwin_test test_galois minimal_test

    # Remove object files
    rm -f *.o *.dSYM

    # Clean darwin_kernel directory but preserve structure
    if [ -d "$ROOT_DIR/darwin_kernel" ]; then
        log_info "Cleaning darwin_kernel directory..."
        find "$ROOT_DIR/darwin_kernel" -name "*.o" -delete 2>/dev/null || true
        find "$ROOT_DIR/darwin_kernel" -name "darwin_math_tests" -delete 2>/dev/null || true
    fi

    log_success "Clean completed!"
}

# Function to set up Darwin kernel structure
setup_kernel_structure() {
    log_info "Setting up Darwin kernel structure..."

    cd "$ROOT_DIR"

    if [ ! -f scripts/darwin_kernel_starter.sh ]; then
        log_error "darwin_kernel_starter.sh not found!"
        exit 1
    fi

    chmod +x scripts/darwin_kernel_starter.sh
    ./scripts/darwin_kernel_starter.sh

    log_success "Kernel structure ready!"
}

# Function to generate GF(256) tables
generate_tables() {
    log_info "Generating GF(256) lookup tables..."

    cd "$SCRIPT_DIR"

    # Build table generator
    if [ -f gf256_table_generator.cpp ]; then
        clang++ -std=c++17 -O2 -Wall -Wextra -o gf256_generator gf256_table_generator.cpp
        ./gf256_generator
        log_success "GF(256) tables generated!"
    else
        log_warning "gf256_table_generator.cpp not found, skipping table generation"
    fi
}

# Function to build all validation tests
build_validations() {
    log_info "Building validation tests..."

    cd "$SCRIPT_DIR"

    # C validation programs
    local c_programs=(
        "debug_fixed_point.c:debug_fp"
        "polynomial_verification.c:poly_check"
        "gf256_reference_check.c:reference_check"
        "comprehensive_verification.c:verify_all"
        "ultra_thorough_check.c:ultra_check"
        "final_validation.c:final_check"
        "validate_complete_foundation.c:final_test"
        "benchmark_realistic.c:benchmark_realistic"
    )

    for program in "${c_programs[@]}"; do
        IFS=':' read -r source binary <<< "$program"
        if [ -f "$source" ]; then
            log_info "Building $source -> $binary"
            clang -std=c99 -O2 -Wall -Wextra -lm -I"$ROOT_DIR" -o "$binary" "$source"
            log_success "Built $binary"
        else
            log_warning "$source not found, skipping"
        fi
    done

    # C++ programs
    local cpp_programs=(
        "working_foundation_demo.cpp:working_foundation_demo"
        "safe_foundation_demo.cpp:safe_foundation_demo"
        "final_darwin_test.cpp:final_darwin_test"
        "minimal_test.cpp:minimal_test"
    )

    for program in "${cpp_programs[@]}"; do
        IFS=':' read -r source binary <<< "$program"
        if [ -f "$source" ]; then
            log_info "Building $source -> $binary"
            # Try to compile, but don't fail if headers are missing
            if clang++ -std=c++17 -O2 -Wall -Wextra -I"$ROOT_DIR" -o "$binary" "$source" 2>/dev/null; then
                log_success "Built $binary"
            else
                log_warning "Failed to build $binary (missing headers?)"
            fi
        else
            log_warning "$source not found, skipping"
        fi
    done
}

# Function to run all tests
run_tests() {
    log_info "Running validation tests..."

    cd "$SCRIPT_DIR"

    local test_programs=(
        "debug_fp:Fixed-point debugging"
        "poly_check:Polynomial verification"
        "reference_check:GF(256) reference check"
        "ultra_check:Ultra-thorough validation"
        "final_check:Final validation"
        "final_test:Foundation validation"
    )

    local passed=0
    local total=0

    for test in "${test_programs[@]}"; do
        IFS=':' read -r binary description <<< "$test"
        if [ -x "$binary" ]; then
            total=$((total + 1))
            log_info "Running $description..."
            if "./$binary" > /dev/null 2>&1; then
                log_success "$description passed"
                passed=$((passed + 1))
            else
                log_error "$description failed"
            fi
        fi
    done

    echo ""
    log_info "Test Results: $passed/$total tests passed"

    if [ $passed -eq $total ] && [ $total -gt 0 ]; then
        log_success "ALL TESTS PASSED! Darwin foundation is ready! 🎉"
    elif [ $total -eq 0 ]; then
        log_warning "No tests found to run"
    else
        log_error "Some tests failed - review the output above"
    fi
}

# Function to show status
show_status() {
    log_info "Darwin Foundation Status"
    echo ""

    if [ -d "$ROOT_DIR/darwin_kernel" ]; then
        log_success "Darwin kernel directory exists"
        ls -la "$ROOT_DIR/darwin_kernel/" | head -10
    else
        log_warning "Darwin kernel directory not found"
    fi

    echo ""
    log_info "Available test binaries in scripts/:"
    cd "$SCRIPT_DIR"
    for binary in debug_fp poly_check reference_check ultra_check final_check final_test; do
        if [ -x "$binary" ]; then
            log_success "$binary (executable)"
        else
            log_warning "$binary (not built)"
        fi
    done
}

# Quick test - build essentials and run key tests
quick_test() {
    log_info "Quick test - building essentials..."

    setup_kernel_structure

    cd "$SCRIPT_DIR"

    # Build just the key validation tests
    clang -std=c99 -O2 -Wall -Wextra -lm -I"$ROOT_DIR" -o final_check final_validation.c 2>/dev/null || log_warning "Failed to build final_validation"
    clang -std=c99 -O2 -Wall -Wextra -lm -I"$ROOT_DIR" -o ultra_check ultra_thorough_check.c 2>/dev/null || log_warning "Failed to build ultra_thorough_check"

    # Run key tests
    if [ -x final_check ]; then
        log_info "Running final validation..."
        if ./final_check; then
            log_success "Final validation PASSED! 🎉"
        else
            log_error "Final validation FAILED"
        fi
    fi

    if [ -x ultra_check ]; then
        log_info "Running ultra-thorough check..."
        if ./ultra_check; then
            log_success "Ultra-thorough validation PASSED! 🎉"
        else
            log_error "Ultra-thorough validation FAILED"
        fi
    fi
}

# Parse command line arguments
if [ $# -eq 1 ]; then
    case $1 in
        clean) clean_all; exit 0 ;;
        setup) setup_kernel_structure; exit 0 ;;
        tables) generate_tables; exit 0 ;;
        build) build_validations; exit 0 ;;
        test) run_tests; exit 0 ;;
        full)
            setup_kernel_structure
            generate_tables
            build_validations
            run_tests
            exit 0
            ;;
        quick) quick_test; exit 0 ;;
        status) show_status; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
fi

# Interactive menu if no arguments
echo ""
echo "Darwin Foundation Build Options:"
echo "1. Clean all"
echo "2. Setup kernel structure"
echo "3. Generate GF(256) tables"
echo "4. Build validation tests"
echo "5. Run all tests"
echo "6. Full build (2-5)"
echo "7. Show status"
echo "8. Quick test (build essential + run)"
echo "0. Exit"
echo ""

while true; do
    read -p "Choose option [0-8]: " choice

    case $choice in
        1) clean_all ;;
        2) setup_kernel_structure ;;
        3) generate_tables ;;
        4) build_validations ;;
        5) run_tests ;;
        6)
            setup_kernel_structure
            generate_tables
            build_validations
            run_tests
            ;;
        7) show_status ;;
        8) quick_test ;;
        0)
            log_success "Darwin foundation development complete! 🍎"
            exit 0
            ;;
        *) log_error "Invalid option. Please choose 0-8." ;;
    esac
    echo ""
    read -p "Press Enter to continue..."
    echo ""
    echo "Darwin Foundation Build Options:"
    echo "1. Clean all"
    echo "2. Setup kernel structure"
    echo "3. Generate GF(256) tables"
    echo "4. Build validation tests"
    echo "5. Run all tests"
    echo "6. Full build (2-5)"
    echo "7. Show status"
    echo "8. Quick test (build essential + run)"
    echo "0. Exit"
    echo ""
done
