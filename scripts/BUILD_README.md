# Darwin Foundation Build System

Easy clean and build script for your macOS Darwin kernel optimization project.

## Quick Start

```bash
# Quick test (recommended first run)
./scripts/build_darwin_foundation.sh quick

# Full build and test
./scripts/build_darwin_foundation.sh full

# Interactive menu
./scripts/build_darwin_foundation.sh
```

## Command Line Options

| Command | Description |
|---------|-------------|
| `clean` | Remove all build artifacts |
| `setup` | Set up Darwin kernel directory structure |
| `tables` | Generate GF(256) lookup tables |
| `build` | Build all validation tests |
| `test` | Run all validation tests |
| `full` | Complete build (setup → tables → build → test) |
| `quick` | Essential build + key tests (fastest validation) |
| `status` | Show current build status |

## What Gets Built

### C Validation Programs
- `debug_fp` - Fixed-point precision debugging
- `poly_check` - GF(256) polynomial verification
- `reference_check` - Cross-reference validation
- `ultra_check` - Ultra-thorough validation
- `final_check` - Final validation suite
- `final_test` - Foundation validation

### C++ Demo Programs
- `working_foundation_demo` - Working foundation demo
- `safe_foundation_demo` - Safe implementation demo
- `final_darwin_test` - Complete Darwin test
- `minimal_test` - Minimal functionality test

## Generated Files

- `darwin_kernel/` - Kernel directory structure
- `darwin_kernel/darwin_radml_real.h` - Complete foundation header
- `darwin_kernel/gf256_tables.h` - GF(256) lookup tables
- Various test binaries in `scripts/`

## Typical Workflow

1. **First time setup**:
   ```bash
   ./scripts/build_darwin_foundation.sh quick
   ```

2. **Development cycle**:
   ```bash
   # Make changes to source files
   ./scripts/build_darwin_foundation.sh build
   ./scripts/build_darwin_foundation.sh test
   ```

3. **Clean rebuild**:
   ```bash
   ./scripts/build_darwin_foundation.sh clean
   ./scripts/build_darwin_foundation.sh full
   ```

## Requirements

- macOS (Darwin)
- Clang compiler
- Standard C/C++ libraries

All dependencies should be available by default on macOS.

## Single Developer Optimized

This build system is optimized for single developer workflow:
- ✅ No complex dependency management
- ✅ Color-coded output for easy reading
- ✅ Graceful failure handling
- ✅ Quick validation options
- ✅ Interactive menu for exploration
- ✅ Command line options for automation

Perfect for your Darwin kernel optimization development! 🍎
