# Radiation Tolerant ML - Student / Contributor Build Guide

This guide provides straightforward instructions for building and testing the RadML framework from source.

## Version Information

Current version: **v1.0.2.5** — includes Reed-Solomon ECC, 8 TMR variants, physics-based radiation models, VAE compression, and evolutionary architecture search.

## Prerequisites

| Dependency | Required | Install |
|------------|----------|---------|
| **CMake** 3.10+ | Yes | `sudo apt install cmake` / `brew install cmake` |
| **C++17 compiler** (GCC 7+, Clang 5+, MSVC 2017+) | Yes | System default |
| **Eigen3** | Yes | `sudo apt install libeigen3-dev` / `brew install eigen` |
| **LMDB** | Optional (auto-downloaded if missing) | `sudo apt install liblmdb-dev` / `brew install lmdb` |
| **GoogleTest** | Optional (auto-downloaded if missing) | `sudo apt install libgtest-dev` / `brew install googletest` |
| **PyTorch/LibTorch** | Optional | See [pytorch.org](https://pytorch.org/get-started/locally/) |

Or run the helper script:
```bash
./tools/install_dependencies.sh
```

## Quick Start (Out-of-Source Build)

```bash
# 1. Clone and enter the repo
git clone https://github.com/r0nlt/Space-Radiation-Tolerant.git
cd Space-Radiation-Tolerant

# 2. Configure (out-of-source)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 3. Build (use number of cores available)
cmake --build build -j$(nproc)    # Linux
cmake --build build -j$(sysctl -n hw.ncpu)  # macOS
cmake --build build -j4           # or just pick a number

# 4. Run all tests
cd build && ctest --output-on-failure
```

That's it. If the build succeeds and tests pass, you're good.

## Build Configurations

### Minimal (core framework only, no PyTorch)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTORCH=OFF
cmake --build build -j$(nproc)
```

### Full (with PyTorch integration)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTORCH=ON -DBUILD_TESTING=ON
cmake --build build -j$(nproc)
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | — | `Release` (optimized) or `Debug` (symbols) |
| `ENABLE_PYTORCH` | OFF | Enable PyTorch/LibTorch integration |
| `BUILD_TESTING` | ON | Build the test suite |
| `DOWNLOAD_LMDB` | ON | Auto-download LMDB if not found |

## Running Tests

```bash
# Run all tests
cd build && ctest --output-on-failure

# Run a specific test by name
ctest -R monte_carlo_validation --output-on-failure

# List all available tests
ctest -N
```

### Key Tests

| Test | What it validates | Runtime |
|------|-------------------|---------|
| `monte_carlo_validation` | Statistical protection across 8 environments | ~3 min |
| `monte_carlo_neuralnetwork` | Neural network under radiation | ~3 min |
| `enhanced_tmr_test` | TMR variants and voting | seconds |
| `framework_verification_test` | Core framework integrity | seconds |
| `scientific_validation_test` | Physics model accuracy | seconds |

## Running Examples

After building, example binaries are in `build/examples/` (or `build/` depending on CMake):

```bash
 ctest -R monte_carlo_validation -V
```

## Troubleshooting

### Eigen3 not found
```
Could not find Eigen3
```
Install it, or point CMake to it:
```bash
cmake -S . -B build -DEigen3_DIR=/usr/lib/cmake/eigen3
# or
cmake -S . -B build -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3
```

### LMDB not found
The build system will auto-download LMDB via FetchContent if `DOWNLOAD_LMDB=ON` (the default). If that fails:
```bash
sudo apt install liblmdb-dev   # Ubuntu/Debian
brew install lmdb              # macOS
```

### PyTorch not found (when `ENABLE_PYTORCH=ON`)
```bash
export PyTorch_ROOT=/path/to/libtorch
cmake -S . -B build -DENABLE_PYTORCH=ON
```

### Clean rebuild
```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Project Structure

```
Space-Radiation-Tolerant/
├── include/rad_ml/       # Headers (the library API)
│   ├── neural/           # Neural network protection, ECC, Galois field
│   ├── tmr/              # 8 TMR variants + adaptive protection
│   ├── physics/          # Weibull, Bendel, SAA, transport models
│   ├── research/         # VAE, evolutionary search
│   └── ...
├── src/rad_ml/           # Implementation files
├── test/                 # Test suite
│   └── verification/     # Monte Carlo validation tests
├── examples/             # Example applications
├── tools/                # Helper scripts
├── RadML_Manualv2.tex    # Technical manual (LaTeX)
└── CMakeLists.txt        # Root build configuration
```

## Documentation

- **Technical Manual**: `RadML_Manualv2.pdf` (compile from `.tex` with `pdflatex`)
- **Auto Arch Search**: `AUTO_ARCH_SEARCH_GUIDE.md`
- **VAE Guide**: `VAE_TUNING_GUIDE.md`
- **FAQ directory**: Various topic-specific guides

## Contact

- **Author**: Rishab Nuguru
- **Email**: spacelabsai@gmail.com
- **GitHub**: https://github.com/r0nlt/Space-Radiation-Tolerant
