# Building Cross-Platform Wheels

This document explains how to build wheels for Windows x86_64 and Linux platforms using GitHub Actions CI/CD.

## Overview

The project now includes comprehensive CI/CD workflows that automatically build wheels for:

- **Linux**: x86_64 (manylinux2014)
- **Windows**: x86_64 
- **macOS**: x86_64 and ARM64 (Apple Silicon)
- **Python versions**: 3.8, 3.9, 3.10, 3.11, 3.12

## CI/CD Workflows

### 1. Main Test Workflow (`.github/workflows/ci.yml`)
- Runs tests on all platforms and Python versions
- Performs linting, type checking, and code formatting checks
- Uploads coverage reports

### 2. Wheel Building Workflow (`.github/workflows/build-wheels.yml`)
- Builds wheels for all supported platforms
- Tests wheel installation and basic functionality
- Uploads wheels as artifacts
- Publishes to PyPI on releases (requires API token)

## How It Works

### Wheel Building Process

1. **System Dependencies Installation**:
   - Linux: `libeigen3-dev` via apt/yum
   - macOS: `eigen` via Homebrew
   - Windows: `eigen3` via vcpkg

2. **Cross-Platform Compilation**:
   - Uses `cibuildwheel` for consistent wheel building
   - Compiles C++ extensions for each platform
   - Handles platform-specific linking and dependency bundling

3. **Testing**:
   - Tests each built wheel on its target platform
   - Verifies import and basic functionality
   - Ensures wheels work across Python versions

### Key Features

- **Automatic dependency management**: Eigen3 is automatically installed on each platform
- **Cross-compilation support**: Linux wheels can be built on GitHub's Linux runners
- **Comprehensive testing**: Each wheel is tested before being marked as successful
- **Artifact storage**: Wheels are stored as GitHub artifacts for download
- **PyPI publishing**: Automatic publishing on tagged releases

## Configuration

### pyproject.toml Configuration

The `[tool.cibuildwheel]` section in `pyproject.toml` configures:

```toml
[tool.cibuildwheel]
build = "cp38-* cp39-* cp310-* cp311-* cp312-*"  # Python versions
skip = ["*-win32", "*-manylinux_i686", "*-musllinux_*", "pp*"]  # Skip patterns
test-requires = ["pytest", "numpy>=1.18.0", "scipy>=1.5.0"]
test-command = [
    "python -c 'import trendfilter; print(\"✓ Import successful!\")'",
    "python -c 'import trendfilter; import numpy as np; tf = trendfilter.TrendFilter(); y = np.random.randn(10); tf.fit(y); print(\"✓ Basic functionality working!\")'"
]
```

### Platform-Specific Settings

- **Linux**: Uses manylinux2014 for broad compatibility
- **Windows**: Sets `EIGEN3_INCLUDE_DIR` environment variable for vcpkg
- **macOS**: Builds for both x86_64 and ARM64, minimum macOS 10.14

## Local Testing

### Build Wheels Locally

Use the provided script to test wheel building:

```bash
# Make script executable (first time only)
chmod +x scripts/build_wheels.sh

# Build wheels for current platform
./scripts/build_wheels.sh
```

### Manual Building with cibuildwheel

```bash
# Install cibuildwheel
pip install cibuildwheel

# Build wheels (reads configuration from pyproject.toml)
cibuildwheel --output-dir wheelhouse

# Test a specific wheel
pip install wheelhouse/trendfilter-*.whl
python -c "import trendfilter; print('Success!')"
```

## GitHub Setup

### Required Secrets

For PyPI publishing, add this secret to your GitHub repository:

1. Go to GitHub repository → Settings → Secrets and variables → Actions
2. Add new secret: `PYPI_API_TOKEN`
3. Value: Your PyPI API token (get from https://pypi.org/manage/account/token/)

### Workflow Triggers

The workflows trigger on:
- **Push** to `main`, `develop`, or `better-doc` branches
- **Pull requests** to `main`
- **Tagged releases** (format: `v*`)
- **Published releases** (for PyPI upload)

## Wheel Distribution Strategy

### For Users

1. **PyPI Installation** (recommended):
   ```bash
   pip install trendfilter
   ```
   This will automatically download the appropriate wheel if available.

2. **From GitHub Releases**:
   Download wheels from the latest release and install manually.

3. **From Source** (fallback):
   ```bash
   pip install trendfilter --no-binary=trendfilter
   ```
   Requires GCC and Eigen3 development headers.

### For Developers

1. **Download Artifacts**: After CI runs, download wheel artifacts from the Actions tab
2. **Test Locally**: Install and test wheels before releases
3. **Release Process**: Create a tagged release to trigger PyPI upload

## Troubleshooting

### Common Issues

1. **Eigen3 Not Found**:
   - Check that system dependencies are properly installed
   - Verify `EIGEN3_INCLUDE_DIR` environment variable on Windows
   - Check vcpkg installation path on Windows

2. **Compilation Failures**:
   - Ensure C++14 compiler is available
   - Check that pybind11 and numpy are in build requirements
   - Verify setup.py can find all required headers

3. **Wheel Testing Failures**:
   - Check that the wheel contains the compiled extension
   - Verify all dependencies are properly bundled
   - Test import in a clean environment

### Platform-Specific Notes

- **Linux**: Uses Docker containers for building, ensuring consistency
- **Windows**: Relies on vcpkg for Eigen3, which can be slow to install
- **macOS**: Builds universal wheels that work on both Intel and Apple Silicon

## Monitoring

### Build Status

- Check the Actions tab in GitHub for build status
- Each platform builds independently, so partial failures are possible
- Artifacts are available even if some builds fail

### Performance

- **Linux builds**: ~10-15 minutes
- **Windows builds**: ~20-30 minutes (due to vcpkg)
- **macOS builds**: ~15-20 minutes

The builds run in parallel, so total time is limited by the slowest platform.
