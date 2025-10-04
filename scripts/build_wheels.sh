#!/bin/bash
# Test wheel building script for different platforms

set -e

echo "🚀 Testing wheel building for trendfilter-py"

# Install cibuildwheel if not present
if ! command -v cibuildwheel &> /dev/null; then
    echo "📦 Installing cibuildwheel..."
    pip install cibuildwheel
fi

# Set platform-specific settings
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Building for Linux..."
    export CIBW_BUILD="cp39-manylinux_x86_64 cp310-manylinux_x86_64 cp311-manylinux_x86_64 cp312-manylinux_x86_64"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Building for macOS..."
    export CIBW_BUILD="cp39-macosx_x86_64 cp310-macosx_x86_64 cp311-macosx_x86_64 cp312-macosx_x86_64"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "🪟 Building for Windows..."
    export CIBW_BUILD="cp39-win_amd64 cp310-win_amd64 cp311-win_amd64 cp312-win_amd64"
fi

# Build wheels
echo "🔨 Building wheels..."
cibuildwheel --output-dir wheelhouse

echo "✅ Wheel building completed!"
echo "📁 Wheels are in ./wheelhouse/"
ls -la wheelhouse/

echo ""
echo "🧪 To test a wheel, run:"
echo "   pip install wheelhouse/[wheel-name].whl"
echo "   python -c 'import trendfilter; print(\"Success!\")'"
