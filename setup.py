#!/usr/bin/env python3
"""Setup script for trendfilter-py package."""

import os
import platform
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup
import numpy as np


def get_eigen_include():
    """Find Eigen include directory."""
    # Common Eigen installation paths
    potential_paths = [
        "/opt/homebrew/include/eigen3",  # macOS Homebrew
        "/usr/local/include/eigen3",     # Linux/Unix
        "/usr/include/eigen3",           # System installation
        "/usr/include/eigen",            # Alternative system path
        "C:/vcpkg/installed/x64-windows/include/eigen3",  # Windows vcpkg
        "C:/Program Files/Eigen3/include/eigen3",         # Windows manual install
    ]
    
    for path in potential_paths:
        if os.path.exists(os.path.join(path, "Eigen")):
            return path
    
    # If not found, try to use pkg-config
    try:
        import subprocess
        result = subprocess.run(['pkg-config', '--cflags-only-I', 'eigen3'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            include_path = result.stdout.strip().replace('-I', '')
            if os.path.exists(os.path.join(include_path, "Eigen")):
                return include_path
    except Exception:
        pass
    
    # Final fallback - warn but continue
    print("Warning: Could not find Eigen installation.")
    print("Build may fail. Please install Eigen3:")
    print("  - Ubuntu/Debian: sudo apt-get install libeigen3-dev")
    print("  - macOS: brew install eigen")
    print("  - Windows: vcpkg install eigen3:x64-windows")
    return "/usr/include/eigen3"  # Default fallback


# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))

# Get the include directories
include_dirs = [
    pybind11.get_include(),
    "trendfilterpy/src",
    np.get_include()
]

# Try to add Eigen include path
try:
    eigen_include = get_eigen_include()
    include_dirs.append(eigen_include)
    print(f"Found Eigen at: {eigen_include}")
except Exception as e:
    print(f"Warning: {e}")
    # Use fallback
    eigen_include = get_eigen_include()
    include_dirs.append(eigen_include)
    print(f"Using fallback Eigen path: {eigen_include}")

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "trendfilterpy._trendfilter",
        [
            "trendfilterpy/src/pybind_wrapper.cpp",
            "trendfilterpy/src/trendfilter.cpp",
            "trendfilterpy/src/kf_utils.cpp",
            "trendfilterpy/src/linearsystem.cpp",
            "trendfilterpy/src/utils.cpp",
            "trendfilterpy/src/matrix_construction.cpp",
            "trendfilterpy/src/matrix_multiplication.cpp",
        ],
        include_dirs=include_dirs,
        language='c++',
        cxx_std=14,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
