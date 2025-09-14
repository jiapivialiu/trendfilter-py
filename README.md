# trendfilter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Python package for fast and flexible univariate trend filtering with C++ backend for high performance.

## Overview

Trend filtering is a method for nonparametric regression that fits a piecewise polynomial function to data. This package provides efficient implementations of trend filtering algorithms with the following features:

- **Fast C++ backend**: High-performance implementations using modern C++
- **Python interface**: Easy-to-use Python API built with pybind11
- **Flexible algorithms**: Support for different orders of trend filtering
- **Cross-validation**: Built-in cross-validation for parameter selection
- **Efficient solvers**: Optimized linear system solvers for large datasets

## Installation

### From PyPI (recommended)
```bash
pip install trendfilter
```

### From source
```bash
git clone https://github.com/jiapivialiu/trendfilter-py.git
cd trendfilter-py
pip install .
```

### Development installation
```bash
git clone https://github.com/jiapivialiu/trendfilter-py.git
cd trendfilter-py
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from trendfilterpy import TrendFilter

# Generate sample data
n = 100
x = np.linspace(0, 1, n)
y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(n)

# Fit trend filter
tf = TrendFilter(order=1, lambda_reg=0.1)
y_fit = tf.fit(y)

# Cross-validation for parameter selection
from trendfilterpy import CVTrendFilter
cv_tf = CVTrendFilter(order=1)
y_fit_cv = cv_tf.fit(y)
```

## Features

### Core Algorithms
- **Trend Filtering**: Fast implementation of univariate trend filtering
- **Cross-Validation**: Automated parameter selection via cross-validation
- **Multiple Orders**: Support for different polynomial orders (0, 1, 2, ...)

### Performance
- **C++ Backend**: High-performance C++ implementations
- **Efficient Solvers**: Specialized linear system solvers
- **Memory Optimized**: Efficient memory usage for large datasets

## Documentation

Full documentation is available at: [https://trendfilter.readthedocs.io](https://trendfilter.readthedocs.io)

## Requirements

- Python >= 3.7
- NumPy >= 1.18.0
- SciPy >= 1.5.0

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Clone the repository
2. Install in development mode: `pip install -e ".[dev]"`
3. Run tests: `pytest`
4. Format code: `black trendfilterpy/`
5. Check types: `mypy trendfilterpy/`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{trendfilter,
  title={trendfilter: Fast univariate trend filtering in Python},
  author={Jiaping Liu, Daniel J McDonald, Addison Hu},
  year={2025},
  url={https://github.com/jiapivialiu/trendfilter-py}
}
```

## Acknowledgments

- Built with [pybind11](https://github.com/pybind/pybind11) for Python-C++ integration
- Inspired by the R package [genlasso](https://github.com/ryantibs/genlasso)
