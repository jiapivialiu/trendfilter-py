# Documentation for trendfilter-py

This directory contains the documentation for the trendfilter-py package.

## Building Documentation

To build the documentation locally:

1. Install documentation dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Build the documentation:
   ```bash
   cd docs
   make html
   ```

3. View the documentation:
   ```bash
   open _build/html/index.html
   ```

## Structure

- `index.rst` - Main documentation index
- `api/` - API reference documentation
- `examples/` - Usage examples and tutorials
- `conf.py` - Sphinx configuration

## TODO

- Set up Sphinx documentation
- Add API reference
- Add usage examples
- Add installation guide
- Add mathematical background
