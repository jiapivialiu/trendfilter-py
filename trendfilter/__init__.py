"""
trendfilter: Fast and flexible univariate trend filtering

A Python package for trend filtering with high-performance C++ backend.
"""

__version__ = "0.1.0"
__author__ = "Jiaping Liu, Daniel J McDonald, Addison Hu"
__email__ = "jiaping.liu@stat.ubc.ca, daniel@stat.ubc.ca, mail@huisaddison.com"

# Import main classes
try:
    from .trendfilter import TrendFilter
    from .cv_trendfilter import CVTrendFilter

    # Try to import C++ backend for testing
    try:
        from . import _trendfilter

        _cpp_available = True
    except ImportError:
        _cpp_available = False

    __all__ = [
        "TrendFilter",
        "CVTrendFilter",
        "__version__",
    ]

except ImportError as e:
    # Graceful fallback if dependencies are missing
    import warnings

    warnings.warn(f"Could not import main classes: {e}")
    __all__ = ["__version__"]


def get_backend_info():
    """
    Get information about available backends.

    Returns
    -------
    dict
        Dictionary with backend availability information
    """
    info = {"cpp_backend": False, "scipy_available": False, "sklearn_available": False}

    try:
        from . import _trendfilter

        info["cpp_backend"] = True
    except ImportError:
        pass

    try:
        import scipy

        info["scipy_available"] = True
    except ImportError:
        pass

    try:
        import sklearn

        info["sklearn_available"] = True
    except ImportError:
        pass

    return info
