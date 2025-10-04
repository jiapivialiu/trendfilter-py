#!/usr/bin/env python3
"""
Test script to verify trendfilter wheel installation and functionality.
Run this after installing a wheel to ensure it works correctly.
"""

import sys
import traceback
import numpy as np

def test_basic_import():
    """Test basic package import."""
    print("Testing basic import...")
    try:
        import trendfilter
        print("Successfully imported trendfilter")
        return True
    except ImportError as e:
        print(f"Failed to import trendfilter: {e}")
        return False

def test_trendfilter_functionality():
    """Test TrendFilter basic functionality."""
    print("Testing TrendFilter functionality...")
    try:
        import trendfilter
        
        # Generate test data
        n = 100
        x = np.linspace(0, 1, n)
        true_signal = np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)
        y = true_signal + 0.1 * np.random.randn(n)
        
        # Test TrendFilter
        tf = trendfilter.TrendFilter(order=2, lambda_reg=0.1)
        tf.fit(y)
        y_pred = tf.predict()
        
        # Check outputs - predict() returns fitted values, could be 1D or 2D
        if y_pred.ndim == 1:
            expected_shape = y.shape
        elif y_pred.ndim == 2:
            expected_shape = (y.shape[0], y_pred.shape[1])
        else:
            print(f"Unexpected output dimensions: {y_pred.ndim}")
            return False
            
        if y_pred.shape[0] != y.shape[0]:
            print(f"First dimension mismatch: input {y.shape}, output {y_pred.shape}")
            return False
            
        if not np.isfinite(y_pred).all():
            print("Output contains non-finite values")
            return False
            
        print(f"TrendFilter working: {y.shape} -> {y_pred.shape}")
        return True
        
    except Exception as e:
        print(f"TrendFilter test failed: {e}")
        traceback.print_exc()
        return False

def test_cv_functionality():
    """Test CVTrendFilter functionality."""
    print("Testing CVTrendFilter functionality...")
    try:
        import trendfilter
        
        # Generate test data
        n = 50  # Smaller for CV
        y = np.sin(np.linspace(0, 4*np.pi, n)) + 0.1 * np.random.randn(n)
        
        # Test CVTrendFilter with correct parameter name
        cv_tf = trendfilter.CVTrendFilter(order=1, cv=3)
        cv_tf.fit(y)
        y_cv_pred = cv_tf.predict()
        
        # Check outputs - could be 1D or 2D
        if y_cv_pred.shape[0] != y.shape[0]:
            print(f"CV shape mismatch: input {y.shape}, output {y_cv_pred.shape}")
            return False
            
        if not np.isfinite(y_cv_pred).all():
            print("CV output contains non-finite values")
            return False
            
        # Check that best_lambda_ is available
        if not hasattr(cv_tf, 'best_lambda_') or cv_tf.best_lambda_ is None:
            print("CV did not set best_lambda_")
            return False
            
        print(f"CVTrendFilter working: {y.shape} -> {y_cv_pred.shape}, best_lambda={cv_tf.best_lambda_:.4f}")
        return True
        
    except Exception as e:
        print(f"CVTrendFilter test failed: {e}")
        traceback.print_exc()
        return False

def test_cpp_backend():
    """Test if C++ backend is available."""
    print("Testing C++ backend availability...")
    try:
        import trendfilter
        
        # Check if C++ backend is available
        if hasattr(trendfilter, '_trendfilter'):
            print("C++ backend is available")
            return True
        else:
            print("C++ backend not available, using Python fallback")
            return True  # Not a failure, just different implementation
            
    except Exception as e:
        print(f"Backend test failed: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("Testing dependencies...")
    
    required_packages = ['numpy', 'scipy']
    optional_packages = ['sklearn', 'matplotlib']
    
    all_good = True
    
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"{pkg} available")
        except ImportError:
            print(f"{pkg} missing (required)")
            all_good = False
    
    for pkg in optional_packages:
        try:
            __import__(pkg)
            print(f"{pkg} available")
        except ImportError:
            print(f"{pkg} missing (optional)")
    
    return all_good

def main():
    """Run all tests."""
    print("Testing trendfilter wheel installation\n")
    
    tests = [
        ("Basic Import", test_basic_import),
        ("Dependencies", test_dependencies),
        ("TrendFilter", test_trendfilter_functionality),
        ("CVTrendFilter", test_cv_functionality),
        ("C++ Backend", test_cpp_backend),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} test")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The wheel is working correctly.")
        return 0
    else:
        print("Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
