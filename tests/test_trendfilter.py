"""
Tests for the main TrendFilter class.
"""

import numpy as np
import pytest
from trendfilter import TrendFilter

try:
    from trendfilter import _trendfilter
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False


class TestTrendFilter:
    """Test cases for TrendFilter class."""
    
    def test_init(self):
        """Test TrendFilter initialization."""
        tf = TrendFilter(order=1, lambda_reg=0.1)
        assert tf.order == 1
        assert tf.lambda_reg == 0.1
        assert tf.method == 'auto'
        
    def test_init_multiple_params(self):
        """Test TrendFilter initialization with multiple parameters."""
        tf = TrendFilter(
            order=2, 
            lambda_reg=0.5,
            nlambda=30,
            max_iter=100,
            tol=1e-6,
            method='sparse_qr'
        )
        assert tf.order == 2
        assert tf.lambda_reg == 0.5
        assert tf.nlambda == 30
        assert tf.max_iter == 100
        assert tf.tol == 1e-6
        assert tf.method == 'sparse_qr'
        
    def test_fit_simple(self):
        """Test basic fitting functionality."""
        # Generate simple test data
        n = 50
        x = np.linspace(0, 1, n)
        np.random.seed(42)
        y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(n)
        
        tf = TrendFilter(order=1, lambda_reg=0.1)
        tf.fit(y, x)
        
        assert tf.coef_.shape[0] == n
        assert hasattr(tf, 'n_features_in_')
        assert tf.n_features_in_ == n
        
    def test_fit_no_x(self):
        """Test fitting without providing x coordinates."""
        n = 20
        np.random.seed(42)
        y = np.random.randn(n)
        
        tf = TrendFilter(order=1, lambda_reg=1.0)
        tf.fit(y)
        
        assert tf.coef_.shape[0] == n
        assert hasattr(tf, 'n_features_in_')
        
    def test_fit_with_weights(self):
        """Test fitting with sample weights."""
        n = 30
        x = np.linspace(0, 1, n)
        np.random.seed(42)
        y = np.random.randn(n)
        weights = np.random.uniform(0.5, 2.0, n)
        
        tf = TrendFilter(order=1, lambda_reg=0.5)
        tf.fit(y, x, sample_weight=weights)
        
        assert tf.coef_.shape[0] == n
        
    def test_fit_multiple_lambdas(self):
        """Test fitting with multiple lambda values."""
        n = 25
        np.random.seed(42)
        y = np.random.randn(n)
        lambda_seq = np.logspace(-2, 0, 10)
        
        tf = TrendFilter(order=1, lambda_reg=lambda_seq)
        tf.fit(y)
        
        if HAS_CPP_BACKEND:
            # With C++ backend, should have multiple solutions
            assert hasattr(tf, 'lambda_')
            assert len(tf.lambda_) > 1
            if tf.coef_.ndim > 1:
                assert tf.coef_.shape == (n, len(tf.lambda_))
        else:
            # With Python fallback
            assert tf.coef_.shape[0] == n
        
    def test_predict(self):
        """Test prediction functionality."""
        n = 30
        np.random.seed(42)
        y = np.random.randn(n)
        
        tf = TrendFilter(order=1, lambda_reg=0.5)
        tf.fit(y)
        
        y_pred = tf.predict()
        assert y_pred.shape[0] == n
        
    def test_predict_before_fit(self):
        """Test that predict raises error before fitting."""
        tf = TrendFilter()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            tf.predict()
            
    def test_different_orders(self):
        """Test different trend filtering orders."""
        n = 30
        np.random.seed(42)
        y = np.random.randn(n)
        
        for order in [0, 1, 2]:
            tf = TrendFilter(order=order, lambda_reg=0.1)
            tf.fit(y)
            assert tf.coef_.shape[0] == n
            
    def test_input_validation(self):
        """Test input validation."""
        tf = TrendFilter()
        
        # Test mismatched x and y lengths
        x = np.array([1, 2, 3])
        y = np.array([1, 2])
        
        with pytest.raises(ValueError, match="x and y must have the same length"):
            tf.fit(y, x)
            
        # Test mismatched weights and y lengths
        y = np.array([1, 2, 3])
        weights = np.array([1, 2])
        
        with pytest.raises(ValueError, match="sample_weight must have the same length"):
            tf.fit(y, sample_weight=weights)
            
    def test_solver_methods(self):
        """Test different solver methods."""
        n = 20
        np.random.seed(42)
        y = np.random.randn(n)
        
        methods = ['auto', 'sparse_qr', 'tridiag', 'kf']
        
        for method in methods:
            tf = TrendFilter(order=1, lambda_reg=0.1, method=method)
            tf.fit(y)
            assert tf.coef_.shape[0] == n
            
    def test_get_coefficients_at_lambda(self):
        """Test getting coefficients at specific lambda."""
        n = 20
        np.random.seed(42)
        y = np.random.randn(n)
        lambda_seq = np.logspace(-2, 0, 5)
        
        tf = TrendFilter(order=1, lambda_reg=lambda_seq)
        tf.fit(y)
        
        if hasattr(tf, 'lambda_') and len(tf.lambda_) > 1:
            coef = tf.get_coefficients_at_lambda(tf.lambda_[0])
            assert coef.shape[0] == n
        
    def test_get_best_lambda(self):
        """Test getting best lambda value."""
        n = 20
        np.random.seed(42)
        y = np.random.randn(n)
        lambda_seq = np.logspace(-2, 0, 5)
        
        tf = TrendFilter(order=1, lambda_reg=lambda_seq)
        tf.fit(y)
        
        if hasattr(tf, 'lambda_'):
            best_lambda = tf.get_best_lambda()
            assert isinstance(best_lambda, (int, float))
            
    @pytest.mark.skipif(not HAS_CPP_BACKEND, reason="C++ backend not available")
    def test_cpp_backend_specific(self):
        """Test C++ backend specific functionality."""
        n = 30
        x = np.linspace(0, 1, n)
        np.random.seed(42)
        y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(n)
        
        tf = TrendFilter(order=1, lambda_reg=0.1)
        tf.fit(y, x)
        
        # Should have additional attributes when using C++ backend
        assert hasattr(tf, 'lambda_')
        assert hasattr(tf, 'objective_')
        assert hasattr(tf, 'n_iter_')
        
    def test_data_types(self):
        """Test different input data types."""
        n = 20
        
        # Test with different dtypes
        y_int = np.random.randint(0, 10, n)
        y_float32 = np.random.randn(n).astype(np.float32)
        y_float64 = np.random.randn(n).astype(np.float64)
        
        for y in [y_int, y_float32, y_float64]:
            tf = TrendFilter(order=1, lambda_reg=0.1)
            tf.fit(y)
            assert tf.coef_.shape[0] == n
            
    def test_edge_cases(self):
        """Test edge cases."""
        # Very small data
        y_small = np.array([1.0, 2.0])
        tf = TrendFilter(order=0, lambda_reg=1.0)  # Use order 0 for small data
        tf.fit(y_small)
        assert tf.coef_.shape[0] == 2
        
        # Very large lambda (should give very smooth result)
        n = 20
        np.random.seed(42)
        y = np.random.randn(n)
        tf = TrendFilter(order=1, lambda_reg=1e6)
        tf.fit(y)
        assert tf.coef_.shape[0] == n
        
        # Very small lambda (should be close to original data)
        tf = TrendFilter(order=1, lambda_reg=1e-6)
        tf.fit(y)
        assert tf.coef_.shape[0] == n
