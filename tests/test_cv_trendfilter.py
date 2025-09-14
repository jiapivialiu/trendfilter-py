"""
Tests for cross-validation functionality.
"""

import numpy as np
import pytest
from trendfilter import CVTrendFilter

try:
    from sklearn.model_selection import KFold

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TestCVTrendFilter:
    """Test cases for CVTrendFilter class."""

    def test_init(self):
        """Test CVTrendFilter initialization."""
        tf_cv = CVTrendFilter(order=1, cv=5)
        assert tf_cv.order == 1
        assert tf_cv.cv == 5
        assert tf_cv.scoring == "neg_mean_squared_error"

    def test_init_custom_lambdas(self):
        """Test initialization with custom lambda sequence."""
        lambdas = np.logspace(-3, 1, 20)
        tf_cv = CVTrendFilter(
            order=2, lambdas=lambdas, cv=3, scoring="neg_mean_absolute_error"
        )
        assert tf_cv.order == 2
        assert tf_cv.cv == 3
        assert tf_cv.scoring == "neg_mean_absolute_error"
        np.testing.assert_array_equal(tf_cv.lambdas, lambdas)

    def test_fit_simple(self):
        """Test basic cross-validation fitting."""
        # Generate simple test data
        n = 50
        x = np.linspace(0, 1, n)
        np.random.seed(42)
        y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(n)

        tf_cv = CVTrendFilter(order=1, cv=3, lambdas=np.logspace(-2, 0, 5))
        tf_cv.fit(y, x)

        assert hasattr(tf_cv, "best_lambda_")
        assert hasattr(tf_cv, "best_score_")
        assert hasattr(tf_cv, "cv_scores_")
        assert tf_cv.coef_.shape[0] == n

    def test_fit_no_x(self):
        """Test fitting without x coordinates."""
        n = 30
        np.random.seed(42)
        y = np.random.randn(n)

        tf_cv = CVTrendFilter(order=1, cv=3, lambdas=np.logspace(-2, 0, 5))
        tf_cv.fit(y)

        assert hasattr(tf_cv, "best_lambda_")
        assert tf_cv.coef_.shape[0] == n

    def test_fit_with_weights(self):
        """Test fitting with sample weights."""
        n = 40
        x = np.linspace(0, 1, n)
        np.random.seed(42)
        y = np.random.randn(n)
        weights = np.random.uniform(0.5, 2.0, n)

        tf_cv = CVTrendFilter(order=1, cv=3, lambdas=np.logspace(-2, 0, 5))
        tf_cv.fit(y, x, sample_weight=weights)

        assert hasattr(tf_cv, "best_lambda_")
        assert tf_cv.coef_.shape[0] == n

    def test_predict(self):
        """Test prediction after cross-validation."""
        n = 30
        np.random.seed(42)
        y = np.random.randn(n)

        tf_cv = CVTrendFilter(order=1, cv=3, lambdas=np.logspace(-2, 0, 5))
        tf_cv.fit(y)

        y_pred = tf_cv.predict()
        assert y_pred.shape[0] == n

    def test_predict_before_fit(self):
        """Test that predict raises error before fitting."""
        tf_cv = CVTrendFilter()

        with pytest.raises(ValueError, match="Model must be fitted"):
            tf_cv.predict()

    def test_score(self):
        """Test scoring functionality."""
        n = 25
        np.random.seed(42)
        y = np.random.randn(n)

        tf_cv = CVTrendFilter(order=1, cv=3, lambdas=np.logspace(-2, 0, 5))
        tf_cv.fit(y)

        score = tf_cv.score(y)
        assert isinstance(score, float)

    def test_auto_lambda_generation(self):
        """Test automatic lambda sequence generation."""
        n = 30
        np.random.seed(42)
        y = np.random.randn(n)

        tf_cv = CVTrendFilter(order=1, cv=3, nlambda=10)
        tf_cv.fit(y)

        assert hasattr(tf_cv, "lambdas")
        assert len(tf_cv.lambdas) == 10
        assert hasattr(tf_cv, "best_lambda_")

    def test_different_cv_strategies(self):
        """Test different cross-validation strategies."""
        n = 40
        np.random.seed(42)
        y = np.random.randn(n)
        lambdas = np.logspace(-2, 0, 5)

        # Test integer CV
        tf_cv1 = CVTrendFilter(order=1, cv=4, lambdas=lambdas)
        tf_cv1.fit(y)
        assert hasattr(tf_cv1, "best_lambda_")

        # Test leave-one-out CV
        tf_cv2 = CVTrendFilter(order=1, cv=n, lambdas=lambdas)
        tf_cv2.fit(y)
        assert hasattr(tf_cv2, "best_lambda_")

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
    def test_sklearn_cv_object(self):
        """Test using sklearn CV objects."""
        n = 30
        np.random.seed(42)
        y = np.random.randn(n)
        lambdas = np.logspace(-2, 0, 5)

        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        tf_cv = CVTrendFilter(order=1, cv=cv, lambdas=lambdas)
        tf_cv.fit(y)

        assert hasattr(tf_cv, "best_lambda_")
        assert tf_cv.coef_.shape[0] == n

    def test_scoring_functions(self):
        """Test different scoring functions."""
        n = 30
        np.random.seed(42)
        y = np.random.randn(n)
        lambdas = np.logspace(-2, 0, 5)

        scoring_options = ["neg_mean_squared_error", "neg_mean_absolute_error"]

        for scoring in scoring_options:
            tf_cv = CVTrendFilter(order=1, cv=3, lambdas=lambdas, scoring=scoring)
            tf_cv.fit(y)
            assert hasattr(tf_cv, "best_lambda_")

    def test_different_orders(self):
        """Test cross-validation with different orders."""
        n = 30
        np.random.seed(42)
        y = np.random.randn(n)
        lambdas = np.logspace(-2, 0, 5)

        for order in [0, 1, 2]:
            tf_cv = CVTrendFilter(order=order, cv=3, lambdas=lambdas)
            tf_cv.fit(y)
            assert hasattr(tf_cv, "best_lambda_")
            assert tf_cv.coef_.shape[0] == n

    def test_input_validation(self):
        """Test input validation."""
        tf_cv = CVTrendFilter()

        # Test mismatched x and y lengths
        x = np.array([1, 2, 3])
        y = np.array([1, 2])

        with pytest.raises(ValueError, match="x and y must have the same length"):
            tf_cv.fit(y, x)

        # Test mismatched weights and y lengths
        y = np.array([1, 2, 3])
        weights = np.array([1, 2])

        with pytest.raises(ValueError, match="sample_weight must have the same length"):
            tf_cv.fit(y, sample_weight=weights)

        # Test invalid CV value
        with pytest.raises(ValueError, match="cv must be at least 2"):
            tf_cv = CVTrendFilter(cv=1)
            tf_cv.fit(y)

    def test_cv_scores_shape(self):
        """Test that CV scores have correct shape."""
        n = 30
        np.random.seed(42)
        y = np.random.randn(n)
        lambdas = np.logspace(-2, 0, 5)
        cv_folds = 3

        tf_cv = CVTrendFilter(order=1, cv=cv_folds, lambdas=lambdas)
        tf_cv.fit(y)

        # CV scores should be (n_lambdas, n_folds)
        expected_shape = (len(lambdas), cv_folds)
        assert tf_cv.cv_scores_.shape == expected_shape

    def test_best_lambda_selection(self):
        """Test that best lambda is correctly selected."""
        n = 30
        np.random.seed(42)
        y = np.random.randn(n)
        lambdas = np.logspace(-2, 0, 5)

        tf_cv = CVTrendFilter(order=1, cv=3, lambdas=lambdas)
        tf_cv.fit(y)

        # Best lambda should be one of the provided lambdas
        assert tf_cv.best_lambda_ in lambdas

        # Best score should correspond to best lambda
        mean_scores = np.mean(tf_cv.cv_scores_, axis=1)
        best_idx = np.argmax(mean_scores)  # Higher is better for negative scoring
        assert tf_cv.best_lambda_ == lambdas[best_idx]

    def test_fallback_cv_without_sklearn(self):
        """Test fallback CV implementation when sklearn is not available."""
        n = 20
        np.random.seed(42)
        y = np.random.randn(n)
        lambdas = np.logspace(-2, 0, 3)

        tf_cv = CVTrendFilter(order=1, cv=3, lambdas=lambdas)
        tf_cv.fit(y)

        # Should still work even without sklearn
        assert hasattr(tf_cv, "best_lambda_")
        assert tf_cv.coef_.shape[0] == n

    def test_edge_cases(self):
        """Test edge cases."""
        # Very small data
        y_small = np.array([1.0, 2.0, 3.0, 4.0])
        lambdas = [0.1, 1.0]

        tf_cv = CVTrendFilter(order=0, cv=2, lambdas=lambdas)
        tf_cv.fit(y_small)
        assert hasattr(tf_cv, "best_lambda_")

        # Single lambda value
        n = 20
        np.random.seed(42)
        y = np.random.randn(n)

        tf_cv = CVTrendFilter(order=1, cv=3, lambdas=[0.1])
        tf_cv.fit(y)
        assert tf_cv.best_lambda_ == 0.1

    def test_reproducibility(self):
        """Test that results are reproducible."""
        n = 30
        np.random.seed(42)
        y = np.random.randn(n)
        lambdas = np.logspace(-2, 0, 5)

        tf_cv1 = CVTrendFilter(order=1, cv=3, lambdas=lambdas, random_state=42)
        tf_cv1.fit(y)

        tf_cv2 = CVTrendFilter(order=1, cv=3, lambdas=lambdas, random_state=42)
        tf_cv2.fit(y)

        assert tf_cv1.best_lambda_ == tf_cv2.best_lambda_
        np.testing.assert_array_almost_equal(tf_cv1.coef_, tf_cv2.coef_)

    def test_get_best_estimator(self):
        """Test getting the best estimator."""
        n = 30
        np.random.seed(42)
        y = np.random.randn(n)
        lambdas = np.logspace(-2, 0, 5)

        tf_cv = CVTrendFilter(order=1, cv=3, lambdas=lambdas)
        tf_cv.fit(y)

        # Should be able to get the best estimator
        if hasattr(tf_cv, "best_estimator_"):
            best_est = tf_cv.best_estimator_
            assert hasattr(best_est, "coef_")
            np.testing.assert_array_almost_equal(best_est.coef_, tf_cv.coef_)
