"""
Cross-validation for trend filtering.

This module provides cross-validation functionality for automatic parameter selection.
"""

import numpy as np
from typing import Optional, Union, Sequence, Any, Generator, Tuple
import warnings
from .trendfilter import TrendFilter

try:
    from . import _trendfilter  # type: ignore

    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False

try:
    from sklearn.model_selection import KFold  # type: ignore

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available. Using simple train/test split for CV.")


class CVTrendFilter:
    """
    Cross-validated trend filtering for automatic parameter selection.

    This class performs k-fold cross-validation to select the optimal
    regularization parameter for trend filtering.

    Parameters
    ----------
    order : int, default=1
        Order of the trend filtering
    lambdas : array-like, optional
        Sequence of regularization parameters to try. If None, will be
        automatically generated.
    cv : int or cross-validation generator, default=5
        Cross-validation splitting strategy
    scoring : str, default='neg_mean_squared_error'
        Scoring metric ('neg_mean_squared_error', 'neg_mean_absolute_error')
    nlambda : int, default=50
        Number of lambda values to generate if lambdas is None
    random_state : int, default=None
        Random state for reproducibility
    max_iter : int, default=200
        Maximum number of ADMM iterations
    tol : float, default=1e-5
        Convergence tolerance
    method : str, default='auto'
        Solver method ('auto', 'sparse_qr', 'tridiag', 'kf')

    Attributes
    ----------
    best_lambda_ : float
        Best regularization parameter found by cross-validation
    best_score_ : float
        Best cross-validation score
    cv_scores_ : ndarray
        Cross-validation scores for each lambda value
    lambdas : ndarray
        Regularization parameters tested
    coef_ : ndarray
        Coefficients from the best model
    best_estimator_ : TrendFilter
        Trend filter fitted with the best lambda value
    """

    def __init__(
        self,
        order: int = 1,
        lambdas: Optional[Union[Sequence[float], np.ndarray]] = None,
        cv: Union[int, Any] = 5,
        scoring: str = "neg_mean_squared_error",
        nlambda: int = 50,
        random_state: Optional[int] = None,
        max_iter: int = 200,
        tol: float = 1e-5,
        method: str = "auto",
    ):
        self.order = order
        self.lambdas = lambdas
        self.cv = cv
        self.scoring = scoring
        self.nlambda = nlambda
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.method = method

    def fit(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "CVTrendFilter":
        """
        Fit cross-validated trend filter.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values
        x : array-like of shape (n_samples,), optional
            Sample locations
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        y = np.asarray(y, dtype=np.float64)
        n = len(y)

        if x is None:
            x = np.arange(n, dtype=np.float64)
        else:
            x = np.asarray(x, dtype=np.float64)

        if sample_weight is None:
            weights = np.ones(n, dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)

        # Generate lambda path if not provided
        if self.lambdas is None:
            lambda_path = self._generate_lambda_path(y, x, weights)
        else:
            lambda_path = np.asarray(self.lambdas, dtype=np.float64)

        # Perform cross-validation
        if HAS_CPP_BACKEND:
            self._fit_cpp_backend(y, x, weights, lambda_path)
        else:
            self._fit_python_backend(y, x, weights, lambda_path)

        # Find best lambda from cv_scores_
        mean_scores = (
            np.mean(self.cv_scores_, axis=1)
            if self.cv_scores_.ndim > 1
            else self.cv_scores_
        )
        if self.scoring in ["neg_mean_squared_error", "mse"]:
            best_idx = np.argmax(mean_scores)  # Higher is better for negative scores
        else:  # For other metrics that might be maximized
            best_idx = np.argmax(mean_scores)

        self.best_lambda_ = lambda_path[best_idx]
        self.best_score_ = mean_scores[best_idx]

        # Fit final model with best lambda
        self.best_estimator_ = TrendFilter(
            order=self.order,
            lambda_reg=self.best_lambda_,
            max_iter=self.max_iter,
            tol=self.tol,
            method=self.method,
        )
        self.best_estimator_.fit(y, x, sample_weight)

        # Store coefficients and lambda path for compatibility
        self.coef_ = self.best_estimator_.coef_
        self.lambdas = lambda_path

        return self

    def _fit_cpp_backend(
        self, y: np.ndarray, x: np.ndarray, weights: np.ndarray, lambda_path: np.ndarray
    ) -> None:
        """Use C++ backend for efficient cross-validation."""
        # For now, use the simple approach with multiple fits
        # TODO: Implement efficient CV in C++ backend
        self._fit_python_backend(y, x, weights, lambda_path)

    def _fit_python_backend(
        self, y: np.ndarray, x: np.ndarray, weights: np.ndarray, lambda_path: np.ndarray
    ) -> None:
        """Python implementation of cross-validation."""
        # Set up cross-validation
        if isinstance(self.cv, int):
            if HAS_SKLEARN:
                cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=42)
            else:
                cv_splitter = self._simple_kfold(len(y), self.cv)
        else:
            cv_splitter = self.cv

        # Perform cross-validation
        cv_scores = []
        all_fold_scores = []
        n_folds = 0

        for lambda_val in lambda_path:
            scores = []
            fold_count = 0

            for train_idx, test_idx in cv_splitter:
                # Fit on training set
                tf = TrendFilter(
                    order=self.order,
                    lambda_reg=lambda_val,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    method=self.method,
                )
                tf.fit(y[train_idx], x[train_idx], weights[train_idx])

                # Predict on test set (for trend filtering, this is interpolation)
                y_test_pred = self._interpolate_prediction(
                    x[train_idx], tf.coef_, x[test_idx]
                )

                # Compute score
                score = self._compute_score(y[test_idx], y_test_pred)
                scores.append(score)
                fold_count += 1

            if n_folds == 0:
                n_folds = fold_count

            cv_scores.append(np.mean(scores))
            # Pad scores to ensure consistent length if needed
            while len(scores) < n_folds:
                scores.append(np.nan)
            all_fold_scores.append(scores[:n_folds])

        # Create 2D array
        self.cv_scores_ = np.array(all_fold_scores)  # Shape: (n_lambdas, n_folds)

    def _simple_kfold(
        self, n: int, n_splits: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Simple k-fold implementation when sklearn is not available."""
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)

        fold_sizes = [
            n // n_splits + (1 if i < n % n_splits else 0) for i in range(n_splits)
        ]
        start = 0
        for fold_size in fold_sizes:
            end = start + fold_size
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            yield train_idx, test_idx
            start = end

    def _generate_lambda_path(
        self, y: np.ndarray, x: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Generate a path of lambda values for cross-validation."""
        if HAS_CPP_BACKEND:
            # Use C++ backend to compute lambda_max
            sqrt_weights = np.sqrt(weights)
            try:
                lambda_max = _trendfilter.get_lambda_max(x, y, sqrt_weights, self.order)
                lambda_min = lambda_max * 1e-4
                return np.logspace(
                    np.log10(lambda_min), np.log10(lambda_max), num=self.nlambda
                )
            except:
                # Fallback if C++ function fails
                pass

        # Fallback: generate based on data variance
        lambda_max = np.var(y)  # Upper bound based on data variance
        lambda_min = lambda_max * 1e-4  # Lower bound

        return np.logspace(np.log10(lambda_min), np.log10(lambda_max), num=self.nlambda)

    def _interpolate_prediction(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate trend filter fit to test points.

        This is a simple implementation. In practice, you might want
        more sophisticated interpolation.
        """
        if y_train.ndim > 1:
            y_train = y_train[:, 0]  # Take first column if multiple lambda values
        return np.interp(x_test, x_train, y_train).astype(np.float64)

    def _compute_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the scoring metric."""
        if self.scoring in ["neg_mean_squared_error", "mse"]:
            return -float(np.mean((y_true - y_pred) ** 2))  # Negative MSE
        elif self.scoring in ["neg_mean_absolute_error", "mae"]:
            return -float(np.mean(np.abs(y_true - y_pred)))  # Negative MAE
        else:
            raise ValueError(f"Unknown scoring metric: {self.scoring}")

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict using the best fitted model.

        Parameters
        ----------
        X : array-like, optional
            Not used, present for API consistency

        Returns
        -------
        y_pred : ndarray
            Predicted values
        """
        if not hasattr(self, "best_estimator_"):
            raise ValueError("Model must be fitted before prediction")

        return np.asarray(self.best_estimator_.predict(X), dtype=np.float64)

    def score(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> float:
        """
        Return the score of the prediction.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            True values
        X : array-like, optional
            Not used, present for API consistency

        Returns
        -------
        score : float
            Score of the prediction
        """
        if not hasattr(self, "best_estimator_"):
            raise ValueError("Model must be fitted before scoring")

        y_pred = self.predict(X)

        # Handle case where prediction returns multiple columns
        if y_pred.ndim > 1:
            # Take first column or reshape appropriately
            if y_pred.shape[1] == 1:
                y_pred = y_pred.ravel()
            else:
                # If there are multiple lambda solutions, take the best one
                y_pred = (
                    y_pred[:, 0] if hasattr(self, "best_estimator_") else y_pred.ravel()
                )
        if y.ndim > 1:
            y = y.ravel()

        # Ensure shapes match
        if y.shape != y_pred.shape:
            min_len = min(len(y), len(y_pred))
            y = y[:min_len]
            y_pred = y_pred[:min_len]

        return -float(np.mean((y - y_pred) ** 2))  # Return negative MSE
