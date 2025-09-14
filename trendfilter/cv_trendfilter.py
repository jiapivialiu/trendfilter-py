"""
Cross-validation for trend filtering.

This module provides cross-validation functionality for automatic parameter selection.
"""

import numpy as np
from typing import Optional, Union, Sequence
import warnings
from .trendfilter import TrendFilter

try:
    from . import _trendfilter
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False

try:
    from sklearn.model_selection import KFold
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
    lambda_path : array-like, optional
        Sequence of regularization parameters to try. If None, will be
        automatically generated.
    cv : int or cross-validation generator, default=5
        Cross-validation splitting strategy
    scoring : str, default='mse'
        Scoring metric ('mse', 'mae')
    n_jobs : int, default=None
        Number of parallel jobs (not yet implemented)
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
    lambda_path_ : ndarray
        Regularization parameters tested
    best_estimator_ : TrendFilter
        Trend filter fitted with the best lambda value
    """
    
    def __init__(
        self,
        order: int = 1,
        lambda_path: Optional[Sequence[float]] = None,
        cv: Union[int, object] = 5,
        scoring: str = 'mse',
        n_jobs: Optional[int] = None,
        max_iter: int = 200,
        tol: float = 1e-5,
        method: str = 'auto'
    ):
        self.order = order
        self.lambda_path = lambda_path
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        
    def fit(self, y: np.ndarray, x: Optional[np.ndarray] = None, 
            sample_weight: Optional[np.ndarray] = None) -> 'CVTrendFilter':
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
        if self.lambda_path is None:
            self.lambda_path_ = self._generate_lambda_path(y, x, weights)
        else:
            self.lambda_path_ = np.asarray(self.lambda_path, dtype=np.float64)
            
        # Perform cross-validation
        if HAS_CPP_BACKEND:
            self._fit_cpp_backend(y, x, weights)
        else:
            self._fit_python_backend(y, x, weights)
        
        # Find best lambda
        if self.scoring == 'mse':
            best_idx = np.argmin(self.cv_scores_)
        else:  # For other metrics that might be maximized
            best_idx = np.argmax(self.cv_scores_)
            
        self.best_lambda_ = self.lambda_path_[best_idx]
        self.best_score_ = self.cv_scores_[best_idx]
        
        # Fit final model with best lambda
        self.best_estimator_ = TrendFilter(
            order=self.order, 
            lambda_reg=self.best_lambda_,
            max_iter=self.max_iter,
            tol=self.tol,
            method=self.method
        )
        self.best_estimator_.fit(y, x, sample_weight)
        
        return self
        
    def _fit_cpp_backend(self, y: np.ndarray, x: np.ndarray, weights: np.ndarray):
        """Use C++ backend for efficient cross-validation."""
        # For now, use the simple approach with multiple fits
        # TODO: Implement efficient CV in C++ backend
        self._fit_python_backend(y, x, weights)
        
    def _fit_python_backend(self, y: np.ndarray, x: np.ndarray, weights: np.ndarray):
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
        
        for lambda_val in self.lambda_path_:
            scores = []
            
            for train_idx, test_idx in cv_splitter:
                # Fit on training set
                tf = TrendFilter(
                    order=self.order, 
                    lambda_reg=lambda_val,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    method=self.method
                )
                tf.fit(y[train_idx], x[train_idx], weights[train_idx])
                
                # Predict on test set (for trend filtering, this is interpolation)
                y_test_pred = self._interpolate_prediction(
                    x[train_idx], tf.coef_, x[test_idx]
                )
                
                # Compute score
                score = self._compute_score(y[test_idx], y_test_pred)
                scores.append(score)
                
            cv_scores.append(np.mean(scores))
            
        self.cv_scores_ = np.array(cv_scores)
        
    def _simple_kfold(self, n: int, n_splits: int):
        """Simple k-fold implementation when sklearn is not available."""
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        fold_size = n // n_splits
        for i in range(n_splits):
            start = i * fold_size
            end = start + fold_size if i < n_splits - 1 else n
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            yield train_idx, test_idx
        
    def _generate_lambda_path(self, y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Generate a path of lambda values for cross-validation."""
        if HAS_CPP_BACKEND:
            # Use C++ backend to compute lambda_max
            sqrt_weights = np.sqrt(weights)
            try:
                lambda_max = _trendfilter.get_lambda_max(x, y, sqrt_weights, self.order)
                lambda_min = lambda_max * 1e-4
                return np.logspace(np.log10(lambda_min), np.log10(lambda_max), num=50)
            except:
                # Fallback if C++ function fails
                pass
        
        # Fallback: generate based on data variance
        lambda_max = np.var(y)  # Upper bound based on data variance
        lambda_min = lambda_max * 1e-4  # Lower bound
        
        return np.logspace(
            np.log10(lambda_min), 
            np.log10(lambda_max), 
            num=50
        )
        
    def _interpolate_prediction(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_test: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate trend filter fit to test points.
        
        This is a simple implementation. In practice, you might want
        more sophisticated interpolation.
        """
        if y_train.ndim > 1:
            y_train = y_train[:, 0]  # Take first column if multiple lambda values
        return np.interp(x_test, x_train, y_train)
        
    def _compute_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the scoring metric."""
        if self.scoring == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.scoring == 'mae':
            return np.mean(np.abs(y_true - y_pred))
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
        if not hasattr(self, 'best_estimator_'):
            raise ValueError("Model must be fitted before prediction")
            
        return self.best_estimator_.predict(X)
