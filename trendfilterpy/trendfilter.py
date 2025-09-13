"""
Trend filtering implementation.

This module provides the main TrendFilter class for univariate trend filtering.
"""

import numpy as np
from typing import Optional, Union, Tuple, Dict, Any
import warnings

try:
    from . import _trendfilter
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False
    warnings.warn("C++ backend not available, falling back to Python implementation")


class TrendFilter:
    """
    Univariate trend filtering with L1 penalty.
    
    Trend filtering fits a piecewise polynomial function to data by solving
    an optimization problem with an L1 penalty on the discrete derivatives.
    
    Parameters
    ----------
    order : int, default=1
        Order of the trend filtering (0=constant, 1=linear, 2=quadratic, etc.)
    lambda_reg : float or array-like, default=1.0
        Regularization parameter(s) controlling the smoothness vs. fidelity tradeoff.
        If array-like, will fit for multiple lambda values.
    nlambda : int, default=50
        Number of lambda values to use if lambda_reg is a scalar and auto-generation is used
    lambda_max : float, default=None
        Maximum lambda value for auto-generation
    lambda_min : float, default=None
        Minimum lambda value for auto-generation  
    lambda_min_ratio : float, default=1e-5
        Ratio of lambda_min to lambda_max for auto-generation
    max_iter : int, default=200
        Maximum number of ADMM iterations
    tol : float, default=1e-5
        Convergence tolerance
    method : str, default='auto'
        Solver method ('auto', 'sparse_qr', 'tridiag', 'kf')
    rho_scale : float, default=1.0
        ADMM penalty parameter scaling
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_samples,) or (n_samples, n_lambda)
        Fitted trend filtering coefficients
    alpha_ : ndarray of shape (n_samples-order-1,) or (n_samples-order-1, n_lambda)
        Fitted dual variables (discrete derivatives)
    lambda_ : ndarray of shape (n_lambda,)
        Lambda values used in fitting
    objective_ : ndarray of shape (n_lambda,)
        Objective function values
    n_iter_ : ndarray of shape (n_lambda,)
        Number of iterations for each lambda
    degrees_of_freedom_ : ndarray of shape (n_lambda,)
        Degrees of freedom for each lambda
    n_features_in_ : int
        Number of features seen during fit
    """
    
    def __init__(
        self, 
        order: int = 1, 
        lambda_reg: Union[float, np.ndarray] = 1.0,
        nlambda: int = 50,
        lambda_max: Optional[float] = None,
        lambda_min: Optional[float] = None,
        lambda_min_ratio: float = 1e-5,
        max_iter: int = 200,
        tol: float = 1e-5,
        method: str = 'auto',
        rho_scale: float = 1.0
    ):
        self.order = order
        self.lambda_reg = lambda_reg
        self.nlambda = nlambda
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.rho_scale = rho_scale
        
    def fit(self, y: np.ndarray, x: Optional[np.ndarray] = None, 
            sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the trend filter to data.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values
        x : array-like of shape (n_samples,), optional
            Sample locations. If None, assumes equally spaced points.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights. If None, assumes equal weights.
            
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
            
        if len(x) != n:
            raise ValueError("x and y must have the same length")
            
        if sample_weight is None:
            weights = np.ones(n, dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if len(weights) != n:
                raise ValueError("sample_weight must have the same length as y")
                
        self.n_features_in_ = n
        
        # Prepare lambda sequence
        if np.isscalar(self.lambda_reg):
            lambda_seq = np.zeros(self.nlambda)
            lambda_max = self.lambda_max if self.lambda_max is not None else -1.0
            lambda_min = self.lambda_min if self.lambda_min is not None else -1.0
        else:
            lambda_seq = np.asarray(self.lambda_reg, dtype=np.float64)
            self.nlambda = len(lambda_seq)
            lambda_max = -1.0
            lambda_min = -1.0
        
        # Choose solver method
        linear_solver = self._get_solver_code(self.method)
        
        if HAS_CPP_BACKEND:
            # Use C++ backend
            result = _trendfilter.admm_lambda_seq(
                x, y, weights, self.order, lambda_seq,
                nlambda=self.nlambda,
                lambda_max=lambda_max,
                lambda_min=lambda_min,  
                lambda_min_ratio=self.lambda_min_ratio,
                max_iter=self.max_iter,
                rho_scale=self.rho_scale,
                tol=self.tol,
                linear_solver=linear_solver
            )
            
            # Unpack results
            theta, alpha, lambda_used, objective, n_iter, dof = result
            
            self.coef_ = theta
            self.alpha_ = alpha
            self.lambda_ = lambda_used
            self.objective_ = np.array(objective)
            self.n_iter_ = np.array(n_iter)
            self.degrees_of_freedom_ = np.array(dof)
            
        else:
            # Use Python fallback
            warnings.warn("Using Python fallback implementation. Install C++ backend for better performance.")
            self.coef_ = self._fit_python(y, x, weights)
            self.lambda_ = lambda_seq if not np.isscalar(self.lambda_reg) else np.array([self.lambda_reg])
            
        return self
        
    def _get_solver_code(self, method: str) -> int:
        """Convert method string to solver code for C++ backend."""
        method_map = {
            'tridiag': 0,
            'sparse_qr': 1, 
            'kf': 2,
            'auto': 2  # Default to Kalman filter
        }
        return method_map.get(method, 2)
        
    def _fit_python(self, y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Python fallback implementation.
        """
        # Simple smoothing as placeholder
        from scipy.ndimage import gaussian_filter1d
        if np.isscalar(self.lambda_reg):
            sigma = 1.0 / np.sqrt(self.lambda_reg)
            return gaussian_filter1d(y, sigma=sigma)
        else:
            # For multiple lambdas, return multiple solutions
            results = []
            for lam in self.lambda_reg:
                sigma = 1.0 / np.sqrt(lam)
                results.append(gaussian_filter1d(y, sigma=sigma))
            return np.column_stack(results)
        
    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict using the fitted trend filter.
        
        For trend filtering, prediction typically returns the fitted values.
        
        Parameters
        ----------
        X : array-like, optional
            Not used, present for API consistency
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_lambda)
            Predicted values
        """
        if not hasattr(self, 'coef_'):
            raise ValueError("Model must be fitted before prediction")
            
        return self.coef_
    
    def get_best_lambda(self, criterion: str = 'cv') -> float:
        """
        Get the best lambda value based on a criterion.
        
        Parameters
        ----------
        criterion : str, default='cv'
            Criterion for selecting lambda ('cv', 'aic', 'bic', 'gcv')
            Currently only returns the middle lambda as placeholder.
            
        Returns
        -------
        best_lambda : float
            The best lambda value
        """
        if not hasattr(self, 'lambda_'):
            raise ValueError("Model must be fitted before selecting lambda")
            
        # Placeholder: return middle lambda
        # TODO: Implement proper model selection criteria
        return self.lambda_[len(self.lambda_) // 2]
    
    def get_coefficients_at_lambda(self, lambda_val: float) -> np.ndarray:
        """
        Get coefficients for a specific lambda value.
        
        Parameters
        ----------
        lambda_val : float
            Lambda value to retrieve coefficients for
            
        Returns
        -------
        coef : ndarray of shape (n_samples,)
            Coefficients at the specified lambda
        """
        if not hasattr(self, 'lambda_'):
            raise ValueError("Model must be fitted before getting coefficients")
            
        # Find closest lambda
        idx = np.argmin(np.abs(self.lambda_ - lambda_val))
        
        if self.coef_.ndim == 1:
            return self.coef_
        else:
            return self.coef_[:, idx]
