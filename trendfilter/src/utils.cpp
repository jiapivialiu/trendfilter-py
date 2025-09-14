#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/QR>
#include "utils.h"

typedef Eigen::COLAMDOrdering<int> Ord;

using Eigen::SparseMatrix;
using Eigen::SparseQR;
using Eigen::ArrayXd;

// Simplified implementations replacing R package dependencies
Eigen::VectorXd Dkv(const Eigen::VectorXd& v, int k, const Eigen::VectorXd& xd,
                    bool tf_weighting) {
  SparseMatrix<double> D = get_dk_mat(k, xd, tf_weighting);
  return D * v;
}

Eigen::VectorXd Dktv(const Eigen::VectorXd& v, int k, const Eigen::VectorXd& xd) {
  SparseMatrix<double> D = get_dk_mat(k, xd, false);
  return D.transpose() * v;
}

// Simplified soft thresholding for trend filtering
Eigen::VectorXd tf_dp(const Eigen::VectorXd& v, double lambda) {
  VectorXd result = v;
  for (int i = 0; i < result.size(); i++) {
    if (result(i) > lambda) {
      result(i) -= lambda;
    } else if (result(i) < -lambda) {
      result(i) += lambda;
    } else {
      result(i) = 0.0;
    }
  }
  return result;
}

Eigen::VectorXd tf_dp_weight(const Eigen::VectorXd& v, double lambda,
                             const Eigen::ArrayXd& w) {
  VectorXd result = v;
  for (int i = 0; i < result.size(); i++) {
    double weighted_lambda = lambda / w(i);
    if (result(i) > weighted_lambda) {
      result(i) -= weighted_lambda;
    } else if (result(i) < -weighted_lambda) {
      result(i) += weighted_lambda;
    } else {
      result(i) = 0.0;
    }
  }
  return result;
}

bool is_equal_space(const Eigen::VectorXd& x, double space_tolerance_ratio) {
  bool equal_space = true;
  int n = x.size();
  double averaged_diff = (x[n-1] - x[0]) / (n - 1);
  double space_tolerance = space_tolerance_ratio * averaged_diff;
  
  for (int i = 0; i < n - 1; ++i) {
    double diff = x[i+1] - x[i];
    // if any signal distance is greater or smaller than the averaged difference with a tolerance, return False
    if (diff < averaged_diff - space_tolerance || diff > averaged_diff + space_tolerance) {
      equal_space = false;
      break;
    }
  }
  return equal_space;
}

using Eigen::VectorXd;

/* General utilities */
Eigen::SparseMatrix<double> row_scale(
    Eigen::SparseMatrix<double> A,
    Eigen::ArrayXd v) {
  return v.matrix().asDiagonal() * A;
}
Eigen::MatrixXd row_scale(Eigen::MatrixXd A, Eigen::ArrayXd v) {
  return v.matrix().asDiagonal() * A;
}
Eigen::SparseMatrix<double> col_scale(
    Eigen::SparseMatrix<double> A,
    Eigen::ArrayXd v) {
  return A * v.matrix().asDiagonal();
}
Eigen::MatrixXd col_scale(Eigen::MatrixXd A, Eigen::ArrayXd v) {
  return A * v.matrix().asDiagonal();
}

/* Matrix construction */
Eigen::SparseMatrix<double> identity(int n) {
  SparseMatrix<double> Id(n, n);
  Id.setIdentity();
  return Id;
}

Eigen::SparseMatrix<double> diagonal(const Eigen::ArrayXd& diag) {
  int n = diag.size();
  SparseMatrix<double> D(n, n);
  D.diagonal() = diag;
  return D;
}

// Simplified implementation of difference matrix construction
Eigen::SparseMatrix<double> get_dk_mat(int k, const Eigen::VectorXd& xd, bool tf_weighting) {
  int n = xd.size();
  if (k == 0) {
    // Identity matrix for k=0
    SparseMatrix<double> D(n-1, n);
    for (int i = 0; i < n-1; i++) {
      D.insert(i, i) = -1.0;
      D.insert(i, i+1) = 1.0;
    }
    D.makeCompressed();
    return D;
  } else {
    // For higher order differences, we need to apply the difference operator k times
    SparseMatrix<double> D = get_dk_mat(0, xd, tf_weighting);  // Start with first difference
    for (int order = 1; order < k; order++) {
      VectorXd x_reduced(D.rows());
      for (int i = 0; i < D.rows(); i++) {
        x_reduced(i) = (xd(i) + xd(i+1)) / 2.0;  // Midpoints
      }
      SparseMatrix<double> D_next = get_dk_mat(0, x_reduced, tf_weighting);
      SparseMatrix<double> D_new = D_next * D;
      D = D_new;
    }
    return D;
  }
}

Eigen::SparseMatrix<double> get_penalty_mat(int k, const Eigen::VectorXd& xd) {
  return get_dk_mat(k+1, xd, true);
}

/* Polynomial subspace projection */
Eigen::VectorXd legendre_polynomial(
    const Eigen::VectorXd& x,
    int k,
    double a,
    double b) {
  ArrayXd xa = 2 * (x.array() - a) / (b - a) - 1;
  if (k == 0) {
    return VectorXd::Ones(x.size());
  } else if (k == 1) {
    return xa.matrix();
  } else if (k == 2) {
    return (1.5*xa.pow(2) - 0.5).matrix();
  } else if (k == 3) {
    return (2.5*xa.pow(3) - 1.5*xa).matrix();
  } else {
    throw std::invalid_argument("`k` must be 0, 1, 2, or 3.");
  }
}

Eigen::MatrixXd polynomial_basis(
    const Eigen::VectorXd& x,
    int k,
    double a = 0.0,
    double b = 1.0) {
  int n = x.size();
  MatrixXd basis_mat(n, k + 1);
  for (int j = 0; j < k + 1; j++) {
    basis_mat.col(j) = legendre_polynomial(x, j, a, b);
  }
  return basis_mat;
}

Eigen::VectorXd project_polynomials(
    const Eigen::VectorXd& x,
    const VectorXd& y,
    const ArrayXd& weights,
    int k) {
  Eigen::ColPivHouseholderQR<MatrixXd> qr;
  ArrayXd sqrt_weights = weights.sqrt();
  MatrixXd basis_mat = polynomial_basis(
    x, k, x.minCoeff(), x.maxCoeff()
  );
  // If this isn't accurate enough, can also use SVD.
  qr.compute(row_scale(basis_mat, sqrt_weights));
  VectorXd beta = qr.solve((y.array()*sqrt_weights).matrix());
  VectorXd projection = basis_mat*beta;
  // if (qr.info() > 0) {
  //  std::cerr << "Eigen QR solve returned nonzero exit status.\n";
  // }
  return projection;
}

/* Tridiagonal matrix solve */
Eigen::VectorXd tridiag_forward(
    const Eigen::VectorXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& c) {
  int n = a.size();
  Eigen::VectorXd cp(n - 1);

  // Forward sweep part 1
  cp[0] = c[0] / b[0];
  for (int i = 1; i < n - 1; i++) {
    cp[i] = c[i] / (b[i] - a[i]*cp[i - 1]);
  }
  return cp;
}

// Technically, construction of dp is also part of the forward sweep,
// but it makes sense to include it in "backsolve" since d can change
// over iterations.
Eigen::VectorXd tridiag_backsolve(
    const Eigen::VectorXd& a,
    const::VectorXd& b,
    const Eigen::VectorXd& cp,
    const Eigen::VectorXd& d) {
  int n = d.size();
  Eigen::VectorXd dp(n);
  Eigen::VectorXd x(n);

  // Forward sweep part 2
  dp[0] = d[0] / b[0];
  for (int i = 1; i < n; i++) {
    dp[i] = (d[i] - a[i]*dp[i - 1]) / (b[i] - a[i]*cp[i - 1]);
  }

  // Backsolve
  x[n - 1] = dp[n - 1];
  for (int i = n - 2; i >= 0; i--) {
    x[i] = dp[i] - cp[i]*x[i + 1];
  }
  return x;
}

std::tuple<Eigen::VectorXd,Eigen::VectorXd,Eigen::VectorXd> extract_tridiag(
    const Eigen::SparseMatrix<double>& A) {
  int n = A.cols();
  VectorXd a(n);
  VectorXd b(n);
  VectorXd c(n);
  // Extract diagonal into b
  b = A.diagonal();
  // Extract (-1)-diagonal into a
  a[0] = 0;
  for (int i = 1; i < n; i++) {
    a[i] = A.coeff(i, i - 1);
  }
  // Extract (+1)-diagonal into c
  for (int i = 0; i < n-1; i++) {
    c[i] = A.coeff(i, i + 1);
  }
  c[n - 1] = 0;
  return std::make_tuple(a, b, c);
}

/* Miscellaneous */
double get_lambda_max(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    const Eigen::ArrayXd& weights,
    int k) {
  ArrayXd sqrt_weights = weights.sqrt();
  SparseMatrix<double> ck1_mat = get_dk_mat(k + 1, x, true);
  SparseQR<SparseMatrix<double>, Ord> qr;
  qr.compute(col_scale(ck1_mat, sqrt_weights.inverse()).transpose());
  VectorXd u_infty = qr.solve((y.array() * sqrt_weights).matrix());
  return u_infty.lpNorm<Eigen::Infinity>();
}

int calc_degrees_of_freedom(const Eigen::VectorXd& v, int k, double tol) {
  int dof = k + 1;
  for (int i = 1; i < v.size(); i++) {
    if (std::abs(v(i) - v(i - 1)) > tol) {
      dof++;
    }
  }
  return dof;
}

Eigen::VectorXd get_lambda_seq_r(
    Eigen::VectorXd lambda,
    double lambda_max,
    double lambda_min,
    double lambda_min_ratio,
    int n_lambda) {

  get_lambda_seq(lambda, lambda_max, lambda_min, lambda_min_ratio, n_lambda);
  return lambda;
}

void get_lambda_seq(
    Eigen::VectorXd& lambda,
    double lambda_max,
    double lambda_min,
    double lambda_min_ratio,
    int n_lambda) {

  if (!(lambda.array() < 1e-12).all()) {
    lambda_min = lambda.minCoeff();
    lambda_max = lambda.maxCoeff();
    n_lambda = lambda.size();
  } else {
    double lmpad = lambda_min_ratio * lambda_max;
    lambda_min = (lambda_min < 0) ? lmpad : lambda_min;
    double ns = static_cast<double>(n_lambda) - 1;
    double p = 0.0;
    lambda[0] = lambda_max;
    if (lambda_min > 1e-20) {
      p = pow(lambda_min / lambda_max, 1 / ns);
      for (int i = 1; i < n_lambda; i++) lambda[i] = lambda[i - 1] * p;
    } else {
      ns -= 1;
      p = pow(lmpad / lambda_max, 1 / ns);
      for (int i = 1; i < n_lambda - 1; i++) lambda[i] = lambda[i - 1] * p;
      lambda[n_lambda - 1] = lambda_min;
    }
  }
}