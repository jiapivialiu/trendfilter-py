#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "utils.h"
#include "linearsystem.h"
#include "kf_utils.h"
#include "trendfilter.h"

typedef Eigen::COLAMDOrdering<int> Ord;

using Eigen::SparseMatrix;
using Eigen::SparseQR;
using Eigen::ArrayXd;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Map;

LinearSystem linear_system;

double tf_gauss_loss(const VectorXd& y,
                     const VectorXd& theta,
                     const ArrayXd& weights) {
  return 0.5 * ((y - theta).array() * weights.sqrt()).matrix().squaredNorm();
}

double tf_penalty(const VectorXd& theta, const Eigen::VectorXd& xd, double lam, int k) {
  Eigen::VectorXd Dv = Dkv(theta, k + 1, xd, true);
  return lam * Dv.lpNorm<1>();
}

// Gaussian only
double tf_objective(const VectorXd& y, const VectorXd& theta,
                    const Eigen::VectorXd& xd,
                    const ArrayXd& weights, double lam, int k) {
  return tf_gauss_loss(y, theta, weights) + tf_penalty(theta, xd, lam, k);
}

// Matches initialization from package glmgen
// Not used in these ADMM updates (projection via Legendre polynomials is
// more stable, but left here as a reference implementation).
std::tuple<VectorXd,int> init_theta_nullspace(const VectorXd& y, const
    SparseMatrix<double>& penalty_mat) {
  SparseQR<SparseMatrix<double>, Ord> qr;
  qr.compute(penalty_mat.transpose());
  // Project y onto rowspace of C^{k+1}
  VectorXd beta = qr.solve(y);
  // Subtract residual off rowspace-projection to obtain nullspace projection.
  return std::make_tuple(y - (penalty_mat.transpose()) * beta, int(qr.info()));
}

VectorXd init_u(const VectorXd& residual, const Eigen::VectorXd& xd, int k,
    const ArrayXd& weights) {
  // (Dk)^T u = y - \theta, from stationarity
  SparseQR<SparseMatrix<double>, Ord> qr;
  SparseMatrix<double> dk_mat = get_dk_mat(k, xd, false);
  qr.compute(dk_mat.transpose());
  return qr.solve((residual.array()*weights).matrix());
}

void admm_single_lambda(int n, const Eigen::VectorXd& y, const Eigen::VectorXd& xd,
  const Eigen::ArrayXd& weights, int k, Eigen::Ref<Eigen::VectorXd> theta,
  Eigen::Ref<Eigen::VectorXd> alpha, Eigen::Ref<Eigen::VectorXd> u, int& iter,
  double& obj_val, const Eigen::SparseMatrix<double>& dk_mat_sq,
  const Eigen::MatrixXd& denseD, const Eigen::VectorXd& s_seq, double lam,
  int max_iter, double rho, double tol = 1e-5, int linear_solver = 2,
  bool equal_space = false) {
  // Initialize internals
  VectorXd tmp(n-k);
  VectorXd Dth_tmp(alpha.size());
  VectorXd alpha_old(alpha);
  VectorXd wy = (y.array()*weights).matrix();
  double rr, ss;

  // LinearSystem linear_system;
  // Technically, can form one SparseQR object, analyze the pattern once,
  // and then re-use it.
  // So call analyzePattern once, and then factorize repeatedly.
  // https://eigen.tuxfamily.org/dox/classEigen_1_1SparseQR.html#aba8ae81fd3d4ce9139eccb6b7a0256b2
  linear_system.construct(y, weights, k, rho, dk_mat_sq, denseD, s_seq, linear_solver);
  linear_system.compute(linear_solver);

  // Perform ADMM updates
  int computation_info;
  iter = 0;
  // double best_objective = tf_objective(y, theta, xd, weights, lam, k);
  // VectorXd best_theta = theta;
  for (iter = 1; iter < max_iter; iter++) {
        if (iter % 1000 == 0) {
            // Check for user interruption could be added here if needed
        }    // theta update
    std::tie(theta, computation_info) = linear_system.solve(y, weights,
        alpha + u, k, xd, rho, denseD, s_seq, linear_solver, equal_space);
    // if (computation_info > 1) {
    //  std::cerr << "Eigen Sparse QR solve returned nonzero exit status.\n";
    // }
    Dth_tmp = Dkv(theta, k, xd);
    tmp = Dth_tmp - u;
    // alpha update
    alpha = tf_dp(tmp, lam / rho);
    // u update
    u += alpha - Dth_tmp;
    // double cur_objective = tf_objective(y, theta, xd, weights, lam, k);

    // Check for convergence
    rr = (Dth_tmp - alpha).norm() / alpha.size();
    ss = rho * Dktv(alpha - alpha_old, k, xd).norm() / dk_mat_sq.rows();
    alpha_old = alpha;
    if (rr < tol && ss < tol) break;
  }
  obj_val = tf_objective(y, theta, xd, weights, lam, k);
  // Rcpp::Rcout << iter << ": rr = " << rr << " ss = " << ss << std::endl;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, std::vector<double>, 
           std::vector<int>, std::vector<int>> 
admm_lambda_seq(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    const Eigen::ArrayXd& weights,
    int k,
    Eigen::VectorXd lambda,
    int nlambda,
    double lambda_max,
    double lambda_min,
    double lambda_min_ratio,
    int max_iter,
    double rho_scale,
    double tol,
    int linear_solver,
    double space_tolerance_ratio) {

  int n = x.size();

  if (lambda[0] < tol / 100 && lambda_max <= 0) {
    lambda_max = get_lambda_max(x, y, weights, k);
  }
  get_lambda_seq(lambda, lambda_max, lambda_min, lambda_min_ratio, nlambda);

  Eigen::MatrixXd theta(n, nlambda);
  std::vector<double> objective_val(nlambda);
  std::vector<int> iters(nlambda);
  std::vector<int> dof(nlambda);

  // Use DP solution for k=0.
  if (k == 0) {
    for (int i = 0; i < nlambda; i++) {
      theta.col(i) = tf_dp_weight(y, lambda[i], weights);
      objective_val[i] = tf_objective(y, theta.col(i), x, weights, lambda[i], k);
      dof[i] = calc_degrees_of_freedom(theta.col(i), k);
    }
    Eigen::MatrixXd alpha = Eigen::MatrixXd::Zero(n-1, nlambda);
    return std::make_tuple(theta, alpha, lambda, objective_val, iters, dof);
  }


  // Initialize difference matrices and other helper objects
  SparseMatrix<double> dk_mat = get_dk_mat(k, x, false);
  SparseMatrix<double> dk_mat_sq = dk_mat.transpose() * dk_mat;
  // Kalman filter objects
  bool equal_space = false;
  MatrixXd denseD;
  VectorXd s_seq;
  // configure `denseD` if using Kalman filter to contain the information in
  //  the first k nonzero columns in `dk_mat`. If evenly spaced, resize it to 1*k.
  if (linear_solver == 2) {
    // check if `x` is equally spaced
    equal_space = is_equal_space(x, space_tolerance_ratio < 0 ? std::sqrt(
      Eigen::NumTraits<double>::epsilon()) : space_tolerance_ratio);
    // initialize with the size of nonzero values in `dk_mat`
    denseD = MatrixXd::Zero(n - k, k + 1);
    // if using Kalman filter, save the rightmost nonzero value per row in
    //  `dk_mat` for unevenly space signals. For equally spaced signals,
    //  simplify it to one value.
    s_seq = equal_space ? VectorXd::Zero(1) : VectorXd::Zero(n);
    configure_denseD(x, denseD, s_seq, dk_mat, k, equal_space);
  }

  Eigen::MatrixXd alpha(n-k, nlambda);

  // Initialize ADMM variables
  // Project onto Legendre polynomials to initialize for largest lambda.
  theta.col(0) = project_polynomials(x, y, weights, k);
  alpha.col(0) = Dkv(theta.col(0), k, x);
  VectorXd u = init_u((theta.col(0) - y)/(lambda[0]*rho_scale), x, k, weights);


  for (int i = 0; i < nlambda; i++) {
    admm_single_lambda(n, y, x, weights, k,
      theta.col(i), alpha.col(i), u,
      iters[i], objective_val[i],
      dk_mat_sq, denseD, s_seq, lambda[i], max_iter, lambda[i]*rho_scale,
      tol, linear_solver, equal_space);
    dof[i] = calc_degrees_of_freedom(alpha.col(i), k);
    if (i + 1 < nlambda) {
      theta.col(i + 1) = theta.col(i);
      alpha.col(i + 1) = alpha.col(i);
    }
  }
  
  return std::make_tuple(theta, alpha, lambda, objective_val, iters, dof);
}

// The below is legacy code, currently unused.

// Rcpp::List admm_single_lambda_with_tracking(NumericVector x,
//     Eigen::VectorXd& y, const Eigen::ArrayXd& weights, int k,
//     double lam, int max_iter, double rho,
//     bool tridiag=false) {
//
//   int n = x.size();
//
//   // Initialize difference matrices and other helper objects
//   SparseMatrix<double> penalty_mat = get_penalty_mat(k+1, x);
//   SparseMatrix<double> dk_mat = get_dk_mat(k, x, false);
//   VectorXd wy = (y.array()*weights).matrix();
//
//   // Initialize theta, alpha, u
//   VectorXd theta = project_polynomials(x, y, weights, k);
//   VectorXd alpha = dk_mat * theta;
//   VectorXd u = init_u((theta-y)/rho, x, k, weights);
//   VectorXd tmp(n-k);
//
//   // Form Gram matrix and set up linear system for theta update
//   SparseMatrix<double> A = rho * (dk_mat.transpose() * dk_mat);
//   A.diagonal().array() += weights;
//   A.makeCompressed();
//
//   if (tridiag & (k != 1)) {
//     throw std::invalid_argument("`tridiag` can only be used with k=1.");
//   }
//   LinearSystem linear_system;
//   linear_system.compute(A, tridiag);
//
//   // Perform ADMM updates
//   int computation_info;
//   MatrixXd theta_mat(n, max_iter);
//   VectorXd objective_vec(max_iter);
//   int iter = 0;
//   theta_mat.col(iter) = theta;
//   objective_vec[iter] = tf_objective(y, theta, x, weights, lam, k);
//   VectorXd best_theta = theta;
//   double best_objective = objective_vec[iter];
//   for (iter = 1; iter < max_iter; iter++) {
//     // theta update
//     std::tie(theta, computation_info) = linear_system.solve(
//         wy + rho * (dk_mat.transpose() * (alpha + u)),
//         tridiag);
//     // if (computation_info > 1) {
//     //   std::cerr << "Eigen Sparse QR solve returned nonzero exit status.\n";
//     // }
//     tmp = dk_mat*theta - u;
//     // alpha update
//     alpha = tf_dp(tmp, lam/rho);
//     // u update
//     u += alpha - dk_mat*theta;
//     objective_vec[iter] = tf_objective(y, theta, x, weights, lam, k);
//     theta_mat.col(iter) = theta;
//     if (objective_vec[iter] < best_objective) {
//       best_objective = objective_vec[iter];
//       best_theta = theta;
//     }
//   }
//   // TODO: Implement stopping criterion
//   Rcpp::List return_list = Rcpp::List::create(
//       Rcpp::Named("theta") = best_theta,
//       Rcpp::Named("objective") = objective_vec,
//       Rcpp::Named("theta_mat") = theta_mat,
//       Rcpp::Named("computation_info") = computation_info);
//   return return_list;
// }
