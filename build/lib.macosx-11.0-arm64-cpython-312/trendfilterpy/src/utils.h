#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <tuple>

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using Eigen::MatrixXd;

/* Matrix construction */
Eigen::SparseMatrix<double> identity(int n);
Eigen::SparseMatrix<double> diagonal(const Eigen::ArrayXd& diag);
Eigen::SparseMatrix<double> get_dk_mat(int k, const Eigen::VectorXd& xd, bool tf_weighting);
Eigen::SparseMatrix<double> get_penalty_mat(int k, const Eigen::VectorXd& xd);
Eigen::VectorXd legendre_polynomial(const Eigen::VectorXd& x, int k, double a, double b);

/* Polynomial subspace projection */
Eigen::VectorXd project_polynomials(const Eigen::VectorXd& x, const VectorXd& y,
    const Eigen::ArrayXd& weights, int k);

/* Tridiagonal matrix solve */
std::tuple<Eigen::VectorXd,Eigen::VectorXd,Eigen::VectorXd> extract_tridiag(
    const Eigen::SparseMatrix<double>& A);
Eigen::VectorXd tridiag_forward(const Eigen::VectorXd& a,
        const Eigen::VectorXd& b, const Eigen::VectorXd& c);
Eigen::VectorXd tridiag_backsolve(
        const Eigen::VectorXd& a, const Eigen::VectorXd& b,
            const Eigen::VectorXd& cp, const Eigen::VectorXd& d);

// Lambda sequence
double get_lambda_max(const Eigen::VectorXd& x, const Eigen::VectorXd& y,
                      const Eigen::ArrayXd& sqrt_weights, int k);
void get_lambda_seq(Eigen::VectorXd& lambda, double lambda_max,
                    double lambda_min, double lambda_min_ratio, int n_lambda);
Eigen::VectorXd get_lambda_seq_r(Eigen::VectorXd lambda, double lambda_max,
                                 double lambda_min, double lambda_min_ratio,
                                 int n_lambda);

int calc_degrees_of_freedom(const Eigen::VectorXd& v, int k, double tol = 1e-8);

bool is_equal_space(const Eigen::VectorXd& x, double space_tolerance_ratio);

// Workarounds for interfacing with dspline / tvdenoising
Eigen::VectorXd Dkv(const Eigen::VectorXd& v, int k, const Eigen::VectorXd& xd,
                    bool tf_weighting = false);
Eigen::VectorXd Dktv(const Eigen::VectorXd& v, int k, const Eigen::VectorXd& xd);
Eigen::VectorXd tf_dp(const Eigen::VectorXd& v, double lambda);
Eigen::VectorXd tf_dp_weight(const Eigen::VectorXd& v, double lambda, const Eigen::ArrayXd& w);

#endif
