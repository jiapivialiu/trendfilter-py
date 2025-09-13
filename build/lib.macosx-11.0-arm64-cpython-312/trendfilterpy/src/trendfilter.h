#ifndef TRENDFILTER_H
#define TRENDFILTER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <tuple>
#include <vector>

// Main trend filtering functions
double tf_gauss_loss(const Eigen::VectorXd& y,
                     const Eigen::VectorXd& theta,
                     const Eigen::ArrayXd& weights);

double tf_penalty(const Eigen::VectorXd& theta, const Eigen::VectorXd& xd, double lam, int k);

double tf_objective(const Eigen::VectorXd& y, const Eigen::VectorXd& theta,
                    const Eigen::VectorXd& xd,
                    const Eigen::ArrayXd& weights, double lam, int k);

std::tuple<Eigen::VectorXd,int> init_theta_nullspace(const Eigen::VectorXd& y, 
    const Eigen::SparseMatrix<double>& penalty_mat);

Eigen::VectorXd init_u(const Eigen::VectorXd& residual, const Eigen::VectorXd& xd, int k,
    const Eigen::ArrayXd& weights);

void admm_single_lambda(int n, const Eigen::VectorXd& y, const Eigen::VectorXd& xd,
  const Eigen::ArrayXd& weights, int k, Eigen::Ref<Eigen::VectorXd> theta,
  Eigen::Ref<Eigen::VectorXd> alpha, Eigen::Ref<Eigen::VectorXd> u, int& iter,
  double& obj_val, const Eigen::SparseMatrix<double>& dk_mat_sq,
  const Eigen::MatrixXd& denseD, const Eigen::VectorXd& s_seq, double lam,
  int max_iter, double rho, double tol, int linear_solver,
  bool equal_space);

// Main interface function
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, std::vector<double>, 
           std::vector<int>, std::vector<int>> 
admm_lambda_seq(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    const Eigen::ArrayXd& weights,
    int k,
    Eigen::VectorXd lambda,
    int nlambda = 50,
    double lambda_max = -1.0,
    double lambda_min = -1.0,
    double lambda_min_ratio = 1e-5,
    int max_iter = 200,
    double rho_scale = 1.0,
    double tol = 1e-5,
    int linear_solver = 2,
    double space_tolerance_ratio = -1.0);

#endif
