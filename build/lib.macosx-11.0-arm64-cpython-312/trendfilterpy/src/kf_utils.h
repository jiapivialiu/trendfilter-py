#ifndef KF_UTILS_H
#define KF_UTILS_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <tuple>

// Forward declarations
Eigen::MatrixXd computePtemp(const Eigen::MatrixXd& A, const Eigen::MatrixXd& P);
Eigen::MatrixXd smat_to_mat(const Eigen::SparseMatrix<double>& sparseMat, int k, bool equal_spaced);
void configure_denseD(const Eigen::VectorXd& x, Eigen::MatrixXd& denseD, Eigen::VectorXd& s_seq, 
                      const Eigen::SparseMatrix<double>& dk_mat, int k, bool equal_space);
std::tuple<Eigen::SparseMatrix<double>, Eigen::VectorXd, Eigen::MatrixXd> 
    configure_denseD_test(const Eigen::VectorXd& x, int k);
void f1step(double y, double c, double Z, double H, const Eigen::MatrixXd& A, 
  double RQR, Eigen::VectorXd& a, Eigen::MatrixXd& P, double& vt, double& Ft,
  Eigen::VectorXd& Kt);
void df1step(double y, double Z, double H, const Eigen::MatrixXd& A, double RQR,
  Eigen::VectorXd& a, Eigen::MatrixXd& P, Eigen::MatrixXd& Pinf, int& rankp,
  double& vt, double& Ft, double& Finf, Eigen::VectorXd& Kt, Eigen::VectorXd& Kinf);

#endif
