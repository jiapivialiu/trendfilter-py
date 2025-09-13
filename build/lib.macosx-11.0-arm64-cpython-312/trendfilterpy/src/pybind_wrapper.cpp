#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "trendfilter.h"
#include "utils.h"
#include "kf_utils.h"
#include "linearsystem.h"

namespace py = pybind11;

PYBIND11_MODULE(_trendfilter, m) {
    m.doc() = "Fast trend filtering with C++ backend";

    // Main trend filtering function
    m.def("admm_lambda_seq", &admm_lambda_seq,
          "Solve trend filtering for a sequence of lambda values",
          py::arg("x"), py::arg("y"), py::arg("weights"), py::arg("k"),
          py::arg("lambda"), py::arg("nlambda") = 50,
          py::arg("lambda_max") = -1.0, py::arg("lambda_min") = -1.0,
          py::arg("lambda_min_ratio") = 1e-5, py::arg("max_iter") = 200,
          py::arg("rho_scale") = 1.0, py::arg("tol") = 1e-5,
          py::arg("linear_solver") = 2, py::arg("space_tolerance_ratio") = -1.0);

    // Utility functions
    m.def("get_dk_mat", &get_dk_mat,
          "Construct difference matrix",
          py::arg("k"), py::arg("x"), py::arg("tf_weighting") = false);

    m.def("get_penalty_mat", &get_penalty_mat,
          "Construct penalty matrix",
          py::arg("k"), py::arg("x"));

    m.def("project_polynomials", &project_polynomials,
          "Project data onto polynomial subspace",
          py::arg("x"), py::arg("y"), py::arg("weights"), py::arg("k"));

    m.def("get_lambda_max", &get_lambda_max,
          "Compute maximum lambda value",
          py::arg("x"), py::arg("y"), py::arg("sqrt_weights"), py::arg("k"));

    m.def("calc_degrees_of_freedom", &calc_degrees_of_freedom,
          "Calculate degrees of freedom",
          py::arg("v"), py::arg("k"), py::arg("tol") = 1e-8);

    m.def("is_equal_space", &is_equal_space,
          "Check if x values are equally spaced",
          py::arg("x"), py::arg("space_tolerance_ratio"));

    // Difference operators
    m.def("Dkv", &Dkv,
          "Apply k-th order difference operator",
          py::arg("v"), py::arg("k"), py::arg("x"), py::arg("tf_weighting") = false);

    m.def("Dktv", &Dktv,
          "Apply transpose of k-th order difference operator",
          py::arg("v"), py::arg("k"), py::arg("x"));

    // Soft thresholding
    m.def("tf_dp", &tf_dp,
          "Soft thresholding operator",
          py::arg("v"), py::arg("lambda"));

    m.def("tf_dp_weight", &tf_dp_weight,
          "Weighted soft thresholding operator",
          py::arg("v"), py::arg("lambda"), py::arg("weights"));

    // Test functions
    m.def("configure_denseD_test", &configure_denseD_test,
          "Test function for dense D configuration",
          py::arg("x"), py::arg("k"));

    // Objective functions
    m.def("tf_gauss_loss", &tf_gauss_loss,
          "Gaussian loss function",
          py::arg("y"), py::arg("theta"), py::arg("weights"));

    m.def("tf_penalty", &tf_penalty,
          "Trend filtering penalty",
          py::arg("theta"), py::arg("x"), py::arg("lambda"), py::arg("k"));

    m.def("tf_objective", &tf_objective,
          "Complete trend filtering objective",
          py::arg("y"), py::arg("theta"), py::arg("x"),
          py::arg("weights"), py::arg("lambda"), py::arg("k"));
}
