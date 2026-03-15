#ifndef QW_PDHG_SOLVER_H_
#define QW_PDHG_SOLVER_H_

// ============================================================
// GPU PDHG (Primal-Dual Hybrid Gradient) LP solver
//
// Solves:  min c(x)  s.t. Ax = b, l <= x <= u
// where c(x) is separable convex (Abs, Quad, Linear, Zero).
//
// Uses cuSPARSE for SpMV (A*x and A^T*y).
// Each iteration: 2 SpMVs + 3 element-wise kernels = ~10μs.
//
// Based on PDLP/cuPDLP (Beale-Orchard-Hays Prize 2024).
// ============================================================

#include <vector>
#include <cstdint>

namespace qw {
namespace cuda {

// Cost types (reuse from admm_solver.h)
enum PDHGCostType : int {
    PDHG_COST_ZERO = 0,
    PDHG_COST_ABS = 1,    // w * |x - target|
    PDHG_COST_QUAD = 2,   // w * (x - target)^2
    PDHG_COST_LINEAR = 3, // w * x
};

// LP problem: min c(x) s.t. Ax = b, l <= x <= u
struct PDHGProblem {
    int num_rows = 0;     // constraints (nodes)
    int num_cols = 0;     // variables (edges)

    // Constraint matrix A in CSR format (num_rows × num_cols)
    std::vector<int> csr_row_ptr;    // [num_rows + 1]
    std::vector<int> csr_col_idx;    // [nnz]
    std::vector<float> csr_values;   // [nnz] (+1 or -1 for incidence)

    // Also store A^T in CSR for fast A^T*y
    std::vector<int> csc_col_ptr;    // [num_cols + 1] (CSR of A^T)
    std::vector<int> csc_row_idx;    // [nnz]
    std::vector<float> csc_values;   // [nnz]

    // RHS: Ax = b
    std::vector<float> rhs;          // [num_rows]

    // Bounds: l <= x <= u
    std::vector<float> lower;        // [num_cols]
    std::vector<float> upper;        // [num_cols]

    // Cost function per variable
    std::vector<int> cost_type;      // [num_cols] PDHGCostType
    std::vector<float> cost_target;  // [num_cols]
    std::vector<float> cost_weight;  // [num_cols]
};

struct PDHGResult {
    std::vector<float> x;            // primal solution [num_cols]
    std::vector<float> y;            // dual solution [num_rows]
    int iterations;
    float primal_residual;           // ||Ax - b||
    float dual_residual;
    float objective;
    float time_ms;
};

// Solve LP with GPU PDHG
PDHGResult pdhg_solve(
    const PDHGProblem& prob,
    int max_iter = 2000,
    float tol = 1e-4f);

// Solve with sin² penalty annealing (converges to integers without rounding)
// penalty(x) = lambda * sin²(π * x), annealed from lambda_start to lambda_end
PDHGResult pdhg_solve_sinpen(
    const PDHGProblem& prob,
    int max_iter = 3000,
    float tol = 1e-3f,
    float lambda_start = 0.0f,
    float lambda_end = 50.0f);

// Feasibility pump: alternate between LP solve and rounding
PDHGResult pdhg_feasibility_pump(
    const PDHGProblem& prob,
    int max_pumps = 20,
    int pdhg_iters_per_pump = 500,
    float tol = 1e-3f);

} // namespace cuda
} // namespace qw

#endif
