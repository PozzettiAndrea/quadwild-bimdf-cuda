#ifndef QW_ADMM_SOLVER_H_
#define QW_ADMM_SOLVER_H_

// ============================================================
// GPU ADMM solver for LP relaxation of BiMDF quantization
//
// Solves the convex relaxation (drop integrality) of:
//   minimize   Σ cost_e(f_e)
//   subject to Σ sign(e,n)*f_e = demand_n   ∀ node n
//              lower_e ≤ f_e ≤ upper_e       ∀ edge e
//
// where cost_e is AbsDeviation, QuadDeviation, or Zero.
//
// ADMM splitting:
//   x-update: proximal on cost (parallel per edge)
//   z-update: project onto flow conservation (parallel per node)
//   u-update: dual update (parallel per edge)
//
// Returns fractional solution to be rounded and used as warm-start.
// ============================================================

#include <vector>
#include <cstdint>

namespace qw {
namespace cuda {

// Cost type enum (same as bimdf_cuda.h)
enum ADMMCostType : int {
    ADMM_COST_ZERO = 0,
    ADMM_COST_ABS = 1,    // w * |f - target|
    ADMM_COST_QUAD = 2,   // w * (f - target)^2
    ADMM_COST_LINEAR = 3, // w * f
};

// Flat problem description for GPU upload
struct ADMMProblem {
    int num_nodes = 0;
    int num_edges = 0;

    // Edge endpoints
    std::vector<int> edge_u;          // [num_edges]
    std::vector<int> edge_v;          // [num_edges]
    std::vector<int> edge_sign_u;     // [num_edges] +1 (head) or -1 (tail)
    std::vector<int> edge_sign_v;     // [num_edges] +1 (head) or -1 (tail)

    // Cost function per edge
    std::vector<int> edge_cost_type;  // [num_edges] ADMMCostType
    std::vector<float> edge_target;   // [num_edges]
    std::vector<float> edge_weight;   // [num_edges]

    // Bounds
    std::vector<float> edge_lower;    // [num_edges]
    std::vector<float> edge_upper;    // [num_edges] (use 1e6 for unbounded)

    // Node demands
    std::vector<float> node_demand;   // [num_nodes]

    // Node-edge incidence (CSR format, built by caller)
    std::vector<int> node_edge_offsets; // [num_nodes + 1]
    std::vector<int> node_edges;        // incident edge indices
    std::vector<int> node_edge_signs;   // sign of each incident edge at this node
};

struct ADMMResult {
    std::vector<float> flow;         // fractional optimal flow [num_edges]
    int iterations;                  // iterations until convergence
    float primal_residual;           // ||x - z||
    float dual_residual;             // ρ * ||z_k - z_{k-1}||
    float objective;                 // final cost value
    float time_ms;                   // total GPU time
};

// Solve LP relaxation via ADMM on GPU
ADMMResult admm_solve(
    const ADMMProblem& prob,
    float rho = 1.0f,
    int max_iter = 500,
    float abs_tol = 1e-3f,
    float rel_tol = 1e-3f);

} // namespace cuda
} // namespace qw

#endif // QW_ADMM_SOLVER_H_
