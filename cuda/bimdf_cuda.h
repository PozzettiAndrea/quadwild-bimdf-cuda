#ifndef QW_BIMDF_CUDA_H_
#define QW_BIMDF_CUDA_H_

// ============================================================
// GPU-accelerated BiMDF refinement via parallel cycle canceling
//
// Replaces satsuma's Blossom-matching refinement (Phase 2) with
// GPU parallel negative-cycle detection and cancellation on the
// directed MCF residual graph.
//
// Algorithm:
//   1. CPU: Build directed MCF from BiMDF via double cover
//      (same reduction as satsuma, but with full piecewise-linear
//       marginal costs, no arc explosion needed)
//   2. GPU: Upload MCF residual graph (CSR, ~5MB)
//   3. GPU: Repeat until convergence:
//      a. k_find_negative_2cycles: test all arc pairs (u→v, v→u)
//      b. k_find_negative_3cycles: test all triangles (u→v→w→u)
//      c. k_resolve_conflicts: Luby independent set selection
//      d. k_apply_augmentations: update flow + residual capacities
//      e. k_check_convergence: any negative cycles found?
//   4. CPU: Translate MCF flow back to BiMDF solution
//
// The key insight: for convex cost functions, the MCF residual
// graph marginal costs can be computed on-the-fly from current
// flow values, so we work directly on the BiMDF without the
// lossy arc-explosion approximation.
// ============================================================

#include <vector>
#include <cstdint>
#include <functional>

namespace qw {
namespace cuda {

// ============================================================
// Flow solver strategy selection
// ============================================================

enum FlowStrategy {
    FLOW_SATSUMA = 0,       // Original satsuma (CPU, full quality, slowest)
    FLOW_EARLY_TERM = 1,    // Phase1 + early-term matching (fast, ~2% quality loss)
    FLOW_ADMM = 2,          // GPU ADMM LP warm-start + matching cleanup
    FLOW_PHASE1_ONLY = 3,   // Phase1 MCF only, no refinement (fastest, 10% quality loss)
    FLOW_PDHG = 4,          // GPU PDHG LP on BiMCF per refinement iter (cuSPARSE SpMV)
    FLOW_PDHG_DIRECT = 5,   // GPU PDHG on full BiMDF LP + round + refine
    FLOW_SA = 6,            // GPU parallel simulated annealing
    FLOW_SUITOR = 7,        // GPU Suitor matching (replace Blossom, fully parallel)
    FLOW_PDHG_V2 = 8,       // cuPDLP+-style PDHG + PID control + batched rounding
    FLOW_HYBRID = 9,         // Directed MCF PDHG (TU→integer) per refinement iter
    FLOW_SINPEN = 10,        // PDHG + sin² penalty annealing (integers without rounding)
    FLOW_PUMP = 11,          // GPU feasibility pump (LP solve + round + re-solve)
    FLOW_SUITOR_AUG = 12,    // Suitor + augmenting improvement (2/3 approx)
    FLOW_ADAPTIVE = 13,      // Adaptive deviation schedule (maxdev=5 first, then 2)
    FLOW_DIRECTED = 14,      // Directed LP formulation (TU, no bidirected edges)
};

FlowStrategy flow_strategy_from_name(const char* name);
const char* flow_strategy_name(FlowStrategy s);

// ============================================================
// GPU BiMDF problem representation (flat arrays for CUDA)
// ============================================================

// Cost function types (matches satsuma CostFunction variants)
enum CostType : int32_t {
    COST_ZERO = 0,
    COST_ABS_DEVIATION = 1,
    COST_QUAD_DEVIATION = 2,
    COST_LINEAR = 3,          // cost = weight * flow (BiMCF per-unit cost)
};

// Flat BiMDF graph for GPU processing
struct BiMDFFlat {
    int num_nodes = 0;
    int num_edges = 0;

    // Edge endpoints
    std::vector<int> edge_u;        // [num_edges]
    std::vector<int> edge_v;        // [num_edges]

    // Bidirected orientation
    std::vector<int> edge_u_head;   // [num_edges] 1=head at u, 0=tail
    std::vector<int> edge_v_head;   // [num_edges] 1=head at v, 0=tail

    // Flow bounds
    std::vector<int> edge_lower;    // [num_edges]
    std::vector<int> edge_upper;    // [num_edges]

    // Cost function per edge
    std::vector<int> edge_cost_type;    // [num_edges] CostType enum
    std::vector<double> edge_cost_target; // [num_edges]
    std::vector<double> edge_cost_weight; // [num_edges]

    // Node demands
    std::vector<int> node_demand;   // [num_nodes]

    // Current flow (input: initial solution from Phase 1)
    std::vector<int> flow;          // [num_edges]
};

// Directed MCF graph (after double cover) for GPU residual graph
struct MCFDirected {
    int num_nodes = 0;  // 2 * bimdf.num_nodes
    int num_arcs = 0;   // 2 * bimdf.num_edges (forward + backward per edge)

    // CSR adjacency: node → outgoing arcs
    std::vector<int> out_offsets;   // [num_nodes + 1]
    std::vector<int> out_arcs;      // [total_out_degree]

    // CSR adjacency: node → incoming arcs
    std::vector<int> in_offsets;    // [num_nodes + 1]
    std::vector<int> in_arcs;       // [total_in_degree]

    // Arc data
    std::vector<int> arc_src;       // [num_arcs]
    std::vector<int> arc_dst;       // [num_arcs]
    std::vector<int> arc_capacity;  // [num_arcs] residual capacity
    std::vector<int64_t> arc_cost;  // [num_arcs] marginal cost (scaled by 2^20)

    // Mapping back to BiMDF
    std::vector<int> arc_bimdf_edge;  // [num_arcs] original BiMDF edge index
    std::vector<int> arc_is_forward;  // [num_arcs] 1=forward, 0=backward (in BiMDF sense)

    // Reverse arc index: for arc i, reverse_arc[i] gives the arc in opposite direction
    // (if it exists), or -1
    std::vector<int> reverse_arc;   // [num_arcs]
};

// Result of GPU refinement
struct BiMDFCudaResult {
    std::vector<int> flow;      // improved flow per BiMDF edge
    double cost;                // total cost
    int iterations;             // number of cycle-canceling rounds
    int total_cycles_canceled;  // total negative cycles found and canceled
    double time_upload_ms;
    double time_kernel_ms;
    double time_download_ms;
};

// ============================================================
// Public API
// ============================================================

// Build directed MCF from BiMDF + initial flow (CPU)
// This performs the double-cover construction with marginal costs
MCFDirected build_mcf_residual(const BiMDFFlat& bimdf);

// Run GPU cycle-canceling refinement on the MCF residual graph
// Returns improved BiMDF flow
BiMDFCudaResult refine_bimdf_cuda(
    BiMDFFlat& bimdf,
    int max_iterations = 500,
    int max_cycle_length = 4);  // 2, 3, or 4

// Evaluate total BiMDF cost for a given flow
double eval_bimdf_cost(const BiMDFFlat& bimdf);

// Check if flow satisfies all constraints
bool check_bimdf_feasibility(const BiMDFFlat& bimdf);

} // namespace cuda
} // namespace qw

#endif // QW_BIMDF_CUDA_H_
