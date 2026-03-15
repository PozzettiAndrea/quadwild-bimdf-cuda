// ============================================================
// GPU BiMDF refinement: parallel negative-cycle canceling
//
// Replaces satsuma's Blossom-matching refinement with GPU
// parallel detection and cancellation of negative-cost cycles
// in the directed MCF residual graph.
//
// Kernels:
//   k_compute_residual_costs  — compute marginal costs from flow
//   k_find_negative_2cycles   — find arc pairs forming neg cycles
//   k_find_negative_3cycles   — find triangles forming neg cycles
//   k_resolve_conflicts       — Luby independent set selection
//   k_apply_augmentations     — update flow atomically
//   k_reduce_improvement      — check if any improvement found
// ============================================================

#include "bimdf_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cfloat>

namespace qw {
namespace cuda {

// ============================================================
// Strategy name mapping
// ============================================================

FlowStrategy flow_strategy_from_name(const char* name) {
    if (!name) return FLOW_SUITOR;
    if (strcmp(name, "satsuma") == 0) return FLOW_SATSUMA;
    if (strcmp(name, "early-term") == 0) return FLOW_EARLY_TERM;
    if (strcmp(name, "admm") == 0) return FLOW_ADMM;
    if (strcmp(name, "phase1-only") == 0) return FLOW_PHASE1_ONLY;
    if (strcmp(name, "pdhg") == 0) return FLOW_PDHG;
    if (strcmp(name, "pdhg-direct") == 0) return FLOW_PDHG_DIRECT;
    if (strcmp(name, "sa") == 0) return FLOW_SA;
    if (strcmp(name, "suitor") == 0) return FLOW_SUITOR;
    if (strcmp(name, "pdhg-v2") == 0) return FLOW_PDHG_V2;
    if (strcmp(name, "hybrid") == 0) return FLOW_HYBRID;
    if (strcmp(name, "sinpen") == 0) return FLOW_SINPEN;
    if (strcmp(name, "pump") == 0) return FLOW_PUMP;
    if (strcmp(name, "suitor-aug") == 0) return FLOW_SUITOR_AUG;
    if (strcmp(name, "adaptive") == 0) return FLOW_ADAPTIVE;
    if (strcmp(name, "directed") == 0) return FLOW_DIRECTED;
    return FLOW_SUITOR;  // default
}

const char* flow_strategy_name(FlowStrategy s) {
    switch (s) {
        case FLOW_SATSUMA: return "satsuma";
        case FLOW_EARLY_TERM: return "early-term";
        case FLOW_ADMM: return "admm";
        case FLOW_PHASE1_ONLY: return "phase1-only";
        case FLOW_PDHG: return "pdhg";
        case FLOW_PDHG_DIRECT: return "pdhg-direct";
        case FLOW_SA: return "sa";
        case FLOW_SUITOR: return "suitor";
        case FLOW_PDHG_V2: return "pdhg-v2";
        case FLOW_HYBRID: return "hybrid";
        case FLOW_SINPEN: return "sinpen";
        case FLOW_PUMP: return "pump";
        case FLOW_SUITOR_AUG: return "suitor-aug";
        case FLOW_ADAPTIVE: return "adaptive";
        case FLOW_DIRECTED: return "directed";
        default: return "unknown";
    }
}

// ============================================================
// Cost evaluation helpers (host + device)
// ============================================================

static inline double eval_edge_cost(int cost_type, double target, double weight, int flow) {
    switch (cost_type) {
        case COST_ABS_DEVIATION:
            return weight * std::abs((double)flow - target);
        case COST_QUAD_DEVIATION: {
            double d = (double)flow - target;
            return weight * d * d;
        }
        case COST_LINEAR:
            return weight * (double)flow;
        default: return 0.0;
    }
}

// Marginal cost of increasing flow by 1 (forward residual cost)
static inline int64_t marginal_cost_fwd(int cost_type, double target, double weight, int flow) {
    double c_next = eval_edge_cost(cost_type, target, weight, flow + 1);
    double c_curr = eval_edge_cost(cost_type, target, weight, flow);
    return (int64_t)llround((c_next - c_curr) * (1LL << 20));
}

// Marginal cost of decreasing flow by 1 (backward residual cost = negative of forward)
static inline int64_t marginal_cost_bwd(int cost_type, double target, double weight, int flow) {
    double c_curr = eval_edge_cost(cost_type, target, weight, flow);
    double c_prev = eval_edge_cost(cost_type, target, weight, flow - 1);
    return (int64_t)llround((c_prev - c_curr) * (1LL << 20));
}

// ============================================================
// Device: cost evaluation
// ============================================================

__device__ int64_t d_marginal_cost_fwd(int cost_type, double target, double weight, int flow) {
    double c_next, c_curr;
    if (cost_type == COST_LINEAR) {
        // Linear: cost(f) = weight * f, marginal = weight
        return (int64_t)llrintf((float)(weight * (double)(1LL << 20)));
    } else if (cost_type == COST_ABS_DEVIATION) {
        c_curr = weight * fabs((double)flow - target);
        c_next = weight * fabs((double)(flow + 1) - target);
    } else if (cost_type == COST_QUAD_DEVIATION) {
        double d0 = (double)flow - target;
        double d1 = (double)(flow + 1) - target;
        c_curr = weight * d0 * d0;
        c_next = weight * d1 * d1;
    } else {
        return 0;
    }
    return (int64_t)llrintf((float)((c_next - c_curr) * (double)(1LL << 20)));
}

__device__ int64_t d_marginal_cost_bwd(int cost_type, double target, double weight, int flow) {
    double c_curr, c_prev;
    if (cost_type == COST_LINEAR) {
        // Linear: marginal of -1 = -weight
        return (int64_t)llrintf((float)(-weight * (double)(1LL << 20)));
    } else if (cost_type == COST_ABS_DEVIATION) {
        c_curr = weight * fabs((double)flow - target);
        c_prev = weight * fabs((double)(flow - 1) - target);
    } else if (cost_type == COST_QUAD_DEVIATION) {
        double d0 = (double)flow - target;
        double dm = (double)(flow - 1) - target;
        c_curr = weight * d0 * d0;
        c_prev = weight * dm * dm;
    } else {
        return 0;
    }
    return (int64_t)llrintf((float)((c_prev - c_curr) * (double)(1LL << 20)));
}

// ============================================================
// Kernel: Compute residual arc costs and capacities from flow
// Each BiMDF edge produces 2 residual arcs (fwd + bwd)
// in the double-cover directed graph.
// ============================================================

__global__ void k_compute_residual(
    // BiMDF edge data
    const int* __restrict__ edge_cost_type,
    const double* __restrict__ edge_cost_target,
    const double* __restrict__ edge_cost_weight,
    const int* __restrict__ edge_lower,
    const int* __restrict__ edge_upper,
    const int* __restrict__ flow,
    // Output: residual arc data (2 arcs per edge: fwd at 2*e, bwd at 2*e+1)
    int64_t* __restrict__ res_cost,
    int* __restrict__ res_cap,
    int num_edges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;

    int f = flow[e];
    int lo = edge_lower[e];
    int hi = edge_upper[e];
    int ct = edge_cost_type[e];
    double tgt = edge_cost_target[e];
    double wgt = edge_cost_weight[e];

    // Forward residual: can increase flow by 1
    int fwd_cap = hi - f;
    int64_t fwd_cost = (fwd_cap > 0) ? d_marginal_cost_fwd(ct, tgt, wgt, f) : 0;

    // Backward residual: can decrease flow by 1
    int bwd_cap = f - lo;
    int64_t bwd_cost = (bwd_cap > 0) ? d_marginal_cost_bwd(ct, tgt, wgt, f) : 0;

    res_cost[2 * e] = fwd_cost;
    res_cap[2 * e] = fwd_cap;
    res_cost[2 * e + 1] = bwd_cost;
    res_cap[2 * e + 1] = bwd_cap;
}

// ============================================================
// Kernel: Find negative-cost 2-cycles in the directed residual
//
// A 2-cycle exists when arc (u→v) and arc (v→u) both have
// positive residual capacity and their combined cost < 0.
//
// In the double-cover MCF: node n splits into n+ and n-.
// Arcs go between {u+,u-} and {v+,v-} based on orientation.
// A 2-cycle is: two BiMDF edges between the same pair of nodes
// with the same orientation, where rerouting 1 unit of flow
// from one to the other reduces cost.
//
// Simplified: for each pair of BiMDF edges (e1, e2) sharing
// both endpoints with same orientation, test if swapping
// flow[e1]+=1, flow[e2]-=1 (or vice versa) reduces total cost.
// ============================================================

__global__ void k_find_negative_2cycles(
    // Node-arc CSR (BiMDF level)
    const int* __restrict__ node_edge_offsets,
    const int* __restrict__ node_edges,
    // Edge data
    const int* __restrict__ edge_u,
    const int* __restrict__ edge_v,
    const int* __restrict__ edge_u_head,
    const int* __restrict__ edge_v_head,
    const int* __restrict__ edge_cost_type,
    const double* __restrict__ edge_cost_target,
    const double* __restrict__ edge_cost_weight,
    const int* __restrict__ edge_lower,
    const int* __restrict__ edge_upper,
    const int* __restrict__ flow,
    // Output: best swap per node
    int* __restrict__ best_e1,       // first edge of swap
    int* __restrict__ best_e2,       // second edge of swap
    int* __restrict__ best_delta_e1, // flow change on e1 (+1 or -1)
    int64_t* __restrict__ best_cost_delta, // cost improvement (negative = good)
    int num_nodes)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= num_nodes) return;

    int start = node_edge_offsets[n];
    int end = node_edge_offsets[n + 1];

    int64_t best_cd = 0;  // only accept negative (improving) deltas
    int be1 = -1, be2 = -1, bd1 = 0;

    for (int i = start; i < end; ++i) {
        int e1 = node_edges[i];
        int f1 = flow[e1];
        int ct1 = edge_cost_type[e1];
        double tgt1 = edge_cost_target[e1];
        double wgt1 = edge_cost_weight[e1];

        // Get sign of e1 at node n
        int sign_e1_at_n;
        int other_node_e1;
        if (edge_u[e1] == n) {
            sign_e1_at_n = edge_u_head[e1] ? 1 : -1;
            other_node_e1 = edge_v[e1];
        } else {
            sign_e1_at_n = edge_v_head[e1] ? 1 : -1;
            other_node_e1 = edge_u[e1];
        }

        for (int j = i + 1; j < end; ++j) {
            int e2 = node_edges[j];
            int f2 = flow[e2];
            int ct2 = edge_cost_type[e2];
            double tgt2 = edge_cost_target[e2];
            double wgt2 = edge_cost_weight[e2];

            // Get sign of e2 at node n
            int sign_e2_at_n;
            int other_node_e2;
            if (edge_u[e2] == n) {
                sign_e2_at_n = edge_u_head[e2] ? 1 : -1;
                other_node_e2 = edge_v[e2];
            } else {
                sign_e2_at_n = edge_v_head[e2] ? 1 : -1;
                other_node_e2 = edge_u[e2];
            }

            // For a valid 2-edge swap at node n:
            // We need sign_e1_at_n = sign_e2_at_n (same contribution direction)
            // AND the other endpoints must match with compatible signs
            if (sign_e1_at_n != sign_e2_at_n) continue;
            if (other_node_e1 != other_node_e2) continue;

            // Check sign at the other endpoint
            int sign_e1_at_other = (edge_u[e1] == other_node_e1)
                ? (edge_u_head[e1] ? 1 : -1)
                : (edge_v_head[e1] ? 1 : -1);
            int sign_e2_at_other = (edge_u[e2] == other_node_e2)
                ? (edge_u_head[e2] ? 1 : -1)
                : (edge_v_head[e2] ? 1 : -1);

            if (sign_e1_at_other != sign_e2_at_other) continue;

            // Valid swap! Try flow[e1]+=1, flow[e2]-=1
            if (f1 + 1 <= edge_upper[e1] && f2 - 1 >= edge_lower[e2]) {
                int64_t delta = d_marginal_cost_fwd(ct1, tgt1, wgt1, f1)
                              + d_marginal_cost_bwd(ct2, tgt2, wgt2, f2);
                if (delta < best_cd) {
                    best_cd = delta; be1 = e1; be2 = e2; bd1 = 1;
                }
            }

            // Try flow[e1]-=1, flow[e2]+=1
            if (f1 - 1 >= edge_lower[e1] && f2 + 1 <= edge_upper[e2]) {
                int64_t delta = d_marginal_cost_bwd(ct1, tgt1, wgt1, f1)
                              + d_marginal_cost_fwd(ct2, tgt2, wgt2, f2);
                if (delta < best_cd) {
                    best_cd = delta; be1 = e1; be2 = e2; bd1 = -1;
                }
            }
        }
    }

    best_e1[n] = be1;
    best_e2[n] = be2;
    best_delta_e1[n] = bd1;
    best_cost_delta[n] = best_cd;
}

// ============================================================
// Kernel: Find negative-cost 3-cycles (triangles)
//
// For each node n, for each pair of incident edges (e1, e2),
// check if there's a third edge e3 connecting the other
// endpoints of e1 and e2, forming a valid augmenting triangle.
// ============================================================

__global__ void k_find_negative_3cycles(
    const int* __restrict__ node_edge_offsets,
    const int* __restrict__ node_edges,
    const int* __restrict__ edge_u,
    const int* __restrict__ edge_v,
    const int* __restrict__ edge_u_head,
    const int* __restrict__ edge_v_head,
    const int* __restrict__ edge_cost_type,
    const double* __restrict__ edge_cost_target,
    const double* __restrict__ edge_cost_weight,
    const int* __restrict__ edge_lower,
    const int* __restrict__ edge_upper,
    const int* __restrict__ flow,
    // Edge lookup: for fast edge(u,v) queries
    // edge_between[u * num_nodes + v] = edge index or -1
    // Only feasible for small graphs (< 16K nodes → 256M entries, too big)
    // Instead use sorted edge list per node
    const int* __restrict__ node_neighbor_offsets,  // CSR: neighbors of each node
    const int* __restrict__ node_neighbors,          // neighbor node indices
    const int* __restrict__ node_neighbor_edges,     // corresponding edge indices
    // Output: best 3-cycle per node (stored alongside 2-cycle results)
    int* __restrict__ best_e1,
    int* __restrict__ best_e2,
    int* __restrict__ best_e3,         // third edge (-1 if 2-cycle)
    int* __restrict__ best_delta_e1,
    int* __restrict__ best_delta_e2,
    int64_t* __restrict__ best_cost_delta,
    int num_nodes)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= num_nodes) return;

    int start_n = node_edge_offsets[n];
    int end_n = node_edge_offsets[n + 1];

    int64_t best_cd = best_cost_delta[n];  // start from best 2-cycle result
    int be1 = best_e1[n], be2 = best_e2[n], be3 = -1;
    int bd1 = best_delta_e1[n], bd2 = 0;

    // For each pair of edges (e1 to node m1, e2 to node m2):
    // look for edge e3 between m1 and m2
    for (int i = start_n; i < end_n; ++i) {
        int e1 = node_edges[i];
        int m1 = (edge_u[e1] == n) ? edge_v[e1] : edge_u[e1];
        int f1 = flow[e1];

        // Sign of e1 at n
        int s1n = (edge_u[e1] == n) ? (edge_u_head[e1] ? 1 : -1) : (edge_v_head[e1] ? 1 : -1);

        for (int j = i + 1; j < end_n; ++j) {
            int e2 = node_edges[j];
            int m2 = (edge_u[e2] == n) ? edge_v[e2] : edge_u[e2];
            if (m1 == m2) continue;  // would be 2-cycle, already handled
            int f2 = flow[e2];

            // Sign of e2 at n
            int s2n = (edge_u[e2] == n) ? (edge_u_head[e2] ? 1 : -1) : (edge_v_head[e2] ? 1 : -1);

            // Search for edge e3 between m1 and m2
            // Binary search in m1's neighbor list for m2
            int start_m1 = node_neighbor_offsets[m1];
            int end_m1 = node_neighbor_offsets[m1 + 1];

            for (int k = start_m1; k < end_m1; ++k) {
                if (node_neighbors[k] != m2) continue;

                int e3 = node_neighbor_edges[k];
                int f3 = flow[e3];

                // Try all 8 combinations of ±1 on (e1, e2, e3)
                // that maintain flow conservation at n, m1, m2
                // For a triangle n-m1-m2-n with edges e1(n,m1), e3(m1,m2), e2(m2,n):
                // Flow conservation: at each node, sum of sign*delta = 0

                // Signs at m1 for e1 and e3
                int s1m1 = (edge_u[e1] == m1) ? (edge_u_head[e1] ? 1 : -1) : (edge_v_head[e1] ? 1 : -1);
                int s3m1 = (edge_u[e3] == m1) ? (edge_u_head[e3] ? 1 : -1) : (edge_v_head[e3] ? 1 : -1);
                // Signs at m2 for e2 and e3
                int s2m2 = (edge_u[e2] == m2) ? (edge_u_head[e2] ? 1 : -1) : (edge_v_head[e2] ? 1 : -1);
                int s3m2 = (edge_u[e3] == m2) ? (edge_u_head[e3] ? 1 : -1) : (edge_v_head[e3] ? 1 : -1);

                // Try d1=+1 on e1: at n, contribution = s1n*1, at m1 = s1m1*1
                // Need d3 such that at m1: s1m1*1 + s3m1*d3 = 0 → d3 = -s1m1/s3m1
                // Need d2 such that at n: s1n*1 + s2n*d2 = 0 → d2 = -s1n/s2n
                // Need at m2: s2m2*d2 + s3m2*d3 = 0 → check

                if (s3m1 != 0 && s2n != 0) {
                    int d1 = 1;
                    int d3 = -(s1m1 * d1) / s3m1;  // = ±1
                    int d2 = -(s1n * d1) / s2n;      // = ±1

                    if ((d3 == 1 || d3 == -1) && (d2 == 1 || d2 == -1)) {
                        // Check m2 conservation
                        if (s2m2 * d2 + s3m2 * d3 == 0) {
                            // Check bounds
                            if (f1 + d1 >= edge_lower[e1] && f1 + d1 <= edge_upper[e1] &&
                                f2 + d2 >= edge_lower[e2] && f2 + d2 <= edge_upper[e2] &&
                                f3 + d3 >= edge_lower[e3] && f3 + d3 <= edge_upper[e3]) {

                                int ct1 = edge_cost_type[e1], ct2 = edge_cost_type[e2], ct3 = edge_cost_type[e3];
                                double t1 = edge_cost_target[e1], t2 = edge_cost_target[e2], t3 = edge_cost_target[e3];
                                double w1 = edge_cost_weight[e1], w2 = edge_cost_weight[e2], w3 = edge_cost_weight[e3];

                                int64_t delta =
                                    (d1 > 0 ? d_marginal_cost_fwd(ct1,t1,w1,f1) : d_marginal_cost_bwd(ct1,t1,w1,f1)) +
                                    (d2 > 0 ? d_marginal_cost_fwd(ct2,t2,w2,f2) : d_marginal_cost_bwd(ct2,t2,w2,f2)) +
                                    (d3 > 0 ? d_marginal_cost_fwd(ct3,t3,w3,f3) : d_marginal_cost_bwd(ct3,t3,w3,f3));

                                if (delta < best_cd) {
                                    best_cd = delta;
                                    be1 = e1; be2 = e2; be3 = e3;
                                    bd1 = d1; bd2 = d2;
                                    // d3 implied by conservation
                                }
                            }
                        }
                    }
                }

                // Also try d1 = -1
                if (s3m1 != 0 && s2n != 0) {
                    int d1 = -1;
                    int d3 = -(s1m1 * d1) / s3m1;
                    int d2 = -(s1n * d1) / s2n;

                    if ((d3 == 1 || d3 == -1) && (d2 == 1 || d2 == -1)) {
                        if (s2m2 * d2 + s3m2 * d3 == 0) {
                            if (f1 + d1 >= edge_lower[e1] && f1 + d1 <= edge_upper[e1] &&
                                f2 + d2 >= edge_lower[e2] && f2 + d2 <= edge_upper[e2] &&
                                f3 + d3 >= edge_lower[e3] && f3 + d3 <= edge_upper[e3]) {

                                int ct1 = edge_cost_type[e1], ct2 = edge_cost_type[e2], ct3 = edge_cost_type[e3];
                                double t1 = edge_cost_target[e1], t2 = edge_cost_target[e2], t3 = edge_cost_target[e3];
                                double w1 = edge_cost_weight[e1], w2 = edge_cost_weight[e2], w3 = edge_cost_weight[e3];

                                int64_t delta =
                                    (d1 > 0 ? d_marginal_cost_fwd(ct1,t1,w1,f1) : d_marginal_cost_bwd(ct1,t1,w1,f1)) +
                                    (d2 > 0 ? d_marginal_cost_fwd(ct2,t2,w2,f2) : d_marginal_cost_bwd(ct2,t2,w2,f2)) +
                                    (d3 > 0 ? d_marginal_cost_fwd(ct3,t3,w3,f3) : d_marginal_cost_bwd(ct3,t3,w3,f3));

                                if (delta < best_cd) {
                                    best_cd = delta;
                                    be1 = e1; be2 = e2; be3 = e3;
                                    bd1 = d1; bd2 = d2;
                                }
                            }
                        }
                    }
                }
                break;  // found the edge between m1 and m2
            }
        }
    }

    best_e1[n] = be1;
    best_e2[n] = be2;
    best_e3[n] = be3;
    best_delta_e1[n] = bd1;
    best_delta_e2[n] = bd2;
    best_cost_delta[n] = best_cd;
}

// ============================================================
// Kernel: Conflict resolution via Luby-like independent set
//
// A node's swap conflicts with another node's swap if they
// share any edge. Select the better improvement when conflicting.
// ============================================================

__global__ void k_resolve_conflicts(
    const int* __restrict__ best_e1,
    const int* __restrict__ best_e2,
    const int* __restrict__ best_e3,
    const int64_t* __restrict__ best_cost_delta,
    const int* __restrict__ edge_u,
    const int* __restrict__ edge_v,
    // Per-edge: which node claims this edge (atomically set)
    int* __restrict__ edge_owner,  // initialized to -1
    int* __restrict__ selected,    // output: 1 if this node's swap is selected
    int num_nodes)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= num_nodes) return;

    selected[n] = 0;
    if (best_cost_delta[n] >= 0) return;  // no improvement

    int e1 = best_e1[n], e2 = best_e2[n], e3 = best_e3[n];
    if (e1 < 0) return;

    // Try to claim all edges atomically
    // Use atomicCAS: only succeed if edge is unclaimed (-1)
    int claimed1 = atomicCAS(&edge_owner[e1], -1, n);
    if (claimed1 != -1 && claimed1 != n) return;  // someone else claimed e1

    int claimed2 = atomicCAS(&edge_owner[e2], -1, n);
    if (claimed2 != -1 && claimed2 != n) {
        // Release e1
        if (claimed1 == -1) atomicCAS(&edge_owner[e1], n, -1);
        return;
    }

    if (e3 >= 0) {
        int claimed3 = atomicCAS(&edge_owner[e3], -1, n);
        if (claimed3 != -1 && claimed3 != n) {
            // Release e1, e2
            if (claimed1 == -1) atomicCAS(&edge_owner[e1], n, -1);
            if (claimed2 == -1) atomicCAS(&edge_owner[e2], n, -1);
            return;
        }
    }

    selected[n] = 1;
}

// ============================================================
// Kernel: Apply selected augmentations
// ============================================================

__global__ void k_apply_augmentations(
    const int* __restrict__ best_e1,
    const int* __restrict__ best_e2,
    const int* __restrict__ best_e3,
    const int* __restrict__ best_delta_e1,
    const int* __restrict__ best_delta_e2,
    const int* __restrict__ selected,
    const int* __restrict__ edge_u,
    const int* __restrict__ edge_v,
    const int* __restrict__ edge_u_head,
    const int* __restrict__ edge_v_head,
    int* __restrict__ flow,
    int* __restrict__ num_applied,  // atomic counter
    int num_nodes)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= num_nodes) return;
    if (!selected[n]) return;

    int e1 = best_e1[n], e2 = best_e2[n], e3 = best_e3[n];
    int d1 = best_delta_e1[n], d2 = best_delta_e2[n];

    atomicAdd(&flow[e1], d1);
    atomicAdd(&flow[e2], d2);

    if (e3 >= 0) {
        // Compute d3 from conservation at the shared node
        // At the node connecting e1 and e3 (not n):
        int m1 = (edge_u[e1] == n) ? edge_v[e1] : edge_u[e1];
        int s1m1 = (edge_u[e1] == m1) ? (edge_u_head[e1] ? 1 : -1) : (edge_v_head[e1] ? 1 : -1);
        int s3m1 = (edge_u[e3] == m1) ? (edge_u_head[e3] ? 1 : -1) : (edge_v_head[e3] ? 1 : -1);
        int d3 = -(s1m1 * d1) / s3m1;
        atomicAdd(&flow[e3], d3);
    }

    atomicAdd(num_applied, 1);
}

// ============================================================
// Host: Build node-edge adjacency (CSR) from BiMDF
// ============================================================

static void build_node_edge_csr(
    const BiMDFFlat& bimdf,
    std::vector<int>& offsets,
    std::vector<int>& edges)
{
    int N = bimdf.num_nodes;
    std::vector<std::vector<int>> adj(N);
    for (int e = 0; e < bimdf.num_edges; ++e) {
        adj[bimdf.edge_u[e]].push_back(e);
        adj[bimdf.edge_v[e]].push_back(e);
    }
    offsets.resize(N + 1);
    edges.clear();
    offsets[0] = 0;
    for (int n = 0; n < N; ++n) {
        edges.insert(edges.end(), adj[n].begin(), adj[n].end());
        offsets[n + 1] = (int)edges.size();
    }
}

// Build node-neighbor CSR (node → neighbor nodes + corresponding edges)
static void build_node_neighbor_csr(
    const BiMDFFlat& bimdf,
    std::vector<int>& offsets,
    std::vector<int>& neighbors,
    std::vector<int>& neighbor_edges)
{
    int N = bimdf.num_nodes;
    std::vector<std::vector<std::pair<int,int>>> adj(N);  // (neighbor, edge)
    for (int e = 0; e < bimdf.num_edges; ++e) {
        int u = bimdf.edge_u[e], v = bimdf.edge_v[e];
        adj[u].push_back({v, e});
        adj[v].push_back({u, e});
    }
    offsets.resize(N + 1);
    neighbors.clear();
    neighbor_edges.clear();
    offsets[0] = 0;
    for (int n = 0; n < N; ++n) {
        // Sort by neighbor index for binary search
        std::sort(adj[n].begin(), adj[n].end());
        for (auto& [nb, edge] : adj[n]) {
            neighbors.push_back(nb);
            neighbor_edges.push_back(edge);
        }
        offsets[n + 1] = (int)neighbors.size();
    }
}

// ============================================================
// Host: Main GPU refinement driver
// ============================================================

BiMDFCudaResult refine_bimdf_cuda(
    BiMDFFlat& bimdf,
    int max_iterations,
    int max_cycle_length)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    int N = bimdf.num_nodes;
    int E = bimdf.num_edges;

    printf("[BIMDF-CUDA] Graph: %d nodes, %d edges\n", N, E);
    printf("[BIMDF-CUDA] Initial cost: %.6f\n", eval_bimdf_cost(bimdf));

    // Build CSR adjacency on CPU
    std::vector<int> h_node_edge_off, h_node_edges;
    build_node_edge_csr(bimdf, h_node_edge_off, h_node_edges);

    std::vector<int> h_nb_off, h_nb_nodes, h_nb_edges;
    if (max_cycle_length >= 3) {
        build_node_neighbor_csr(bimdf, h_nb_off, h_nb_nodes, h_nb_edges);
    }

    // Allocate GPU memory
    int *d_edge_u, *d_edge_v, *d_edge_u_head, *d_edge_v_head;
    int *d_edge_lower, *d_edge_upper, *d_edge_cost_type;
    double *d_edge_cost_target, *d_edge_cost_weight;
    int *d_flow;
    int *d_node_edge_off, *d_node_edges;
    int *d_best_e1, *d_best_e2, *d_best_e3, *d_best_delta_e1, *d_best_delta_e2;
    int64_t *d_best_cost_delta;
    int *d_edge_owner, *d_selected, *d_num_applied;

    cudaMalloc(&d_edge_u, E * sizeof(int));
    cudaMalloc(&d_edge_v, E * sizeof(int));
    cudaMalloc(&d_edge_u_head, E * sizeof(int));
    cudaMalloc(&d_edge_v_head, E * sizeof(int));
    cudaMalloc(&d_edge_lower, E * sizeof(int));
    cudaMalloc(&d_edge_upper, E * sizeof(int));
    cudaMalloc(&d_edge_cost_type, E * sizeof(int));
    cudaMalloc(&d_edge_cost_target, E * sizeof(double));
    cudaMalloc(&d_edge_cost_weight, E * sizeof(double));
    cudaMalloc(&d_flow, E * sizeof(int));
    cudaMalloc(&d_node_edge_off, (N + 1) * sizeof(int));
    cudaMalloc(&d_node_edges, h_node_edges.size() * sizeof(int));
    cudaMalloc(&d_best_e1, N * sizeof(int));
    cudaMalloc(&d_best_e2, N * sizeof(int));
    cudaMalloc(&d_best_e3, N * sizeof(int));
    cudaMalloc(&d_best_delta_e1, N * sizeof(int));
    cudaMalloc(&d_best_delta_e2, N * sizeof(int));
    cudaMalloc(&d_best_cost_delta, N * sizeof(int64_t));
    cudaMalloc(&d_edge_owner, E * sizeof(int));
    cudaMalloc(&d_selected, N * sizeof(int));
    cudaMalloc(&d_num_applied, sizeof(int));

    // 3-cycle neighbor data
    int *d_nb_off = nullptr, *d_nb_nodes = nullptr, *d_nb_edges = nullptr;
    if (max_cycle_length >= 3) {
        cudaMalloc(&d_nb_off, (N + 1) * sizeof(int));
        cudaMalloc(&d_nb_nodes, h_nb_nodes.size() * sizeof(int));
        cudaMalloc(&d_nb_edges, h_nb_edges.size() * sizeof(int));
        cudaMemcpy(d_nb_off, h_nb_off.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nb_nodes, h_nb_nodes.data(), h_nb_nodes.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nb_edges, h_nb_edges.data(), h_nb_edges.size() * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Upload
    auto t_upload = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_edge_u, bimdf.edge_u.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_v, bimdf.edge_v.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_u_head, bimdf.edge_u_head.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_v_head, bimdf.edge_v_head.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_lower, bimdf.edge_lower.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_upper, bimdf.edge_upper.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_cost_type, bimdf.edge_cost_type.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_cost_target, bimdf.edge_cost_target.data(), E * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_cost_weight, bimdf.edge_cost_weight.data(), E * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flow, bimdf.flow.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_edge_off, h_node_edge_off.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_edges, h_node_edges.data(), h_node_edges.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto t_kernels = std::chrono::high_resolution_clock::now();

    // Main cycle-canceling loop
    int block = 256;
    int grid_n = (N + block - 1) / block;
    int total_canceled = 0;
    int iter;

    for (iter = 0; iter < max_iterations; ++iter) {
        // Reset per-iteration state
        cudaMemset(d_best_e1, 0xff, N * sizeof(int));  // -1
        cudaMemset(d_best_e2, 0xff, N * sizeof(int));
        cudaMemset(d_best_e3, 0xff, N * sizeof(int));
        cudaMemset(d_best_delta_e1, 0, N * sizeof(int));
        cudaMemset(d_best_delta_e2, 0, N * sizeof(int));
        cudaMemset(d_best_cost_delta, 0, N * sizeof(int64_t));
        cudaMemset(d_edge_owner, 0xff, E * sizeof(int));  // -1
        cudaMemset(d_selected, 0, N * sizeof(int));
        cudaMemset(d_num_applied, 0, sizeof(int));

        // Step 1: Find negative 2-cycles (parallel over all nodes)
        k_find_negative_2cycles<<<grid_n, block>>>(
            d_node_edge_off, d_node_edges,
            d_edge_u, d_edge_v, d_edge_u_head, d_edge_v_head,
            d_edge_cost_type, d_edge_cost_target, d_edge_cost_weight,
            d_edge_lower, d_edge_upper, d_flow,
            d_best_e1, d_best_e2, d_best_delta_e1, d_best_cost_delta,
            N);

        // Step 2: Find negative 3-cycles (improves on 2-cycle results)
        if (max_cycle_length >= 3 && d_nb_off) {
            k_find_negative_3cycles<<<grid_n, block>>>(
                d_node_edge_off, d_node_edges,
                d_edge_u, d_edge_v, d_edge_u_head, d_edge_v_head,
                d_edge_cost_type, d_edge_cost_target, d_edge_cost_weight,
                d_edge_lower, d_edge_upper, d_flow,
                d_nb_off, d_nb_nodes, d_nb_edges,
                d_best_e1, d_best_e2, d_best_e3,
                d_best_delta_e1, d_best_delta_e2, d_best_cost_delta,
                N);
        }

        // Step 3: Resolve conflicts (Luby-like independent set)
        k_resolve_conflicts<<<grid_n, block>>>(
            d_best_e1, d_best_e2, d_best_e3, d_best_cost_delta,
            d_edge_u, d_edge_v,
            d_edge_owner, d_selected,
            N);

        // Step 4: Apply selected augmentations
        k_apply_augmentations<<<grid_n, block>>>(
            d_best_e1, d_best_e2, d_best_e3,
            d_best_delta_e1, d_best_delta_e2,
            d_selected,
            d_edge_u, d_edge_v, d_edge_u_head, d_edge_v_head,
            d_flow, d_num_applied,
            N);

        // Check convergence
        int h_num_applied = 0;
        cudaMemcpy(&h_num_applied, d_num_applied, sizeof(int), cudaMemcpyDeviceToHost);

        total_canceled += h_num_applied;

        if (h_num_applied == 0) {
            break;  // No more negative cycles found
        }
    }

    cudaDeviceSynchronize();
    auto t_download = std::chrono::high_resolution_clock::now();

    // Download improved flow
    cudaMemcpy(bimdf.flow.data(), d_flow, E * sizeof(int), cudaMemcpyDeviceToHost);

    auto t_end = std::chrono::high_resolution_clock::now();

    // Cleanup
    cudaFree(d_edge_u); cudaFree(d_edge_v);
    cudaFree(d_edge_u_head); cudaFree(d_edge_v_head);
    cudaFree(d_edge_lower); cudaFree(d_edge_upper);
    cudaFree(d_edge_cost_type); cudaFree(d_edge_cost_target); cudaFree(d_edge_cost_weight);
    cudaFree(d_flow);
    cudaFree(d_node_edge_off); cudaFree(d_node_edges);
    cudaFree(d_best_e1); cudaFree(d_best_e2); cudaFree(d_best_e3);
    cudaFree(d_best_delta_e1); cudaFree(d_best_delta_e2);
    cudaFree(d_best_cost_delta);
    cudaFree(d_edge_owner); cudaFree(d_selected); cudaFree(d_num_applied);
    if (d_nb_off) cudaFree(d_nb_off);
    if (d_nb_nodes) cudaFree(d_nb_nodes);
    if (d_nb_edges) cudaFree(d_nb_edges);

    double final_cost = eval_bimdf_cost(bimdf);
    double upload_ms = std::chrono::duration<double, std::milli>(t_kernels - t_upload).count();
    double kernel_ms = std::chrono::duration<double, std::milli>(t_download - t_kernels).count();
    double download_ms = std::chrono::duration<double, std::milli>(t_end - t_download).count();

    printf("[BIMDF-CUDA] Converged in %d iterations, %d cycles canceled\n", iter, total_canceled);
    printf("[BIMDF-CUDA] Final cost: %.6f\n", final_cost);
    printf("[BIMDF-CUDA] Timing: upload=%.1fms kernel=%.1fms download=%.1fms total=%.1fms\n",
           upload_ms, kernel_ms, download_ms, upload_ms + kernel_ms + download_ms);

    return {
        .flow = bimdf.flow,
        .cost = final_cost,
        .iterations = iter,
        .total_cycles_canceled = total_canceled,
        .time_upload_ms = upload_ms,
        .time_kernel_ms = kernel_ms,
        .time_download_ms = download_ms,
    };
}

// ============================================================
// Host: Evaluate total BiMDF cost
// ============================================================

double eval_bimdf_cost(const BiMDFFlat& bimdf) {
    double total = 0;
    for (int e = 0; e < bimdf.num_edges; ++e) {
        total += eval_edge_cost(bimdf.edge_cost_type[e],
                                bimdf.edge_cost_target[e],
                                bimdf.edge_cost_weight[e],
                                bimdf.flow[e]);
    }
    return total;
}

// ============================================================
// Host: Check feasibility
// ============================================================

bool check_bimdf_feasibility(const BiMDFFlat& bimdf) {
    // Check bounds
    for (int e = 0; e < bimdf.num_edges; ++e) {
        if (bimdf.flow[e] < bimdf.edge_lower[e] || bimdf.flow[e] > bimdf.edge_upper[e]) {
            printf("[BIMDF-CUDA] Bound violation: edge %d flow=%d bounds=[%d,%d]\n",
                   e, bimdf.flow[e], bimdf.edge_lower[e], bimdf.edge_upper[e]);
            return false;
        }
    }

    // Check flow conservation
    std::vector<int> balance(bimdf.num_nodes, 0);
    for (int e = 0; e < bimdf.num_edges; ++e) {
        int f = bimdf.flow[e];
        int u = bimdf.edge_u[e], v = bimdf.edge_v[e];
        balance[u] += bimdf.edge_u_head[e] ? f : -f;
        balance[v] += bimdf.edge_v_head[e] ? f : -f;
    }
    for (int n = 0; n < bimdf.num_nodes; ++n) {
        if (balance[n] != bimdf.node_demand[n]) {
            printf("[BIMDF-CUDA] Conservation violation: node %d balance=%d demand=%d\n",
                   n, balance[n], bimdf.node_demand[n]);
            return false;
        }
    }
    return true;
}

} // namespace cuda
} // namespace qw
