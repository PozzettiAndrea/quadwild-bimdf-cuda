#ifndef QW_SUITOR_SOLVER_H_
#define QW_SUITOR_SOLVER_H_

// ============================================================
// GPU Suitor Algorithm for Weighted Matching
//
// Replaces Blossom matching in BiMDF refinement.
// Based on: "AMG Based on Compatible Weighted Matching for GPUs"
//           (D'Ambra, Durastante, Filippone 2020)
//
// The Suitor algorithm finds a half-approximate maximum weight
// matching in parallel:
//   1. Each vertex proposes to its best unmatched neighbor
//   2. Each vertex accepts its best proposal (mutual best = match)
//   3. Repeat until no more proposals
//
// Fully GPU-parallel, no augmenting paths, no Blossom.
// For BiMDF refinement: we reformulate the matching problem
// as finding the best flow swap at each node.
// ============================================================

#include "bimdf_cuda.h"
#include <vector>

namespace qw {
namespace cuda {

struct SuitorResult {
    std::vector<int> flow;   // improved flow
    double cost;
    int rounds;              // suitor rounds until convergence
    float time_ms;
};

// Run Suitor-based refinement on BiMDF
// Phase1 solution → iterative Suitor matching → improved solution
SuitorResult suitor_refine_bimdf(
    BiMDFFlat& bimdf,
    int max_outer_iters = 50,    // re-linearization rounds
    int max_suitor_rounds = 20,  // suitor rounds per matching solve
    double cost_threshold = -0.01);

} // namespace cuda
} // namespace qw

#endif
