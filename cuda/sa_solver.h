#ifndef QW_SA_SOLVER_H_
#define QW_SA_SOLVER_H_

// GPU parallel simulated annealing for BiMDF
// Runs K independent chains in parallel on GPU.
// Each chain proposes random flow swaps at valid 2-cycles,
// accepts/rejects via Metropolis criterion.

#include "bimdf_cuda.h"
#include <vector>

namespace qw {
namespace cuda {

struct SAResult {
    std::vector<int> best_flow;
    double best_cost;
    int total_steps;
    float time_ms;
};

SAResult sa_solve_bimdf(
    const BiMDFFlat& bimdf,
    int num_chains = 4096,
    int steps_per_chain = 2000,
    float temp_start = 10.0f,
    float temp_end = 0.01f);

} // namespace cuda
} // namespace qw

#endif
