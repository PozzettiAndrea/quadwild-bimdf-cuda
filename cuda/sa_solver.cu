// GPU parallel simulated annealing for BiMDF
// K chains run in parallel. Each chain:
//   1. Pick random node n
//   2. Pick two incident edges e1, e2 with compatible orientation
//   3. Propose: flow[e1]+=1, flow[e2]-=1 (or vice versa)
//   4. Accept if Δcost < 0, or with prob exp(-Δcost/T)
//   5. Cool temperature

#include "sa_solver.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace qw {
namespace cuda {

__device__ float d_sa_eval_cost(int ct, float tgt, float wgt, int f) {
    if (ct == COST_ABS_DEVIATION) return wgt * fabsf((float)f - tgt);
    if (ct == COST_QUAD_DEVIATION) { float d = (float)f - tgt; return wgt * d * d; }
    if (ct == COST_LINEAR) return wgt * (float)f;
    return 0.0f;
}

// Each thread = one SA chain
__global__ void k_sa_run(
    // Graph (read-only)
    const int* __restrict__ node_edge_off,
    const int* __restrict__ node_edges,
    const int* __restrict__ edge_u,
    const int* __restrict__ edge_v,
    const int* __restrict__ edge_u_head,
    const int* __restrict__ edge_v_head,
    const int* __restrict__ edge_cost_type,
    const float* __restrict__ edge_cost_target,
    const float* __restrict__ edge_cost_weight,
    const int* __restrict__ edge_lower,
    const int* __restrict__ edge_upper,
    // Per-chain state
    int* __restrict__ chain_flow,       // [num_chains * num_edges]
    float* __restrict__ chain_cost,     // [num_chains]
    // Parameters
    int num_nodes, int num_edges,
    int steps, float temp_start, float temp_end,
    unsigned long long seed)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    // num_chains is grid*block

    // Init RNG
    curandState rng;
    curand_init(seed, c, 0, &rng);

    int* flow = chain_flow + (long long)c * num_edges;
    float cost = chain_cost[c];

    float log_ratio = logf(temp_end / temp_start);

    for (int step = 0; step < steps; ++step) {
        float T = temp_start * expf(log_ratio * (float)step / (float)steps);

        // Pick random node
        int n = curand(&rng) % num_nodes;
        int start = node_edge_off[n];
        int end = node_edge_off[n + 1];
        int deg = end - start;
        if (deg < 2) continue;

        // Pick two random incident edges
        int i1 = start + (curand(&rng) % deg);
        int i2 = start + (curand(&rng) % deg);
        if (i1 == i2) continue;

        int e1 = node_edges[i1], e2 = node_edges[i2];

        // Check orientation compatibility at node n
        int s1 = (edge_u[e1] == n) ? edge_u_head[e1] : edge_v_head[e1];
        int s2 = (edge_u[e2] == n) ? edge_u_head[e2] : edge_v_head[e2];
        if (s1 != s2) continue;  // incompatible

        // Check other endpoints match
        int m1 = (edge_u[e1] == n) ? edge_v[e1] : edge_u[e1];
        int m2 = (edge_u[e2] == n) ? edge_v[e2] : edge_u[e2];
        if (m1 != m2) continue;  // not a 2-cycle

        int s1m = (edge_u[e1] == m1) ? edge_u_head[e1] : edge_v_head[e1];
        int s2m = (edge_u[e2] == m2) ? edge_u_head[e2] : edge_v_head[e2];
        if (s1m != s2m) continue;

        // Propose: flow[e1]+=1, flow[e2]-=1
        int d1 = 1, d2 = -1;
        if (curand(&rng) % 2 == 0) { d1 = -1; d2 = 1; }

        int f1_new = flow[e1] + d1;
        int f2_new = flow[e2] + d2;

        if (f1_new < edge_lower[e1] || f1_new > edge_upper[e1]) continue;
        if (f2_new < edge_lower[e2] || f2_new > edge_upper[e2]) continue;

        // Compute cost delta
        float old_c = d_sa_eval_cost(edge_cost_type[e1], edge_cost_target[e1], edge_cost_weight[e1], flow[e1])
                     + d_sa_eval_cost(edge_cost_type[e2], edge_cost_target[e2], edge_cost_weight[e2], flow[e2]);
        float new_c = d_sa_eval_cost(edge_cost_type[e1], edge_cost_target[e1], edge_cost_weight[e1], f1_new)
                     + d_sa_eval_cost(edge_cost_type[e2], edge_cost_target[e2], edge_cost_weight[e2], f2_new);
        float delta = new_c - old_c;

        // Metropolis
        bool accept = (delta < 0.0f);
        if (!accept && T > 1e-8f) {
            float p = expf(-delta / T);
            accept = (curand_uniform(&rng) < p);
        }

        if (accept) {
            flow[e1] = f1_new;
            flow[e2] = f2_new;
            cost += delta;
        }
    }

    chain_cost[c] = cost;
}

SAResult sa_solve_bimdf(
    const BiMDFFlat& bimdf,
    int num_chains,
    int steps_per_chain,
    float temp_start,
    float temp_end)
{
    int N = bimdf.num_nodes, E = bimdf.num_edges;
    printf("[SA] %d chains × %d steps on %d nodes, %d edges\n",
           num_chains, steps_per_chain, N, E);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Build node-edge CSR
    std::vector<std::vector<int>> adj(N);
    for (int e = 0; e < E; ++e) {
        adj[bimdf.edge_u[e]].push_back(e);
        adj[bimdf.edge_v[e]].push_back(e);
    }
    std::vector<int> h_off(N + 1, 0), h_edges;
    for (int n = 0; n < N; ++n) {
        for (int e : adj[n]) h_edges.push_back(e);
        h_off[n + 1] = (int)h_edges.size();
    }

    // Compute initial cost
    float init_cost = 0;
    for (int e = 0; e < E; ++e) {
        float f = (float)bimdf.flow[e], t = (float)bimdf.edge_cost_target[e], w = (float)bimdf.edge_cost_weight[e];
        int ct = bimdf.edge_cost_type[e];
        if (ct == COST_ABS_DEVIATION) init_cost += w * fabsf(f - t);
        else if (ct == COST_QUAD_DEVIATION) init_cost += w * (f - t) * (f - t);
        else if (ct == COST_LINEAR) init_cost += w * f;
    }

    // Upload graph
    int *d_off, *d_ne, *d_eu, *d_ev, *d_euh, *d_evh, *d_ect, *d_elo, *d_eup;
    float *d_ect_tgt, *d_ect_wgt;
    cudaMalloc(&d_off, (N+1)*sizeof(int));
    cudaMalloc(&d_ne, h_edges.size()*sizeof(int));
    cudaMalloc(&d_eu, E*sizeof(int)); cudaMalloc(&d_ev, E*sizeof(int));
    cudaMalloc(&d_euh, E*sizeof(int)); cudaMalloc(&d_evh, E*sizeof(int));
    cudaMalloc(&d_ect, E*sizeof(int));
    cudaMalloc(&d_ect_tgt, E*sizeof(float)); cudaMalloc(&d_ect_wgt, E*sizeof(float));
    cudaMalloc(&d_elo, E*sizeof(int)); cudaMalloc(&d_eup, E*sizeof(int));

    cudaMemcpy(d_off, h_off.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ne, h_edges.data(), h_edges.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eu, bimdf.edge_u.data(), E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ev, bimdf.edge_v.data(), E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_euh, bimdf.edge_u_head.data(), E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_evh, bimdf.edge_v_head.data(), E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ect, bimdf.edge_cost_type.data(), E*sizeof(int), cudaMemcpyHostToDevice);

    // Convert double → float for GPU
    std::vector<float> h_tgt(E), h_wgt(E);
    for (int e = 0; e < E; ++e) { h_tgt[e] = (float)bimdf.edge_cost_target[e]; h_wgt[e] = (float)bimdf.edge_cost_weight[e]; }
    cudaMemcpy(d_ect_tgt, h_tgt.data(), E*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ect_wgt, h_wgt.data(), E*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_elo, bimdf.edge_lower.data(), E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eup, bimdf.edge_upper.data(), E*sizeof(int), cudaMemcpyHostToDevice);

    // Per-chain flow (copy initial flow to each chain)
    int *d_chain_flow;
    float *d_chain_cost;
    cudaMalloc(&d_chain_flow, (long long)num_chains * E * sizeof(int));
    cudaMalloc(&d_chain_cost, num_chains * sizeof(float));

    // Copy initial flow to all chains
    for (int c = 0; c < num_chains; ++c) {
        cudaMemcpy(d_chain_flow + (long long)c * E, bimdf.flow.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    }
    std::vector<float> h_costs(num_chains, init_cost);
    cudaMemcpy(d_chain_cost, h_costs.data(), num_chains * sizeof(float), cudaMemcpyHostToDevice);

    // Launch
    int block = 128;
    int grid = (num_chains + block - 1) / block;
    k_sa_run<<<grid, block>>>(
        d_off, d_ne, d_eu, d_ev, d_euh, d_evh, d_ect, d_ect_tgt, d_ect_wgt, d_elo, d_eup,
        d_chain_flow, d_chain_cost,
        N, E, steps_per_chain, temp_start, temp_end, 42ULL);
    cudaDeviceSynchronize();

    // Find best chain
    cudaMemcpy(h_costs.data(), d_chain_cost, num_chains * sizeof(float), cudaMemcpyDeviceToHost);
    int best_chain = std::min_element(h_costs.begin(), h_costs.end()) - h_costs.begin();

    SAResult result;
    result.best_flow.resize(E);
    cudaMemcpy(result.best_flow.data(), d_chain_flow + (long long)best_chain * E, E * sizeof(int), cudaMemcpyDeviceToHost);
    result.best_cost = h_costs[best_chain];
    result.total_steps = num_chains * steps_per_chain;

    // Cleanup
    cudaFree(d_off); cudaFree(d_ne); cudaFree(d_eu); cudaFree(d_ev);
    cudaFree(d_euh); cudaFree(d_evh); cudaFree(d_ect);
    cudaFree(d_ect_tgt); cudaFree(d_ect_wgt); cudaFree(d_elo); cudaFree(d_eup);
    cudaFree(d_chain_flow); cudaFree(d_chain_cost);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    printf("[SA] Best chain %d: cost=%.4f (init=%.4f), %.1fms\n",
           best_chain, result.best_cost, init_cost, result.time_ms);

    return result;
}

} // namespace cuda
} // namespace qw
