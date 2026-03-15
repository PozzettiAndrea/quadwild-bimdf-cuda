// ============================================================
// GPU ADMM solver for LP relaxation of BiMDF quantization
//
// Uses CONSENSUS ADMM to handle overlapping constraints:
// each edge has TWO z-copies (one per incident node).
// The x-update averages the two z-copies.
//
// Variables:
//   x[e]    — global edge flow (num_edges)
//   z[j]    — local copy per node-edge incidence (num_incidences)
//   u[j]    — dual per node-edge incidence (num_incidences)
//
// Per iteration:
//   z-update: project each node's local copies onto conservation
//   x-update: x[e] = argmin cost(x) + (ρ/2) Σ_j (x - z[j] + u[j])²
//   u-update: u[j] += x[e] - z[j]
// ============================================================

#include "admm_solver.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <algorithm>

namespace qw {
namespace cuda {

// ============================================================
// Kernel: z-update (per-node projection onto conservation)
//
// For node n with incident edges in node_edges[start..end]:
//   v_j = x[e_j] + u[j]  (current estimates)
//   Project: z[j] = v_j + sign_j * (demand - Σ sign_k * v_k) / degree
// ============================================================

__global__ void k_z_update(
    float* __restrict__ z,            // [num_incidences] local copies
    const float* __restrict__ x,      // [num_edges] global edge flows
    const float* __restrict__ u,      // [num_incidences] duals
    const int* __restrict__ node_edge_offsets,
    const int* __restrict__ node_edges,
    const int* __restrict__ node_edge_signs,
    const float* __restrict__ node_demand,
    int num_nodes)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= num_nodes) return;

    int start = node_edge_offsets[n];
    int end = node_edge_offsets[n + 1];
    int degree = end - start;
    if (degree == 0) return;

    float balance = 0.0f;
    for (int j = start; j < end; ++j) {
        int e = node_edges[j];
        float v = x[e] + u[j];
        balance += (float)node_edge_signs[j] * v;
    }

    float demand = node_demand[n];
    float correction = (demand - balance) / (float)degree;

    for (int j = start; j < end; ++j) {
        int e = node_edges[j];
        z[j] = x[e] + u[j] + (float)node_edge_signs[j] * correction;
    }
}

// ============================================================
// Kernel: x-update (proximal on cost + average consensus copies)
//
// x[e] = argmin cost_e(x) + (ρ/2) Σ_{j∈inc(e)} (x - z[j] + u[j])²
//
// For edge e with 2 incident nodes (j1, j2):
//   Let v_bar = (z[j1] - u[j1] + z[j2] - u[j2]) / 2
//   x[e] = prox_{cost/(2ρ)}(v_bar), clamped to bounds
// ============================================================

__global__ void k_x_update(
    float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ u,
    const int* __restrict__ edge_inc_offsets,  // [num_edges+1] incidence indices for each edge
    const int* __restrict__ edge_inc_indices,  // incidence indices (j values)
    const int* __restrict__ cost_type,
    const float* __restrict__ target,
    const float* __restrict__ weight,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    float rho,
    int num_edges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;

    // Average the consensus copies
    int start = edge_inc_offsets[e];
    int end = edge_inc_offsets[e + 1];
    int n_copies = end - start;

    float v_sum = 0.0f;
    for (int k = start; k < end; ++k) {
        int j = edge_inc_indices[k];
        v_sum += z[j] - u[j];
    }
    float v_bar = (n_copies > 0) ? v_sum / (float)n_copies : 0.0f;
    float effective_rho = rho * (float)n_copies;

    float w = weight[e];
    float t = target[e];
    float lo = lower[e];
    float hi = upper[e];
    float result;

    switch (cost_type[e]) {
        case ADMM_COST_ABS: {
            float kappa = w / effective_rho;
            float shifted = v_bar - t;
            if (shifted > kappa) result = t + shifted - kappa;
            else if (shifted < -kappa) result = t + shifted + kappa;
            else result = t;
            break;
        }
        case ADMM_COST_QUAD: {
            result = (effective_rho * v_bar + 2.0f * w * t) / (effective_rho + 2.0f * w);
            break;
        }
        case ADMM_COST_LINEAR: {
            result = v_bar - w / effective_rho;
            break;
        }
        default:
            result = v_bar;
            break;
    }

    x[e] = fmaxf(lo, fminf(hi, result));
}

// ============================================================
// Kernel: u-update (per incidence)
//   u[j] += x[e_j] - z[j]
// ============================================================

__global__ void k_u_update(
    float* __restrict__ u,
    const float* __restrict__ x,
    const float* __restrict__ z,
    const int* __restrict__ inc_edge,  // [num_incidences] edge index for each incidence
    int num_incidences)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_incidences) return;

    u[j] += x[inc_edge[j]] - z[j];
}

// ============================================================
// Kernel: compute primal+dual residuals
// ============================================================

__global__ void k_residuals(
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ z_prev,
    const int* __restrict__ inc_edge,
    float rho,
    float* __restrict__ primal_sq,
    float* __restrict__ dual_sq,
    int num_incidences)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int j = blockIdx.x * blockDim.x + tid;

    float p = 0, d = 0;
    if (j < num_incidences) {
        float r = x[inc_edge[j]] - z[j];
        p = r * r;
        float s = z[j] - z_prev[j];
        d = rho * rho * s * s;
    }
    sdata[tid] = p;
    sdata[tid + blockDim.x] = d;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + blockDim.x + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(primal_sq, sdata[0]);
        atomicAdd(dual_sq, sdata[blockDim.x]);
    }
}

// ============================================================
// Host driver
// ============================================================

ADMMResult admm_solve(
    const ADMMProblem& prob,
    float rho,
    int max_iter,
    float abs_tol,
    float rel_tol)
{
    int E = prob.num_edges;
    int N = prob.num_nodes;
    int J = (int)prob.node_edges.size();  // total incidences

    if (E == 0 || N == 0) {
        return {.flow = std::vector<float>(E, 0.0f), .iterations = 0};
    }

    printf("[ADMM] Problem: %d nodes, %d edges, %d incidences\n", N, E, J);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Build edge→incidence CSR (reverse of node→incidence)
    std::vector<std::vector<int>> edge_incs(E);
    std::vector<int> inc_edge_map(J);
    for (int j = 0; j < J; ++j) {
        int e = prob.node_edges[j];
        edge_incs[e].push_back(j);
        inc_edge_map[j] = e;
    }
    std::vector<int> edge_inc_offsets(E + 1, 0);
    std::vector<int> edge_inc_indices;
    for (int e = 0; e < E; ++e) {
        for (int j : edge_incs[e]) edge_inc_indices.push_back(j);
        edge_inc_offsets[e + 1] = (int)edge_inc_indices.size();
    }

    // GPU alloc
    float *d_x, *d_z, *d_z_prev, *d_u;
    int *d_cost_type;
    float *d_target, *d_weight, *d_lower, *d_upper, *d_demand;
    int *d_ne_off, *d_ne_idx, *d_ne_sign;
    int *d_ei_off, *d_ei_idx, *d_inc_edge;
    float *d_psq, *d_dsq;

    cudaMalloc(&d_x, E * sizeof(float));
    cudaMalloc(&d_z, J * sizeof(float));
    cudaMalloc(&d_z_prev, J * sizeof(float));
    cudaMalloc(&d_u, J * sizeof(float));
    cudaMalloc(&d_cost_type, E * sizeof(int));
    cudaMalloc(&d_target, E * sizeof(float));
    cudaMalloc(&d_weight, E * sizeof(float));
    cudaMalloc(&d_lower, E * sizeof(float));
    cudaMalloc(&d_upper, E * sizeof(float));
    cudaMalloc(&d_demand, N * sizeof(float));
    cudaMalloc(&d_ne_off, (N + 1) * sizeof(int));
    cudaMalloc(&d_ne_idx, J * sizeof(int));
    cudaMalloc(&d_ne_sign, J * sizeof(int));
    cudaMalloc(&d_ei_off, (E + 1) * sizeof(int));
    cudaMalloc(&d_ei_idx, (int)edge_inc_indices.size() * sizeof(int));
    cudaMalloc(&d_inc_edge, J * sizeof(int));
    cudaMalloc(&d_psq, sizeof(float));
    cudaMalloc(&d_dsq, sizeof(float));

    // Init x = clamp(target, lower, upper)
    std::vector<float> h_x(E);
    for (int e = 0; e < E; ++e) {
        h_x[e] = std::max(prob.edge_lower[e], std::min(prob.edge_upper[e], prob.edge_target[e]));
    }
    // Init z = x for each incidence
    std::vector<float> h_z(J);
    for (int j = 0; j < J; ++j) h_z[j] = h_x[inc_edge_map[j]];

    cudaMemcpy(d_x, h_x.data(), E * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z.data(), J * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_u, 0, J * sizeof(float));

    cudaMemcpy(d_cost_type, prob.edge_cost_type.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, prob.edge_target.data(), E * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, prob.edge_weight.data(), E * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lower, prob.edge_lower.data(), E * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upper, prob.edge_upper.data(), E * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_demand, prob.node_demand.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ne_off, prob.node_edge_offsets.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ne_idx, prob.node_edges.data(), J * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ne_sign, prob.node_edge_signs.data(), J * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ei_off, edge_inc_offsets.data(), (E + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ei_idx, edge_inc_indices.data(), edge_inc_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inc_edge, inc_edge_map.data(), J * sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid_e = (E + block - 1) / block;
    int grid_n = (N + block - 1) / block;
    int grid_j = (J + block - 1) / block;

    int iter;
    float h_pri = 0, h_dua = 0;

    for (iter = 0; iter < max_iter; ++iter) {
        cudaMemcpy(d_z_prev, d_z, J * sizeof(float), cudaMemcpyDeviceToDevice);

        // z-update: project per-node
        k_z_update<<<grid_n, block>>>(d_z, d_x, d_u, d_ne_off, d_ne_idx, d_ne_sign, d_demand, N);

        // x-update: proximal + consensus average
        k_x_update<<<grid_e, block>>>(d_x, d_z, d_u, d_ei_off, d_ei_idx,
            d_cost_type, d_target, d_weight, d_lower, d_upper, rho, E);

        // u-update
        k_u_update<<<grid_j, block>>>(d_u, d_x, d_z, d_inc_edge, J);

        // Convergence check every 50 iterations
        if ((iter + 1) % 50 == 0 || iter == max_iter - 1) {
            cudaMemset(d_psq, 0, sizeof(float));
            cudaMemset(d_dsq, 0, sizeof(float));
            k_residuals<<<grid_j, block, 2 * block * sizeof(float)>>>(
                d_x, d_z, d_z_prev, d_inc_edge, rho, d_psq, d_dsq, J);
            float hp, hd;
            cudaMemcpy(&hp, d_psq, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&hd, d_dsq, sizeof(float), cudaMemcpyDeviceToHost);
            h_pri = sqrtf(hp);
            h_dua = sqrtf(hd);
            if (h_pri < abs_tol * sqrtf((float)J) && h_dua < abs_tol * sqrtf((float)J)) {
                ++iter;
                break;
            }
        }
    }

    ADMMResult result;
    result.flow.resize(E);
    cudaMemcpy(result.flow.data(), d_x, E * sizeof(float), cudaMemcpyDeviceToHost);
    result.iterations = iter;
    result.primal_residual = h_pri;
    result.dual_residual = h_dua;

    float obj = 0;
    for (int e = 0; e < E; ++e) {
        float f = result.flow[e], t = prob.edge_target[e], w = prob.edge_weight[e];
        switch (prob.edge_cost_type[e]) {
            case ADMM_COST_ABS: obj += w * fabsf(f - t); break;
            case ADMM_COST_QUAD: obj += w * (f - t) * (f - t); break;
            case ADMM_COST_LINEAR: obj += w * f; break;
            default: break;
        }
    }
    result.objective = obj;

    cudaFree(d_x); cudaFree(d_z); cudaFree(d_z_prev); cudaFree(d_u);
    cudaFree(d_cost_type); cudaFree(d_target); cudaFree(d_weight);
    cudaFree(d_lower); cudaFree(d_upper); cudaFree(d_demand);
    cudaFree(d_ne_off); cudaFree(d_ne_idx); cudaFree(d_ne_sign);
    cudaFree(d_ei_off); cudaFree(d_ei_idx); cudaFree(d_inc_edge);
    cudaFree(d_psq); cudaFree(d_dsq);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    printf("[ADMM] Converged in %d iters (pri=%.4f dua=%.4f obj=%.4f) %.1fms\n",
           result.iterations, h_pri, h_dua, result.objective, result.time_ms);

    return result;
}

} // namespace cuda
} // namespace qw
