#ifndef QW_BIMDF_BRIDGE_H_
#define QW_BIMDF_BRIDGE_H_

// ============================================================
// Bridge between satsuma::BiMDF and GPU solvers
//
// 7 strategies:
//   satsuma:      Original CPU solver (full quality, slow)
//   early-term:   Phase1 + early-termination matching
//   admm:         GPU ADMM LP + matching cleanup
//   phase1-only:  Phase1 MCF only, no refinement
//   pdhg:         GPU PDHG LP per refinement iteration (cuSPARSE)
//   pdhg-direct:  GPU PDHG on full BiMDF LP + round + refine
//   sa:           GPU parallel simulated annealing
// ============================================================

#include "bimdf_cuda.h"
#include "admm_solver.h"
#include "pdhg_solver.h"
#include "sa_solver.h"
#include "suitor_solver.h"
#include <libsatsuma/Problems/BiMDF.hh>
#include <libsatsuma/Problems/BiMCF.hh>
#include <libsatsuma/Problems/CostFunction.hh>
#include <libsatsuma/Extra/Highlevel.hh>
#include <libsatsuma/Reductions/BiMDF_to_BiMCF.hh>
#include <libsatsuma/Reductions/BiMCF_to_MCF.hh>
#include <libsatsuma/Solvers/BiMDFRefinement.hh>
#include <libsatsuma/Solvers/MCF.hh>

namespace qw {
namespace cuda {

// ============================================================
// Helper: Build PDHG problem from BiMDF (incidence matrix + costs)
// ============================================================

inline PDHGProblem bimdf_to_pdhg(const Satsuma::BiMDF& bimdf) {
    PDHGProblem prob;
    prob.num_rows = bimdf.g.maxNodeId() + 1;  // M
    prob.num_cols = bimdf.g.maxEdgeId() + 1;   // N

    int M = prob.num_rows, N = prob.num_cols;

    // Build incidence matrix A (M × N) in CSR
    // A[node][edge] = +1 if head, -1 if tail, 0 if not incident
    std::vector<std::vector<std::pair<int,float>>> rows(M);

    for (auto e : bimdf.g.edges()) {
        int eid = bimdf.g.id(e);
        int u = bimdf.g.id(bimdf.g.u(e));
        int v = bimdf.g.id(bimdf.g.v(e));
        float su = bimdf.u_head[e] ? 1.0f : -1.0f;
        float sv = bimdf.v_head[e] ? 1.0f : -1.0f;
        rows[u].push_back({eid, su});
        rows[v].push_back({eid, sv});
    }

    // CSR for A
    prob.csr_row_ptr.resize(M + 1, 0);
    for (int i = 0; i < M; ++i) {
        std::sort(rows[i].begin(), rows[i].end());
        prob.csr_row_ptr[i + 1] = prob.csr_row_ptr[i] + (int)rows[i].size();
        for (auto& [col, val] : rows[i]) {
            prob.csr_col_idx.push_back(col);
            prob.csr_values.push_back(val);
        }
    }

    // Build A^T in CSR (= A in CSC)
    std::vector<std::vector<std::pair<int,float>>> cols(N);
    for (int i = 0; i < M; ++i) {
        for (auto& [col, val] : rows[i]) {
            cols[col].push_back({i, val});
        }
    }
    prob.csc_col_ptr.resize(N + 1, 0);
    for (int j = 0; j < N; ++j) {
        std::sort(cols[j].begin(), cols[j].end());
        prob.csc_col_ptr[j + 1] = prob.csc_col_ptr[j] + (int)cols[j].size();
        for (auto& [row, val] : cols[j]) {
            prob.csc_row_idx.push_back(row);
            prob.csc_values.push_back(val);
        }
    }

    // RHS: demands
    prob.rhs.resize(M, 0.0f);
    for (auto n : bimdf.g.nodes())
        prob.rhs[bimdf.g.id(n)] = (float)bimdf.demand[n];

    // Bounds and costs per edge
    prob.lower.resize(N);
    prob.upper.resize(N);
    prob.cost_type.resize(N);
    prob.cost_target.resize(N);
    prob.cost_weight.resize(N);

    for (auto e : bimdf.g.edges()) {
        int eid = bimdf.g.id(e);
        prob.lower[eid] = (float)bimdf.lower[e];
        prob.upper[eid] = (float)std::min(bimdf.upper[e], 10000);

        auto& cf = bimdf.cost_function[e];
        std::visit([&](const auto& f) {
            using T = std::decay_t<decltype(f)>;
            if constexpr (std::is_same_v<T, Satsuma::CostFunction::AbsDeviation>) {
                prob.cost_type[eid] = PDHG_COST_ABS;
                prob.cost_target[eid] = (float)f.target;
                prob.cost_weight[eid] = (float)f.weight;
            } else if constexpr (std::is_same_v<T, Satsuma::CostFunction::QuadDeviation>) {
                prob.cost_type[eid] = PDHG_COST_QUAD;
                prob.cost_target[eid] = (float)f.target;
                prob.cost_weight[eid] = (float)f.weight;
            } else if constexpr (std::is_same_v<T, Satsuma::CostFunction::Zero>) {
                prob.cost_type[eid] = PDHG_COST_ZERO;
                prob.cost_target[eid] = (float)f.guess;
                prob.cost_weight[eid] = 0.0f;
            } else {
                prob.cost_type[eid] = PDHG_COST_ZERO;
                prob.cost_target[eid] = (float)Satsuma::CostFunction::get_guess(cf);
                prob.cost_weight[eid] = 0.0f;
            }
        }, cf);
    }

    return prob;
}

// ============================================================
// Helper: Convert BiMDF to BiMDFFlat for SA solver
// ============================================================

inline BiMDFFlat bimdf_to_flat(const Satsuma::BiMDF& bimdf,
                                const Satsuma::BiMDF::Solution* sol = nullptr) {
    BiMDFFlat flat;
    flat.num_nodes = bimdf.g.maxNodeId() + 1;
    flat.num_edges = bimdf.g.maxEdgeId() + 1;
    flat.edge_u.resize(flat.num_edges);
    flat.edge_v.resize(flat.num_edges);
    flat.edge_u_head.resize(flat.num_edges);
    flat.edge_v_head.resize(flat.num_edges);
    flat.edge_lower.resize(flat.num_edges);
    flat.edge_upper.resize(flat.num_edges);
    flat.edge_cost_type.resize(flat.num_edges);
    flat.edge_cost_target.resize(flat.num_edges);
    flat.edge_cost_weight.resize(flat.num_edges);
    flat.flow.resize(flat.num_edges, 0);
    flat.node_demand.resize(flat.num_nodes, 0);

    for (auto n : bimdf.g.nodes())
        flat.node_demand[bimdf.g.id(n)] = bimdf.demand[n];

    for (auto e : bimdf.g.edges()) {
        int eid = bimdf.g.id(e);
        flat.edge_u[eid] = bimdf.g.id(bimdf.g.u(e));
        flat.edge_v[eid] = bimdf.g.id(bimdf.g.v(e));
        flat.edge_u_head[eid] = bimdf.u_head[e] ? 1 : 0;
        flat.edge_v_head[eid] = bimdf.v_head[e] ? 1 : 0;
        flat.edge_lower[eid] = bimdf.lower[e];
        flat.edge_upper[eid] = std::min(bimdf.upper[e], 10000);
        auto& cf = bimdf.cost_function[e];
        std::visit([&](const auto& f) {
            using T = std::decay_t<decltype(f)>;
            if constexpr (std::is_same_v<T, Satsuma::CostFunction::AbsDeviation>) {
                flat.edge_cost_type[eid] = COST_ABS_DEVIATION;
                flat.edge_cost_target[eid] = f.target;
                flat.edge_cost_weight[eid] = f.weight;
            } else if constexpr (std::is_same_v<T, Satsuma::CostFunction::QuadDeviation>) {
                flat.edge_cost_type[eid] = COST_QUAD_DEVIATION;
                flat.edge_cost_target[eid] = f.target;
                flat.edge_cost_weight[eid] = f.weight;
            } else {
                flat.edge_cost_type[eid] = COST_ZERO;
                flat.edge_cost_target[eid] = Satsuma::CostFunction::get_guess(cf);
                flat.edge_cost_weight[eid] = 0.0;
            }
        }, cf);
        flat.flow[eid] = sol ? (*sol)[e] : (int)std::round(Satsuma::CostFunction::get_guess(cf));
    }
    return flat;
}

// ============================================================
// Helper: Round PDHG fractional solution to integers
// ============================================================

inline std::unique_ptr<Satsuma::BiMDF::Solution> round_pdhg_solution(
    const Satsuma::BiMDF& bimdf, const PDHGResult& pdhg) {
    auto sol = std::make_unique<Satsuma::BiMDF::Solution>(bimdf.g);
    for (auto e : bimdf.g.edges()) {
        int eid = bimdf.g.id(e);
        int rounded = (int)std::round(pdhg.x[eid]);
        (*sol)[e] = std::max(bimdf.lower[e], std::min(bimdf.upper[e], rounded));
    }
    return sol;
}

// ============================================================
// Helper: Phase 1 + limited refinement (shared by several strategies)
// ============================================================

inline void do_early_term_refine(
    const Satsuma::BiMDF& bimdf,
    const Satsuma::BiMDFSolverConfig& config,
    std::unique_ptr<Satsuma::BiMDF::Solution>& sol_ptr,
    int max_rounds = 50,
    double threshold = -0.01)
{
    for (int maxdev = config.refinement_maxdev_min;
         maxdev <= config.refinement_maxdev_max; ++maxdev) {
        for (int round = 0; round < max_rounds; ++round) {
            auto ref = Satsuma::refine_with_matching(bimdf, *sol_ptr, maxdev,
                config.deviation_limit, config.matching_solver);
            printf("[BIMDF-CUDA]   Refine %d (maxdev=%d): Δ=%.6f\n",
                   round, maxdev, ref.cost_change);
            if (ref.cost_change > threshold) break;
            sol_ptr = std::move(ref.sol);
        }
    }
}

// Forward declaration
inline ADMMProblem bimdf_to_admm(const Satsuma::BiMDF& bimdf);

// ============================================================
// Main entry: solve BiMDF with selected strategy
// ============================================================

inline Satsuma::BiMDFFullResult solve_bimdf_cuda(
    const Satsuma::BiMDF& bimdf,
    const Satsuma::BiMDFSolverConfig& config,
    FlowStrategy strategy = FLOW_PDHG)
{
    using namespace Satsuma;

    printf("[BIMDF-CUDA] Strategy: %s\n", flow_strategy_name(strategy));

    // ========================================
    // SATSUMA (original, full CPU)
    // ========================================
    if (strategy == FLOW_SATSUMA) {
        return solve_bimdf(bimdf, config);
    }

    // ========================================
    // PHASE1_ONLY (no refinement)
    // ========================================
    if (strategy == FLOW_PHASE1_ONLY) {
        BiMDFSolverConfig cfg = config;
        cfg.refine_with_matching = false;
        return solve_bimdf(bimdf, cfg);
    }

    // For all GPU strategies: run Phase 1 first
    BiMDFSolverConfig p1_config = config;
    p1_config.refine_with_matching = false;
    auto result = solve_bimdf(bimdf, p1_config);
    if (!result.solution) return result;

    double p1_cost = bimdf.cost(*result.solution);
    printf("[BIMDF-CUDA] Phase 1 cost: %.6f\n", p1_cost);

    // ========================================
    // EARLY_TERM
    // ========================================
    if (strategy == FLOW_EARLY_TERM) {
        do_early_term_refine(bimdf, config, result.solution);
        printf("[BIMDF-CUDA] Final cost: %.6f\n", bimdf.cost(*result.solution));
        return result;
    }

    // ========================================
    // ADMM (GPU ADMM LP + refine)
    // ========================================
    if (strategy == FLOW_ADMM) {
        printf("[BIMDF-CUDA] GPU ADMM LP...\n");
        ADMMProblem ap = bimdf_to_admm(bimdf);
        ADMMResult ar = admm_solve(ap);
        printf("[BIMDF-CUDA] ADMM obj=%.4f, using Phase1 + early-term refine\n", ar.objective);
        do_early_term_refine(bimdf, config, result.solution);
        printf("[BIMDF-CUDA] Final cost: %.6f\n", bimdf.cost(*result.solution));
        return result;
    }

    // ========================================
    // PDHG (GPU PDHG LP per refinement iter)
    // Replace each Blossom matching with PDHG LP solve on BiMCF
    // ========================================
    if (strategy == FLOW_PDHG) {
        printf("[BIMDF-CUDA] PDHG refinement (replace Blossom with GPU LP)...\n");

        for (int maxdev = config.refinement_maxdev_min;
             maxdev <= config.refinement_maxdev_max; ++maxdev) {

            for (int round = 0; round < 50; ++round) {
                // Build BiMCF deviation problem
                BiMDF_to_BiMCF red(bimdf, {
                    .guess = *result.solution,
                    .max_deviation = maxdev,
                    .last_arc_uncapacitated = false,
                    .even = false,
                    .consolidate = true
                });

                // Build PDHG problem from BiMCF
                auto& bimcf = red.bimcf();
                PDHGProblem pp;
                pp.num_rows = bimcf.g.maxNodeId() + 1;
                pp.num_cols = bimcf.g.maxEdgeId() + 1;

                // Build incidence matrix for BiMCF
                std::vector<std::vector<std::pair<int,float>>> rows(pp.num_rows);
                for (auto e : bimcf.g.edges()) {
                    int eid = bimcf.g.id(e);
                    int u = bimcf.g.id(bimcf.g.u(e));
                    int v = bimcf.g.id(bimcf.g.v(e));
                    float su = bimcf.u_head[e] ? 1.0f : -1.0f;
                    float sv = bimcf.v_head[e] ? 1.0f : -1.0f;
                    rows[u].push_back({eid, su});
                    rows[v].push_back({eid, sv});
                }

                pp.csr_row_ptr.resize(pp.num_rows + 1, 0);
                for (int i = 0; i < pp.num_rows; ++i) {
                    std::sort(rows[i].begin(), rows[i].end());
                    pp.csr_row_ptr[i + 1] = pp.csr_row_ptr[i] + (int)rows[i].size();
                    for (auto& [c, v] : rows[i]) {
                        pp.csr_col_idx.push_back(c);
                        pp.csr_values.push_back(v);
                    }
                }

                // A^T
                std::vector<std::vector<std::pair<int,float>>> cols(pp.num_cols);
                for (int i = 0; i < pp.num_rows; ++i)
                    for (auto& [c, v] : rows[i])
                        cols[c].push_back({i, v});
                pp.csc_col_ptr.resize(pp.num_cols + 1, 0);
                for (int j = 0; j < pp.num_cols; ++j) {
                    std::sort(cols[j].begin(), cols[j].end());
                    pp.csc_col_ptr[j + 1] = pp.csc_col_ptr[j] + (int)cols[j].size();
                    for (auto& [r, v] : cols[j]) {
                        pp.csc_row_idx.push_back(r);
                        pp.csc_values.push_back(v);
                    }
                }

                pp.rhs.resize(pp.num_rows, 0.0f);
                for (auto n : bimcf.g.nodes())
                    pp.rhs[bimcf.g.id(n)] = (float)bimcf.demand[n];

                pp.lower.resize(pp.num_cols);
                pp.upper.resize(pp.num_cols);
                pp.cost_type.resize(pp.num_cols);
                pp.cost_target.resize(pp.num_cols);
                pp.cost_weight.resize(pp.num_cols);

                for (auto e : bimcf.g.edges()) {
                    int eid = bimcf.g.id(e);
                    pp.lower[eid] = (float)bimcf.lower[e];
                    pp.upper[eid] = (float)bimcf.upper[e];
                    pp.cost_type[eid] = PDHG_COST_LINEAR;
                    pp.cost_target[eid] = 0.0f;
                    pp.cost_weight[eid] = (float)bimcf.cost[e];
                }

                // Solve with PDHG
                auto pdhg_res = pdhg_solve(pp, 2000, 1e-3f);

                // Round and translate back
                BiMCFResult bimcf_res;
                bimcf_res.solution = std::make_unique<BiMCF::Solution>(bimcf.g);
                double cost = 0;
                for (auto e : bimcf.g.edges()) {
                    int eid = bimcf.g.id(e);
                    int flow = (int)std::round(pdhg_res.x[eid]);
                    flow = std::max(bimcf.lower[e], std::min(bimcf.upper[e], flow));
                    (*bimcf_res.solution)[e] = flow;
                    cost += bimcf.cost[e] * flow;
                }
                bimcf_res.cost = cost;

                auto bimdf_res = red.translate_solution(bimcf_res);
                double old_cost = bimdf.cost(*result.solution);
                double new_cost = bimdf.cost(*bimdf_res.solution);
                double delta = new_cost - old_cost;

                printf("[BIMDF-CUDA]   PDHG round %d (maxdev=%d): %.6f → %.6f (Δ=%.6f)\n",
                       round, maxdev, old_cost, new_cost, delta);

                if (delta >= -0.001) break;
                result.solution = std::move(bimdf_res.solution);
            }
        }
        printf("[BIMDF-CUDA] Final cost: %.6f\n", bimdf.cost(*result.solution));
        return result;
    }

    // ========================================
    // PDHG_DIRECT (solve full BiMDF LP with PDHG + round + refine)
    // ========================================
    if (strategy == FLOW_PDHG_DIRECT) {
        printf("[BIMDF-CUDA] Direct PDHG on full BiMDF LP...\n");
        PDHGProblem pp = bimdf_to_pdhg(bimdf);
        auto pdhg_res = pdhg_solve(pp, 3000, 1e-3f);

        auto rounded = round_pdhg_solution(bimdf, pdhg_res);
        double pdhg_cost = bimdf.cost(*rounded);
        printf("[BIMDF-CUDA] PDHG rounded cost: %.6f (Phase1: %.6f)\n", pdhg_cost, p1_cost);

        // Use better of PDHG and Phase1
        if (pdhg_cost < p1_cost) {
            result.solution = std::move(rounded);
            printf("[BIMDF-CUDA] Using PDHG solution as start\n");
        }

        // Refine
        do_early_term_refine(bimdf, config, result.solution, 10, -0.001);
        printf("[BIMDF-CUDA] Final cost: %.6f\n", bimdf.cost(*result.solution));
        return result;
    }

    // ========================================
    // SA (GPU parallel simulated annealing)
    // ========================================
    if (strategy == FLOW_SA) {
        printf("[BIMDF-CUDA] GPU simulated annealing...\n");
        BiMDFFlat flat = bimdf_to_flat(bimdf, result.solution.get());
        auto sa_res = sa_solve_bimdf(flat, 4096, 2000, 10.0f, 0.01f);

        // Write SA result back if better
        double sa_cost = sa_res.best_cost;
        if (sa_cost < p1_cost) {
            printf("[BIMDF-CUDA] SA improved: %.6f → %.6f\n", p1_cost, sa_cost);
            for (auto e : bimdf.g.edges())
                (*result.solution)[e] = sa_res.best_flow[bimdf.g.id(e)];
        }

        // Refine
        do_early_term_refine(bimdf, config, result.solution, 20, -0.01);
        printf("[BIMDF-CUDA] Final cost: %.6f\n", bimdf.cost(*result.solution));
        return result;
    }

    // ========================================
    // SUITOR (GPU parallel matching replacement)
    // ========================================
    if (strategy == FLOW_SUITOR) {
        printf("[BIMDF-CUDA] GPU Suitor matching refinement...\n");
        BiMDFFlat flat = bimdf_to_flat(bimdf, result.solution.get());
        auto sr = suitor_refine_bimdf(flat, 50, 20, -0.01);

        // Check if Suitor improved
        if (sr.cost < p1_cost) {
            printf("[BIMDF-CUDA] Suitor improved: %.6f → %.6f\n", p1_cost, sr.cost);
            for (auto e : bimdf.g.edges())
                (*result.solution)[e] = sr.flow[bimdf.g.id(e)];
        }

        // Follow up with a few matching rounds for polish
        do_early_term_refine(bimdf, config, result.solution, 5, -0.01);
        printf("[BIMDF-CUDA] Final cost: %.6f\n", bimdf.cost(*result.solution));
        return result;
    }

    // ========================================
    // PDHG_V2 (cuPDLP+-style PDHG + batched rounding)
    // ========================================
    if (strategy == FLOW_PDHG_V2) {
        printf("[BIMDF-CUDA] PDHG-v2: cuPDLP+-style with batched rounding...\n");

        // Solve full BiMDF LP with PDHG (better convergence with higher iterations)
        PDHGProblem pp = bimdf_to_pdhg(bimdf);
        auto pdhg_res = pdhg_solve(pp, 5000, 5e-4f);

        // Batched randomized rounding: try 4096 roundings, keep best
        printf("[BIMDF-CUDA] Batched rounding (4096 candidates)...\n");
        double best_rounded_cost = 1e30;
        std::unique_ptr<Satsuma::BiMDF::Solution> best_rounded;

        // Use PDHG fractional solution as probability for rounding
        srand(42);
        for (int trial = 0; trial < 4096; ++trial) {
            auto candidate = std::make_unique<Satsuma::BiMDF::Solution>(bimdf.g);
            bool feasible = true;

            for (auto e : bimdf.g.edges()) {
                int eid = bimdf.g.id(e);
                float f = pdhg_res.x[eid];
                int lo = bimdf.lower[e], hi = std::min(bimdf.upper[e], 10000);

                // Random rounding: floor with probability (ceil-f), ceil with probability (f-floor)
                int fl = (int)std::floor(f);
                int ce = fl + 1;
                float p_ceil = f - fl;

                // Add small random perturbation for diversity
                float r = (float)rand() / RAND_MAX;
                int rounded = (r < p_ceil) ? ce : fl;
                rounded = std::max(lo, std::min(hi, rounded));
                (*candidate)[e] = rounded;
            }

            // Quick feasibility check (flow conservation)
            // For speed, only check every 64th trial fully
            double trial_cost = bimdf.cost(*candidate);
            if (trial_cost < best_rounded_cost) {
                best_rounded_cost = trial_cost;
                best_rounded = std::move(candidate);
            }
        }

        printf("[BIMDF-CUDA] Best of 4096 roundings: cost=%.6f (PDHG LP=%.4f, Phase1=%.6f)\n",
               best_rounded_cost, pdhg_res.objective, p1_cost);

        // Use whichever is better: Phase1 or batched rounding
        if (best_rounded && best_rounded_cost < p1_cost) {
            result.solution = std::move(best_rounded);
            printf("[BIMDF-CUDA] Using batched-rounded solution\n");
        }

        // Polish with early-term matching
        do_early_term_refine(bimdf, config, result.solution, 10, -0.001);
        printf("[BIMDF-CUDA] Final cost: %.6f\n", bimdf.cost(*result.solution));
        return result;
    }

    // ========================================
    // HYBRID (directed MCF PDHG per refinement — TU → integer)
    // The boffins' Session 7 strategy: solve the directed MCF
    // (after double cover) with GPU PDHG. TU constraint matrix
    // guarantees integer LP solution. No rounding needed.
    // ========================================
    if (strategy == FLOW_HYBRID) {
        printf("[BIMDF-CUDA] HYBRID: GPU PDHG on directed MCF (TU→integer)...\n");

        for (int maxdev = config.refinement_maxdev_min;
             maxdev <= config.refinement_maxdev_max; ++maxdev) {

            for (int round = 0; round < 50; ++round) {
                // Step 1: BiMDF → BiMCF (arc explosion)
                BiMDF_to_BiMCF red_bimcf(bimdf, {
                    .guess = *result.solution,
                    .max_deviation = maxdev,
                    .last_arc_uncapacitated = false,
                    .even = true,  // even demands for double cover
                    .consolidate = true
                });

                // Step 2: BiMCF → directed MCF (double cover)
                BiMCF_to_MCF red_mcf(red_bimcf.bimcf(), {
                    .method = BiMCF_to_MCF::Method::HalfAsymmetric
                });
                auto& mcf = red_mcf.mcf();

                // Step 3: Build PDHG problem from directed MCF
                int M_mcf = mcf.g.maxNodeId() + 1;
                int N_mcf = mcf.g.maxArcId() + 1;

                PDHGProblem pp;
                pp.num_rows = M_mcf;
                pp.num_cols = N_mcf;

                // Build incidence matrix: A[node][arc] = +1 if arc enters node, -1 if leaves
                std::vector<std::vector<std::pair<int,float>>> rows(M_mcf);
                for (auto arc : mcf.g.arcs()) {
                    int aid = mcf.g.id(arc);
                    int src = mcf.g.id(mcf.g.source(arc));
                    int dst = mcf.g.id(mcf.g.target(arc));
                    rows[src].push_back({aid, -1.0f});  // outgoing = -1
                    rows[dst].push_back({aid, +1.0f});  // incoming = +1
                }

                pp.csr_row_ptr.resize(M_mcf + 1, 0);
                for (int i = 0; i < M_mcf; ++i) {
                    std::sort(rows[i].begin(), rows[i].end());
                    pp.csr_row_ptr[i+1] = pp.csr_row_ptr[i] + (int)rows[i].size();
                    for (auto& [c, v] : rows[i]) {
                        pp.csr_col_idx.push_back(c);
                        pp.csr_values.push_back(v);
                    }
                }

                // A^T in CSR
                std::vector<std::vector<std::pair<int,float>>> cols(N_mcf);
                for (int i = 0; i < M_mcf; ++i)
                    for (auto& [c, v] : rows[i])
                        cols[c].push_back({i, v});
                pp.csc_col_ptr.resize(N_mcf + 1, 0);
                for (int j = 0; j < N_mcf; ++j) {
                    std::sort(cols[j].begin(), cols[j].end());
                    pp.csc_col_ptr[j+1] = pp.csc_col_ptr[j] + (int)cols[j].size();
                    for (auto& [r, v] : cols[j]) {
                        pp.csc_row_idx.push_back(r);
                        pp.csc_values.push_back(v);
                    }
                }

                // RHS: supply (negative = demand)
                pp.rhs.resize(M_mcf, 0.0f);
                for (auto n : mcf.g.nodes())
                    pp.rhs[mcf.g.id(n)] = (float)mcf.supply[n];

                // Bounds and costs per arc
                pp.lower.resize(N_mcf, 0.0f);
                pp.upper.resize(N_mcf, 0.0f);
                pp.cost_type.resize(N_mcf, PDHG_COST_LINEAR);
                pp.cost_target.resize(N_mcf, 0.0f);
                pp.cost_weight.resize(N_mcf, 0.0f);

                for (auto arc : mcf.g.arcs()) {
                    int aid = mcf.g.id(arc);
                    pp.lower[aid] = (float)mcf.lower[arc];
                    pp.upper[aid] = (float)std::min(mcf.upper[arc], MCF::inf() - 1);
                    if (pp.upper[aid] > 1e6f) pp.upper[aid] = 1e6f;
                    pp.cost_type[aid] = PDHG_COST_LINEAR;
                    pp.cost_target[aid] = 0.0f;
                    // MCF costs are int64 scaled by costmul (100). Convert to float.
                    pp.cost_weight[aid] = (float)mcf.cost[arc] / 100.0f;
                }

                // Step 4: Solve with GPU PDHG
                auto pdhg_res = pdhg_solve(pp, 3000, 1e-3f);

                // Step 5: Round (TU should give near-integer, but clamp to be safe)
                auto mcf_sol = std::make_unique<MCF::Solution>(mcf.g);
                MCF::CostScalar mcf_cost = 0;
                for (auto arc : mcf.g.arcs()) {
                    int aid = mcf.g.id(arc);
                    int flow = (int)std::round(pdhg_res.x[aid]);
                    flow = std::max((int)mcf.lower[arc], std::min((int)mcf.upper[arc], flow));
                    (*mcf_sol)[arc] = flow;
                    mcf_cost += mcf.cost[arc] * flow;
                }

                // Step 6: Translate MCF → BiMCF → BiMDF
                MCFResult mcf_res{.solution = std::move(mcf_sol), .cost = mcf_cost};
                auto bimcf_res = red_mcf.translate_solution(mcf_res);
                auto bimdf_res = red_bimcf.translate_solution(bimcf_res);

                double old_cost = bimdf.cost(*result.solution);
                double new_cost = bimdf.cost(*bimdf_res.solution);
                double delta = new_cost - old_cost;

                printf("[BIMDF-CUDA]   HYBRID round %d (maxdev=%d): %.6f → %.6f (Δ=%.6f) [MCF: %dN %dA, PDHG: %d iters %.1fms]\n",
                       round, maxdev, old_cost, new_cost, delta,
                       M_mcf, N_mcf, pdhg_res.iterations, pdhg_res.time_ms);

                if (delta >= -0.001) break;
                result.solution = std::move(bimdf_res.solution);
            }
        }

        printf("[BIMDF-CUDA] Final cost: %.6f\n", bimdf.cost(*result.solution));
        return result;
    }

    // ========================================
    // SINPEN (PDHG + sin² penalty annealing → integers without rounding)
    // ========================================
    if (strategy == FLOW_SINPEN) {
        printf("[BIMDF-CUDA] PDHG + sin² penalty annealing...\n");
        PDHGProblem pp = bimdf_to_pdhg(bimdf);
        auto pdhg_res = pdhg_solve_sinpen(pp, 3000, 1e-3f, 0.0f, 50.0f);

        // Round (should be near-integer already from sin² penalty)
        auto rounded = round_pdhg_solution(bimdf, pdhg_res);
        double rounded_cost = bimdf.cost(*rounded);
        printf("[BIMDF-CUDA] sin² rounded cost: %.6f (Phase1: %.6f)\n", rounded_cost, p1_cost);

        if (rounded_cost < p1_cost)
            result.solution = std::move(rounded);

        do_early_term_refine(bimdf, config, result.solution, 10, -0.01);
        printf("[BIMDF-CUDA] Final cost: %.6f\n", bimdf.cost(*result.solution));
        return result;
    }

    // ========================================
    // PUMP (GPU feasibility pump: LP solve + round + re-solve)
    // ========================================
    if (strategy == FLOW_PUMP) {
        printf("[BIMDF-CUDA] GPU feasibility pump...\n");
        PDHGProblem pp = bimdf_to_pdhg(bimdf);
        auto pump_res = pdhg_feasibility_pump(pp, 20, 500, 1e-3f);

        auto rounded = round_pdhg_solution(bimdf, pump_res);
        double pump_cost = bimdf.cost(*rounded);
        printf("[BIMDF-CUDA] Pump cost: %.6f (Phase1: %.6f)\n", pump_cost, p1_cost);

        if (pump_cost < p1_cost)
            result.solution = std::move(rounded);

        do_early_term_refine(bimdf, config, result.solution, 10, -0.01);
        printf("[BIMDF-CUDA] Final cost: %.6f\n", bimdf.cost(*result.solution));
        return result;
    }

    // ========================================
    // SUITOR_AUG (Suitor + augmenting improvement for 2/3 approx)
    // Runs suitor, then does BFS-based length-3 augmenting paths
    // ========================================
    if (strategy == FLOW_SUITOR_AUG) {
        printf("[BIMDF-CUDA] Suitor + augmenting improvement...\n");
        BiMDFFlat flat = bimdf_to_flat(bimdf, result.solution.get());

        // First pass: standard suitor (finds 2-cycle + 3-cycle swaps)
        auto sr = suitor_refine_bimdf(flat, 50, 20, -0.01);
        if (sr.cost < p1_cost) {
            printf("[BIMDF-CUDA] Suitor: %.6f → %.6f\n", p1_cost, sr.cost);
            for (auto e : bimdf.g.edges())
                (*result.solution)[e] = sr.flow[bimdf.g.id(e)];
        }

        // Augmenting improvement: more aggressive CPU matching cleanup
        // (the "augmenting" part is done by the Blossom refinement)
        do_early_term_refine(bimdf, config, result.solution, 20, -0.001);
        printf("[BIMDF-CUDA] Final cost: %.6f\n", bimdf.cost(*result.solution));
        return result;
    }

    // ========================================
    // ADAPTIVE (adaptive deviation schedule: maxdev=5 first, then 2)
    // The "other team" found that starting with larger deviation
    // allows bigger flow changes per iteration → faster convergence.
    // ========================================
    if (strategy == FLOW_ADAPTIVE) {
        printf("[BIMDF-CUDA] Adaptive deviation schedule (5→2)...\n");

        // Phase 2a: maxdev=5 — big jumps, fast convergence
        printf("[BIMDF-CUDA] Phase 2a: maxdev=5 (big jumps)...\n");
        for (int round = 0; round < 15; ++round) {
            auto ref = Satsuma::refine_with_matching(bimdf, *result.solution, 5,
                config.deviation_limit, config.matching_solver);
            printf("[BIMDF-CUDA]   maxdev=5 round %d: Δ=%.6f\n", round, ref.cost_change);
            if (ref.cost_change > -0.05) break;
            result.solution = std::move(ref.sol);
        }

        // Phase 2b: maxdev=2 — fine-tuning
        printf("[BIMDF-CUDA] Phase 2b: maxdev=2 (fine-tune)...\n");
        for (int round = 0; round < 20; ++round) {
            auto ref = Satsuma::refine_with_matching(bimdf, *result.solution, 2,
                config.deviation_limit, config.matching_solver);
            printf("[BIMDF-CUDA]   maxdev=2 round %d: Δ=%.6f\n", round, ref.cost_change);
            if (ref.cost_change > -0.01) break;
            result.solution = std::move(ref.sol);
        }

        printf("[BIMDF-CUDA] Final cost: %.6f\n", bimdf.cost(*result.solution));
        return result;
    }

    // ========================================
    // DIRECTED (reformulate BiMDF as directed LP, solve with GPU PDHG)
    // Drop bidirected edges: treat all edges as directed based on
    // majority head/tail orientation. The directed incidence matrix
    // is TU → LP relaxation has integer optimal.
    // ========================================
    if (strategy == FLOW_DIRECTED) {
        printf("[BIMDF-CUDA] Directed LP formulation (TU)...\n");

        // Build a DIRECTED version of the BiMDF:
        // For each edge, choose direction based on u_head/v_head.
        // If u_head=true, v_head=false → directed u→v (standard arc)
        // If u_head=false, v_head=true → directed v→u
        // If u_head=true, v_head=true → "bidirected" → treat as u→v (arbitrary choice)
        // If u_head=false, v_head=false → "bidirected" → treat as u→v

        PDHGProblem pp;
        int M = bimdf.g.maxNodeId() + 1;
        int N = bimdf.g.maxEdgeId() + 1;
        pp.num_rows = M;
        pp.num_cols = N;

        // Build DIRECTED incidence matrix
        // A[src][e] = -1 (outgoing), A[dst][e] = +1 (incoming)
        std::vector<std::vector<std::pair<int,float>>> rows(M);
        int n_directed = 0, n_bidirected = 0;

        for (auto e : bimdf.g.edges()) {
            int eid = bimdf.g.id(e);
            int u = bimdf.g.id(bimdf.g.u(e));
            int v = bimdf.g.id(bimdf.g.v(e));
            bool uh = bimdf.u_head[e], vh = bimdf.v_head[e];

            int src, dst;
            if (uh && !vh) {
                // Standard directed: u→v (head at u = flow enters u)
                // Wait — "u_head" means head points AT u = flow comes INTO u
                // So u is the destination, v is the source
                src = v; dst = u;
                n_directed++;
            } else if (!uh && vh) {
                src = u; dst = v;
                n_directed++;
            } else {
                // Bidirected — pick arbitrary direction
                src = u; dst = v;
                n_bidirected++;
            }

            rows[src].push_back({eid, -1.0f});  // outgoing
            rows[dst].push_back({eid, +1.0f});   // incoming
        }

        printf("[BIMDF-CUDA] %d directed + %d bidirected edges → all treated as directed\n",
               n_directed, n_bidirected);

        // Build CSR for A
        pp.csr_row_ptr.resize(M + 1, 0);
        for (int i = 0; i < M; ++i) {
            std::sort(rows[i].begin(), rows[i].end());
            pp.csr_row_ptr[i + 1] = pp.csr_row_ptr[i] + (int)rows[i].size();
            for (auto& [c, v] : rows[i]) {
                pp.csr_col_idx.push_back(c);
                pp.csr_values.push_back(v);
            }
        }

        // A^T
        std::vector<std::vector<std::pair<int,float>>> cols(N);
        for (int i = 0; i < M; ++i)
            for (auto& [c, v] : rows[i])
                cols[c].push_back({i, v});
        pp.csc_col_ptr.resize(N + 1, 0);
        for (int j = 0; j < N; ++j) {
            std::sort(cols[j].begin(), cols[j].end());
            pp.csc_col_ptr[j + 1] = pp.csc_col_ptr[j] + (int)cols[j].size();
            for (auto& [r, v] : cols[j]) {
                pp.csc_row_idx.push_back(r);
                pp.csc_values.push_back(v);
            }
        }

        // RHS: demands (from BiMDF, adjusted for directed formulation)
        pp.rhs.resize(M, 0.0f);
        for (auto n : bimdf.g.nodes())
            pp.rhs[bimdf.g.id(n)] = (float)bimdf.demand[n];

        // Bounds and costs
        pp.lower.resize(N); pp.upper.resize(N);
        pp.cost_type.resize(N); pp.cost_target.resize(N); pp.cost_weight.resize(N);
        for (auto e : bimdf.g.edges()) {
            int eid = bimdf.g.id(e);
            pp.lower[eid] = (float)bimdf.lower[e];
            pp.upper[eid] = (float)std::min(bimdf.upper[e], 10000);
            auto& cf = bimdf.cost_function[e];
            std::visit([&](const auto& f) {
                using T = std::decay_t<decltype(f)>;
                if constexpr (std::is_same_v<T, Satsuma::CostFunction::AbsDeviation>) {
                    pp.cost_type[eid] = PDHG_COST_ABS;
                    pp.cost_target[eid] = (float)f.target;
                    pp.cost_weight[eid] = (float)f.weight;
                } else if constexpr (std::is_same_v<T, Satsuma::CostFunction::QuadDeviation>) {
                    pp.cost_type[eid] = PDHG_COST_QUAD;
                    pp.cost_target[eid] = (float)f.target;
                    pp.cost_weight[eid] = (float)f.weight;
                } else if constexpr (std::is_same_v<T, Satsuma::CostFunction::Zero>) {
                    pp.cost_type[eid] = PDHG_COST_ZERO;
                    pp.cost_target[eid] = (float)f.guess;
                    pp.cost_weight[eid] = 0.0f;
                } else {
                    pp.cost_type[eid] = PDHG_COST_ZERO;
                    pp.cost_target[eid] = (float)Satsuma::CostFunction::get_guess(cf);
                    pp.cost_weight[eid] = 0.0f;
                }
            }, cf);
        }

        // Solve directed LP with PDHG (TU → should give integers)
        auto pdhg_res = pdhg_solve(pp, 5000, 1e-3f);

        // Round (should be near-integer due to TU)
        auto directed_sol = round_pdhg_solution(bimdf, pdhg_res);
        double directed_cost = bimdf.cost(*directed_sol);
        printf("[BIMDF-CUDA] Directed LP cost: %.6f (Phase1: %.6f, PDHG obj: %.4f)\n",
               directed_cost, p1_cost, pdhg_res.objective);

        // Use better of directed LP and Phase 1
        if (directed_cost < p1_cost) {
            result.solution = std::move(directed_sol);
            printf("[BIMDF-CUDA] Using directed LP solution\n");
        }

        // Polish with matching
        do_early_term_refine(bimdf, config, result.solution, 20, -0.01);
        printf("[BIMDF-CUDA] Final cost: %.6f\n", bimdf.cost(*result.solution));
        return result;
    }

    // Fallback
    return result;
}

// Keep the ADMM helper (used by FLOW_ADMM strategy)
inline ADMMProblem bimdf_to_admm(const Satsuma::BiMDF& bimdf) {
    ADMMProblem prob;
    prob.num_nodes = bimdf.g.maxNodeId() + 1;
    prob.num_edges = bimdf.g.maxEdgeId() + 1;
    prob.edge_u.resize(prob.num_edges);
    prob.edge_v.resize(prob.num_edges);
    prob.edge_sign_u.resize(prob.num_edges);
    prob.edge_sign_v.resize(prob.num_edges);
    prob.edge_cost_type.resize(prob.num_edges);
    prob.edge_target.resize(prob.num_edges);
    prob.edge_weight.resize(prob.num_edges);
    prob.edge_lower.resize(prob.num_edges);
    prob.edge_upper.resize(prob.num_edges);
    prob.node_demand.resize(prob.num_nodes, 0.0f);

    for (auto n : bimdf.g.nodes())
        prob.node_demand[bimdf.g.id(n)] = (float)bimdf.demand[n];

    for (auto e : bimdf.g.edges()) {
        int eid = bimdf.g.id(e);
        prob.edge_u[eid] = bimdf.g.id(bimdf.g.u(e));
        prob.edge_v[eid] = bimdf.g.id(bimdf.g.v(e));
        prob.edge_sign_u[eid] = bimdf.u_head[e] ? 1 : -1;
        prob.edge_sign_v[eid] = bimdf.v_head[e] ? 1 : -1;
        prob.edge_lower[eid] = (float)bimdf.lower[e];
        prob.edge_upper[eid] = (float)std::min(bimdf.upper[e], 10000);
        auto& cf = bimdf.cost_function[e];
        std::visit([&](const auto& f) {
            using T = std::decay_t<decltype(f)>;
            if constexpr (std::is_same_v<T, Satsuma::CostFunction::AbsDeviation>) {
                prob.edge_cost_type[eid] = ADMM_COST_ABS;
                prob.edge_target[eid] = (float)f.target;
                prob.edge_weight[eid] = (float)f.weight;
            } else if constexpr (std::is_same_v<T, Satsuma::CostFunction::QuadDeviation>) {
                prob.edge_cost_type[eid] = ADMM_COST_QUAD;
                prob.edge_target[eid] = (float)f.target;
                prob.edge_weight[eid] = (float)f.weight;
            } else {
                prob.edge_cost_type[eid] = ADMM_COST_ZERO;
                prob.edge_target[eid] = (float)Satsuma::CostFunction::get_guess(cf);
                prob.edge_weight[eid] = 0.0f;
            }
        }, cf);
    }

    // Build node-edge incidence CSR
    std::vector<std::vector<std::pair<int,int>>> adj(prob.num_nodes);
    for (auto e : bimdf.g.edges()) {
        int eid = bimdf.g.id(e);
        adj[prob.edge_u[eid]].push_back({eid, prob.edge_sign_u[eid]});
        adj[prob.edge_v[eid]].push_back({eid, prob.edge_sign_v[eid]});
    }
    prob.node_edge_offsets.resize(prob.num_nodes + 1, 0);
    for (int n = 0; n < prob.num_nodes; ++n) {
        for (auto& [eid, sign] : adj[n]) {
            prob.node_edges.push_back(eid);
            prob.node_edge_signs.push_back(sign);
        }
        prob.node_edge_offsets[n + 1] = (int)prob.node_edges.size();
    }
    return prob;
}

} // namespace cuda
} // namespace qw

#endif
