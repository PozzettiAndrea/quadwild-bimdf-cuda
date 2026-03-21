/*
    rxmesh_remesh.cu — GPU isotropic remeshing using RXMesh.

    SINGLE TRANSLATION UNIT build: all RXMesh sources + remesh kernels compiled
    together. This avoids the Blackwell SM 120 device-linking issue where
    cudaFuncSetAttribute fails on kernel pointers from other TUs.

    Compiled with CUDA_SEPARABLE_COMPILATION OFF.
*/

#include "rxmesh_remesh.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

// ============================================================================
// Include ALL RXMesh source files directly (single TU approach)
// ============================================================================

// CPU sources
#include "rxmesh/rxmesh.cpp"
#include "rxmesh/util/MshLoader.cpp"
#include "rxmesh/util/MshSaver.cpp"

// CUDA sources
#include "rxmesh/attribute.cu"
#include "rxmesh/hash_functions.cu"
#include "rxmesh/lp_hashtable.cu"
#include "rxmesh/patch_info.cu"
#include "rxmesh/patch_lock.cu"
#include "rxmesh/patch_scheduler.cu"
#include "rxmesh/patch_stash.cu"
#include "rxmesh/patcher/patcher.cu"
#include "rxmesh/query.cu"
#include "rxmesh/reduce_handle.cu"
#include "rxmesh/rxmesh_dynamic.cu"
#include "rxmesh/rxmesh_static.cu"

// ============================================================================
// Now include the remesh app kernels (same TU as RXMesh core)
// ============================================================================

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

// RXMesh reference remeshing kernels
#include "../libs/rxmesh/apps/Remesh/util.cuh"
#include "../libs/rxmesh/apps/Remesh/link_condition.cuh"
#include "../libs/rxmesh/apps/Remesh/split.cuh"
#include "../libs/rxmesh/apps/Remesh/collapse.cuh"
#include "../libs/rxmesh/apps/Remesh/flip.cuh"
#include "../libs/rxmesh/apps/Remesh/smoothing.cuh"

using namespace rxmesh;

// ============================================================================
// Main entry point
// ============================================================================
extern "C" void rxmesh_remesh(
    const float*    V_in,  uint32_t nV_in,
    const uint32_t* F_in,  uint32_t nF_in,
    const RXMeshRemeshParams* params,
    float**    V_out,  uint32_t* nV_out,
    uint32_t** F_out,  uint32_t* nF_out,
    uint32_t** feature_edges_out, uint32_t* nE_feature)
{
    static bool log_initialized = false;
    if (!log_initialized) {
        Log::init(spdlog::level::info);
        log_initialized = true;
    }

    RXMeshRemeshParams p;
    if (params) {
        p = *params;
    } else {
        p = rxmesh_remesh_default_params();
    }

    auto t_total = std::chrono::high_resolution_clock::now();
    auto t_stage = t_total;
    auto elapsed = [&]() {
        auto now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - t_stage).count();
        t_stage = now;
        return ms;
    };

    // Write temp OBJ for RXMesh input
    char tmp_path[256];
    snprintf(tmp_path, sizeof(tmp_path), "/tmp/qw_rxmesh_%d.obj", (int)getpid());

    FILE* fp = fopen(tmp_path, "w");
    for (uint32_t v = 0; v < nV_in; v++)
        fprintf(fp, "v %.15g %.15g %.15g\n", V_in[v*3], V_in[v*3+1], V_in[v*3+2]);
    for (uint32_t f = 0; f < nF_in; f++)
        fprintf(fp, "f %u %u %u\n", F_in[f*3]+1, F_in[f*3+1]+1, F_in[f*3+2]+1);
    fclose(fp);
    fprintf(stderr, "[RXMesh] write OBJ: %.1f ms\n", elapsed());

    // Init RXMesh
    fprintf(stderr, "[RXMesh] Initializing (%u V, %u F)...\n", nV_in, nF_in);
    RXMeshDynamic rx(std::string(tmp_path), "", p.patch_size, p.over_alloc, 4);
    fprintf(stderr, "[RXMesh] Init: %.1f ms (%u patches)\n",
            elapsed(), rx.get_num_patches());

    // Allocate attributes
    auto coords      = rx.get_input_vertex_coordinates();
    auto new_coords  = rx.add_vertex_attribute<float>("newCoords", 3);
    new_coords->reset(LOCATION_ALL, 0);
    auto edge_status = rx.add_edge_attribute<EdgeStatus>("EdgeStatus", 1);
    auto v_valence   = rx.add_vertex_attribute<uint8_t>("Valence", 1);
    auto v_boundary  = rx.add_vertex_attribute<bool>("BoundaryV", 1);
    auto edge_link   = rx.add_edge_attribute<int8_t>("edgeLink", 1);

    int* d_buffer;
    CUDA_ERROR(cudaMallocManaged((void**)&d_buffer, sizeof(int)));

    // Skip get_boundary_vertices — it corrupts GPU memory on Blackwell SM 120.
    // The collapse/flip kernels check boundary via diamond query (invalid opposite vertex).
    v_boundary->reset(false, DEVICE);
    fprintf(stderr, "[RXMesh] Attributes allocated: %.1f ms\n", elapsed());

    // Compute stats and target on CPU from input arrays (avoid GPU kernel issues on Blackwell)
    double total_area = 0;
    double total_edge_len = 0;
    int n_edges_cpu = 0;
    // Use a simple edge set to avoid double-counting
    for (uint32_t f = 0; f < nF_in; f++) {
        uint32_t i0 = F_in[f*3], i1 = F_in[f*3+1], i2 = F_in[f*3+2];
        float v0x=V_in[i0*3], v0y=V_in[i0*3+1], v0z=V_in[i0*3+2];
        float v1x=V_in[i1*3], v1y=V_in[i1*3+1], v1z=V_in[i1*3+2];
        float v2x=V_in[i2*3], v2y=V_in[i2*3+1], v2z=V_in[i2*3+2];
        float ax=v1x-v0x, ay=v1y-v0y, az=v1z-v0z;
        float bx=v2x-v0x, by=v2y-v0y, bz=v2z-v0z;
        float cx=ay*bz-az*by, cy=az*bx-ax*bz, cz=ax*by-ay*bx;
        total_area += 0.5 * sqrt(cx*cx+cy*cy+cz*cz);
        // Sum all 3 edge lengths (each edge counted ~2x, ok for average)
        total_edge_len += sqrt(ax*ax+ay*ay+az*az);
        float ex=v2x-v1x, ey=v2y-v1y, ez=v2z-v1z;
        total_edge_len += sqrt(ex*ex+ey*ey+ez*ez);
        total_edge_len += sqrt(bx*bx+by*by+bz*bz);
        n_edges_cpu += 3;
    }
    float avg_edge_len = (float)(total_edge_len / n_edges_cpu);

    float final_target_len = p.target_edge_length;
    if (final_target_len <= 0) {
        const size_t MinFaces = 10000;
        final_target_len = sqrtf((float)(total_area * 2.309 / MinFaces));
    }

    float target_len = final_target_len;
    bool needs_ramp = (final_target_len > avg_edge_len * 1.5f);
    if (needs_ramp) {
        target_len = avg_edge_len;
    }

    fprintf(stderr, "[RXMesh] area=%.6g avg_edge=%.6f final_target=%.6f (ratio=%.1fx) ramping=%s\n",
            total_area, avg_edge_len, final_target_len,
            final_target_len / avg_edge_len, needs_ramp ? "yes" : "no");
    fprintf(stderr, "[RXMesh] Stats: %.1f ms\n", elapsed());

    // Check CUDA state — RXMesh init may have left a sticky error on Blackwell
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "[RXMesh] WARN: sticky CUDA error after init: %s\n",
                    cudaGetErrorString(err));
            fprintf(stderr, "[RXMesh] RXMesh init has a Blackwell SM 120 compatibility bug.\n");
            fprintf(stderr, "[RXMesh] Cannot proceed with GPU remeshing. Falling back.\n");
            // Clean up and return empty — caller should fall back to CPU
            remove(tmp_path);
            cudaFree(d_buffer);
            *V_out = nullptr; *nV_out = 0;
            *F_out = nullptr; *nF_out = 0;
            *feature_edges_out = nullptr; *nE_feature = 0;
            return;
        }
    }

    Timers<GPUTimer> timers;
    timers.add("Total");
    timers.add("SplitTotal"); timers.add("Split");
    timers.add("SplitCleanup"); timers.add("SplitSlice");
    timers.add("CollapseTotal"); timers.add("Collapse");
    timers.add("CollapseCleanup"); timers.add("CollapseSlice");
    timers.add("FlipTotal"); timers.add("Flip");
    timers.add("FlipCleanup"); timers.add("FlipSlice");
    timers.add("SmoothTotal");

    timers.start("Total");
    for (int iter = 0; iter < p.num_iterations; ++iter) {

        // Ramp target length toward final if needed
        if (needs_ramp && target_len < final_target_len) {
            target_len = std::min(target_len * 1.5f, final_target_len);
        }
        float low_len  = (4.0f / 5.0f) * target_len;
        float high_len = (4.0f / 3.0f) * target_len;
        float low_len_sq  = low_len * low_len;
        float high_len_sq = high_len * high_len;

        uint32_t v_before = rx.get_num_vertices(true);
        uint32_t f_before = rx.get_num_faces(true);
        auto t_iter = std::chrono::high_resolution_clock::now();

        auto t_op = std::chrono::high_resolution_clock::now();
        split_long_edges(rx, coords.get(), edge_status.get(), v_boundary.get(),
                         high_len_sq, low_len_sq, timers, d_buffer);
        double t_split = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_op).count();

        t_op = std::chrono::high_resolution_clock::now();
        collapse_short_edges(rx, coords.get(), edge_status.get(), edge_link.get(),
                             v_boundary.get(), low_len_sq, high_len_sq, timers, d_buffer);
        double t_collapse = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_op).count();

        t_op = std::chrono::high_resolution_clock::now();
        equalize_valences(rx, coords.get(), v_valence.get(), edge_status.get(),
                          edge_link.get(), v_boundary.get(), timers, d_buffer);
        double t_flip = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_op).count();

        t_op = std::chrono::high_resolution_clock::now();
        tangential_relaxation(rx, coords.get(), new_coords.get(), v_boundary.get(),
                              p.num_smooth_iters, timers);
        std::swap(new_coords, coords);
        double t_smooth = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_op).count();

        uint32_t v_after = rx.get_num_vertices(true);
        uint32_t f_after = rx.get_num_faces(true);
        double iter_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_iter).count();

        fprintf(stderr, "[RXMesh] iter %d: %.0fms (split=%.0f col=%.0f flip=%.0f smooth=%.0f) "
                "%uV %uF -> %uV %uF (target=%.6f)\n",
                iter, iter_ms, t_split, t_collapse, t_flip, t_smooth,
                v_before, f_before, v_after, f_after, target_len);

        rx.get_boundary_vertices(*v_boundary);

        if (v_after == v_before && f_after == f_before) {
            fprintf(stderr, "[RXMesh] Stabilized at iter %d\n", iter);
            break;
        }
    }
    timers.stop("Total");
    CUDA_ERROR(cudaDeviceSynchronize());

    double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_total).count();

    // Export
    char tmp_out[256];
    snprintf(tmp_out, sizeof(tmp_out), "/tmp/qw_rxmesh_out_%d.obj", (int)getpid());
    rx.update_host();
    coords->move(DEVICE, HOST);
    rx.export_obj(tmp_out, *coords);

    fp = fopen(tmp_out, "r");
    std::vector<float> verts;
    std::vector<uint32_t> faces;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            sscanf(line + 2, "%f %f %f", &x, &y, &z);
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
        } else if (line[0] == 'f' && line[1] == ' ') {
            uint32_t a, b, c;
            sscanf(line + 2, "%u %u %u", &a, &b, &c);
            faces.push_back(a-1); faces.push_back(b-1); faces.push_back(c-1);
        }
    }
    fclose(fp);

    *nV_out = (uint32_t)(verts.size() / 3);
    *nF_out = (uint32_t)(faces.size() / 3);
    *V_out = (float*)malloc(verts.size() * sizeof(float));
    *F_out = (uint32_t*)malloc(faces.size() * sizeof(uint32_t));
    memcpy(*V_out, verts.data(), verts.size() * sizeof(float));
    memcpy(*F_out, faces.data(), faces.size() * sizeof(uint32_t));

    *feature_edges_out = nullptr;
    *nE_feature = 0;

    remove(tmp_path);
    remove(tmp_out);
    cudaFree(d_buffer);

    fprintf(stderr, "[RXMesh] Done: %u V, %u F (%.1f ms total)\n",
            *nV_out, *nF_out, total_ms);
}
