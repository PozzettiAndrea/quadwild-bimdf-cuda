// ============================================================
// CUDA kernels for chart data computation
//
// GPU-accelerated components of computeChartData():
//   1. Face-to-chart label assignment
//   2. Subside length computation (parallel reduction)
//   3. Chart adjacency matrix construction
//   4. Border face detection
//
// These kernels operate on flat arrays and produce results
// that are assembled into ChartData on the CPU side.
// ============================================================

#include "chart_kernels.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <cstdio>

namespace qw {
namespace cuda {

// ============================================================
// Kernel: Compute face-to-partition mapping
// Given partitions (list of face indices per patch), builds
// a per-face label array.
// ============================================================

__global__ void k_assign_face_labels(
    int* __restrict__ face_labels,     // output: label per face
    const int* __restrict__ partition_offsets,  // CSR offsets per partition
    const int* __restrict__ partition_faces,    // face indices (flattened)
    int num_partitions)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int start = partition_offsets[pid];
    int end = partition_offsets[pid + 1];

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        face_labels[partition_faces[i]] = pid;
    }
}

// ============================================================
// Kernel: Detect border faces (faces with edges on patch boundary)
// A face is a border face if any of its edges connects to a
// face in a different partition.
// ============================================================

__global__ void k_detect_border_faces(
    const int* __restrict__ face_labels,
    const int* __restrict__ face_adj,      // 3 adjacent face indices per face (-1 = boundary)
    int* __restrict__ is_border,           // output: 1 if border face
    int num_faces)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_faces) return;

    int my_label = face_labels[i];
    int border = 0;

    for (int e = 0; e < 3; ++e) {
        int adj = face_adj[i * 3 + e];
        if (adj < 0 || face_labels[adj] != my_label) {
            border = 1;
            break;
        }
    }

    is_border[i] = border;
}

// ============================================================
// Kernel: Extract patch boundary edges
// For each face edge, if the adjacent face is in a different
// partition, emit a boundary edge record.
// ============================================================

__global__ void k_extract_boundary_edges(
    const int* __restrict__ face_labels,
    const int* __restrict__ face_adj,         // 3 adjacent faces per face
    const int* __restrict__ face_verts,       // 3 vertex indices per face
    int* __restrict__ edge_v0,                // output edge vertex 0
    int* __restrict__ edge_v1,                // output edge vertex 1
    int* __restrict__ edge_patch_left,        // patch on left side
    int* __restrict__ edge_patch_right,       // patch on right side (-1 = boundary)
    int* __restrict__ edge_count,             // atomic counter
    int num_faces)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_faces) return;

    int my_label = face_labels[i];

    for (int e = 0; e < 3; ++e) {
        int adj = face_adj[i * 3 + e];

        bool is_boundary_edge = false;
        int other_label = -1;

        if (adj < 0) {
            is_boundary_edge = true;
            other_label = -1;
        } else {
            other_label = face_labels[adj];
            if (other_label != my_label) {
                // Only emit from the face with lower index to avoid duplicates
                if (i < adj) {
                    is_boundary_edge = true;
                }
            }
        }

        if (is_boundary_edge) {
            int idx = atomicAdd(edge_count, 1);
            int v0 = face_verts[i * 3 + e];
            int v1 = face_verts[i * 3 + ((e + 1) % 3)];
            edge_v0[idx] = v0;
            edge_v1[idx] = v1;
            edge_patch_left[idx] = my_label;
            edge_patch_right[idx] = other_label;
        }
    }
}

// ============================================================
// Kernel: Compute subside lengths (parallel reduction)
// Each subside is a chain of edges. Compute total length.
// ============================================================

__global__ void k_compute_subside_lengths(
    const float3* __restrict__ verts,
    const int* __restrict__ subside_offsets,    // CSR offsets per subside
    const int* __restrict__ subside_vert_indices, // vertex indices per subside
    double* __restrict__ lengths,               // output: length per subside
    int num_subsides)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_subsides) return;

    int start = subside_offsets[sid];
    int end = subside_offsets[sid + 1];
    int n_verts = end - start;

    double total_len = 0.0;
    for (int i = 0; i < n_verts - 1; ++i) {
        float3 a = verts[subside_vert_indices[start + i]];
        float3 b = verts[subside_vert_indices[start + i + 1]];
        double dx = (double)(b.x - a.x);
        double dy = (double)(b.y - a.y);
        double dz = (double)(b.z - a.z);
        total_len += sqrt(dx*dx + dy*dy + dz*dz);
    }

    lengths[sid] = total_len;
}

// ============================================================
// Kernel: Build chart adjacency from boundary edges
// For each boundary edge, record (patch_left, patch_right) pair
// ============================================================

__global__ void k_build_adjacency_pairs(
    const int* __restrict__ edge_patch_left,
    const int* __restrict__ edge_patch_right,
    int2* __restrict__ adj_pairs,    // output: (min, max) patch pairs
    int num_boundary_edges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boundary_edges) return;

    int p0 = edge_patch_left[i];
    int p1 = edge_patch_right[i];

    if (p1 >= 0) {
        adj_pairs[i] = make_int2(min(p0, p1), max(p0, p1));
    } else {
        adj_pairs[i] = make_int2(-1, -1);  // boundary, ignore
    }
}

// ============================================================
// Host: Run chart computation pipeline on GPU
// ============================================================

void cuda_compute_chart_data(
    const float* h_verts,       // vertex positions (x,y,z interleaved)
    int num_verts,
    const int* h_faces,         // face vertex indices (v0,v1,v2 interleaved)
    const int* h_face_adj,      // face adjacency (3 per face, -1 = boundary)
    int num_faces,
    const std::vector<std::vector<int>>& partitions,  // face indices per partition
    // Outputs:
    std::vector<int>& face_labels,
    std::vector<int>& border_faces,
    std::vector<std::array<int, 2>>& boundary_edge_patches,
    std::vector<std::array<int, 2>>& boundary_edge_verts)
{
    int num_partitions = (int)partitions.size();

    // Build flattened partition arrays
    std::vector<int> h_part_offsets(num_partitions + 1, 0);
    std::vector<int> h_part_faces;
    for (int p = 0; p < num_partitions; ++p) {
        h_part_offsets[p + 1] = h_part_offsets[p] + (int)partitions[p].size();
        for (int f : partitions[p]) h_part_faces.push_back(f);
    }

    // Upload to GPU
    int *d_face_labels, *d_part_offsets, *d_part_faces;
    int *d_face_adj, *d_face_verts, *d_is_border;
    float3* d_verts;

    cudaMalloc(&d_face_labels, num_faces * sizeof(int));
    cudaMalloc(&d_part_offsets, (num_partitions + 1) * sizeof(int));
    cudaMalloc(&d_part_faces, h_part_faces.size() * sizeof(int));
    cudaMalloc(&d_face_adj, num_faces * 3 * sizeof(int));
    cudaMalloc(&d_face_verts, num_faces * 3 * sizeof(int));
    cudaMalloc(&d_is_border, num_faces * sizeof(int));
    cudaMalloc(&d_verts, num_verts * sizeof(float3));

    cudaMemset(d_face_labels, 0xff, num_faces * sizeof(int));
    cudaMemcpy(d_part_offsets, h_part_offsets.data(), (num_partitions + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_part_faces, h_part_faces.data(), h_part_faces.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_face_adj, h_face_adj, num_faces * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_face_verts, h_faces, num_faces * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_verts, h_verts, num_verts * sizeof(float3), cudaMemcpyHostToDevice);

    // Step 1: Assign face labels
    k_assign_face_labels<<<num_partitions, 256>>>(
        d_face_labels, d_part_offsets, d_part_faces, num_partitions);

    // Step 2: Detect border faces
    int block = 256;
    int grid = (num_faces + block - 1) / block;
    cudaMemset(d_is_border, 0, num_faces * sizeof(int));
    k_detect_border_faces<<<grid, block>>>(
        d_face_labels, d_face_adj, d_is_border, num_faces);

    // Step 3: Extract boundary edges
    int max_boundary = num_faces * 3;
    int *d_edge_v0, *d_edge_v1, *d_edge_pl, *d_edge_pr, *d_edge_count;
    cudaMalloc(&d_edge_v0, max_boundary * sizeof(int));
    cudaMalloc(&d_edge_v1, max_boundary * sizeof(int));
    cudaMalloc(&d_edge_pl, max_boundary * sizeof(int));
    cudaMalloc(&d_edge_pr, max_boundary * sizeof(int));
    cudaMalloc(&d_edge_count, sizeof(int));
    cudaMemset(d_edge_count, 0, sizeof(int));

    k_extract_boundary_edges<<<grid, block>>>(
        d_face_labels, d_face_adj, d_face_verts,
        d_edge_v0, d_edge_v1, d_edge_pl, d_edge_pr,
        d_edge_count, num_faces);

    // Download results
    int h_edge_count;
    cudaMemcpy(&h_edge_count, d_edge_count, sizeof(int), cudaMemcpyDeviceToHost);

    face_labels.resize(num_faces);
    cudaMemcpy(face_labels.data(), d_face_labels, num_faces * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> h_is_border(num_faces);
    cudaMemcpy(h_is_border.data(), d_is_border, num_faces * sizeof(int), cudaMemcpyDeviceToHost);
    border_faces.clear();
    for (int i = 0; i < num_faces; ++i) {
        if (h_is_border[i]) border_faces.push_back(i);
    }

    std::vector<int> h_ev0(h_edge_count), h_ev1(h_edge_count);
    std::vector<int> h_epl(h_edge_count), h_epr(h_edge_count);
    cudaMemcpy(h_ev0.data(), d_edge_v0, h_edge_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ev1.data(), d_edge_v1, h_edge_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_epl.data(), d_edge_pl, h_edge_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_epr.data(), d_edge_pr, h_edge_count * sizeof(int), cudaMemcpyDeviceToHost);

    boundary_edge_patches.resize(h_edge_count);
    boundary_edge_verts.resize(h_edge_count);
    for (int i = 0; i < h_edge_count; ++i) {
        boundary_edge_patches[i] = {h_epl[i], h_epr[i]};
        boundary_edge_verts[i] = {h_ev0[i], h_ev1[i]};
    }

    // Cleanup
    cudaFree(d_face_labels);
    cudaFree(d_part_offsets);
    cudaFree(d_part_faces);
    cudaFree(d_face_adj);
    cudaFree(d_face_verts);
    cudaFree(d_is_border);
    cudaFree(d_verts);
    cudaFree(d_edge_v0);
    cudaFree(d_edge_v1);
    cudaFree(d_edge_pl);
    cudaFree(d_edge_pr);
    cudaFree(d_edge_count);
}

// ============================================================
// Host: Compute subside lengths on GPU
// ============================================================

void cuda_compute_subside_lengths(
    const float* h_verts,
    int num_verts,
    const std::vector<std::vector<int>>& subside_vert_chains,
    std::vector<double>& lengths)
{
    int num_subsides = (int)subside_vert_chains.size();
    if (num_subsides == 0) return;

    // Build flattened arrays
    std::vector<int> h_offsets(num_subsides + 1, 0);
    std::vector<int> h_indices;
    for (int s = 0; s < num_subsides; ++s) {
        h_offsets[s + 1] = h_offsets[s] + (int)subside_vert_chains[s].size();
        for (int vi : subside_vert_chains[s]) h_indices.push_back(vi);
    }

    // Upload
    float3* d_verts;
    int *d_offsets, *d_indices;
    double* d_lengths;

    cudaMalloc(&d_verts, num_verts * sizeof(float3));
    cudaMalloc(&d_offsets, (num_subsides + 1) * sizeof(int));
    cudaMalloc(&d_indices, h_indices.size() * sizeof(int));
    cudaMalloc(&d_lengths, num_subsides * sizeof(double));

    cudaMemcpy(d_verts, h_verts, num_verts * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets.data(), (num_subsides + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices.data(), h_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (num_subsides + block - 1) / block;
    k_compute_subside_lengths<<<grid, block>>>(
        d_verts, d_offsets, d_indices, d_lengths, num_subsides);

    lengths.resize(num_subsides);
    cudaMemcpy(lengths.data(), d_lengths, num_subsides * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_verts);
    cudaFree(d_offsets);
    cudaFree(d_indices);
    cudaFree(d_lengths);
}

} // namespace cuda
} // namespace qw
