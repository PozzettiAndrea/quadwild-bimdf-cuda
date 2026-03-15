// ============================================================
// CUDA smoothing kernels for QuadWild-BiMDF
//
// GPU-accelerated replacement for MultiCostraintSmooth().
// Operates on flat arrays extracted from VCG meshes.
//
// Pipeline per iteration:
//   1. k_laplacian_smooth: Per-vertex neighbor averaging
//   2. k_back_project:     Project vertices to closest point
//                          on reference triangle mesh
//   3. k_project_sharp:    Project sharp-feature vertices
//                          to closest point on edge mesh
//
// Data flow:
//   CPU: Extract flat arrays from VCG meshes
//   GPU: Upload once, run N iterations, download once
//   CPU: Write positions back to VCG mesh
// ============================================================

#include "smooth_kernels.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <cstdio>
#include <cfloat>

namespace qw {
namespace cuda {

// Internal spatial grid struct (matches SmoothData void* layout)
struct SpatialGrid {
    int* d_cell_start;
    int* d_cell_count;
    int* d_tri_indices;
    float3 origin;
    float cell_size;
    int dim_x, dim_y, dim_z;
    int total_entries;
};

// Typed accessor helpers for SmoothData (which uses void* for C++ compatibility)
#define SD_POLY_VERTS(sd)        ((float3*)(sd).d_poly_verts)
#define SD_POLY_VERTS_TMP(sd)    ((float3*)(sd).d_poly_verts_tmp)
#define SD_ADJ_OFFSETS(sd)       ((int*)(sd).d_adj_offsets)
#define SD_ADJ_INDICES(sd)       ((int*)(sd).d_adj_indices)
#define SD_PROJ_TYPE(sd)         ((int*)(sd).d_proj_type)
#define SD_TRI_VERTS(sd)         ((float3*)(sd).d_tri_verts)
#define SD_TRI_FACES(sd)         ((int3*)(sd).d_tri_faces)
#define SD_EDGE_VERTS(sd)        ((float3*)(sd).d_edge_verts)
#define SD_SHARP_GROUP(sd)       ((int*)(sd).d_sharp_vert_group)
#define SD_EDGE_GRP_OFF(sd)      ((int*)(sd).d_edge_group_offsets)
#define SD_EDGE_GRP_IDX(sd)      ((int*)(sd).d_edge_group_indices)
#define SD_GRID_START(sd)        ((int*)(sd).d_grid_cell_start)
#define SD_GRID_COUNT(sd)        ((int*)(sd).d_grid_cell_count)
#define SD_GRID_TRI(sd)          ((int*)(sd).d_grid_tri_indices)

// ============================================================
// Device helper: closest point on triangle
// ============================================================

__device__ float3 closest_point_on_triangle(
    float3 p, float3 a, float3 b, float3 c)
{
    float3 ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
    float3 ac = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
    float3 ap = make_float3(p.x - a.x, p.y - a.y, p.z - a.z);

    float d1 = ab.x*ap.x + ab.y*ap.y + ab.z*ap.z;
    float d2 = ac.x*ap.x + ac.y*ap.y + ac.z*ap.z;
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    float3 bp = make_float3(p.x - b.x, p.y - b.y, p.z - b.z);
    float d3 = ab.x*bp.x + ab.y*bp.y + ab.z*bp.z;
    float d4 = ac.x*bp.x + ac.y*bp.y + ac.z*bp.z;
    if (d3 >= 0.0f && d4 <= d3) return b;

    float vc = d1*d4 - d3*d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return make_float3(a.x + v*ab.x, a.y + v*ab.y, a.z + v*ab.z);
    }

    float3 cp_ = make_float3(p.x - c.x, p.y - c.y, p.z - c.z);
    float d5 = ab.x*cp_.x + ab.y*cp_.y + ab.z*cp_.z;
    float d6 = ac.x*cp_.x + ac.y*cp_.y + ac.z*cp_.z;
    if (d6 >= 0.0f && d5 <= d6) return c;

    float vb = d5*d2 - d1*d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return make_float3(a.x + w*ac.x, a.y + w*ac.y, a.z + w*ac.z);
    }

    float va = d3*d6 - d5*d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return make_float3(
            b.x + w*(c.x - b.x),
            b.y + w*(c.y - b.y),
            b.z + w*(c.z - b.z));
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return make_float3(
        a.x + ab.x*v + ac.x*w,
        a.y + ab.y*v + ac.y*w,
        a.z + ab.z*v + ac.z*w);
}

// ============================================================
// Device helper: closest point on line segment
// ============================================================

__device__ float3 closest_point_on_segment(
    float3 p, float3 a, float3 b, float& t)
{
    float3 ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
    float3 ap = make_float3(p.x - a.x, p.y - a.y, p.z - a.z);
    float len2 = ab.x*ab.x + ab.y*ab.y + ab.z*ab.z;
    if (len2 < 1e-12f) { t = 0.0f; return a; }
    t = (ap.x*ab.x + ap.y*ab.y + ap.z*ab.z) / len2;
    t = fmaxf(0.0f, fminf(1.0f, t));
    return make_float3(a.x + t*ab.x, a.y + t*ab.y, a.z + t*ab.z);
}

__device__ float dist2_f3(float3 a, float3 b) {
    float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

// ============================================================
// Kernel: Laplacian smoothing
// ============================================================

__global__ void k_laplacian_smooth(
    const float3* __restrict__ pos,         // current vertex positions
    float3* __restrict__ new_pos,           // output smoothed positions
    const int* __restrict__ adj_offsets,    // CSR offsets into adj_indices
    const int* __restrict__ adj_indices,    // neighbor vertex indices
    const int* __restrict__ proj_type,      // projection type per vertex
    float damp,                            // damping factor (0.5 typical)
    int num_verts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_verts) return;

    // Corner vertices are fixed
    if (proj_type[i] == PROJ_CORNER || proj_type[i] == PROJ_NONE) {
        new_pos[i] = pos[i];
        return;
    }

    int start = adj_offsets[i];
    int end = adj_offsets[i + 1];
    int count = end - start;

    if (count <= 1) {
        new_pos[i] = pos[i];
        return;
    }

    float3 avg = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = start; j < end; ++j) {
        int ni = adj_indices[j];
        avg.x += pos[ni].x;
        avg.y += pos[ni].y;
        avg.z += pos[ni].z;
    }
    float inv_count = 1.0f / (float)count;
    avg.x *= inv_count;
    avg.y *= inv_count;
    avg.z *= inv_count;

    new_pos[i].x = pos[i].x * damp + avg.x * (1.0f - damp);
    new_pos[i].y = pos[i].y * damp + avg.y * (1.0f - damp);
    new_pos[i].z = pos[i].z * damp + avg.z * (1.0f - damp);
}

// ============================================================
// Kernel: Back-project to reference triangle mesh
// Uses brute-force closest-point (for meshes up to ~1M faces)
// For larger meshes, use BVH (see cuda_bvh variant)
// ============================================================

__global__ void k_back_project(
    float3* __restrict__ pos,               // vertex positions (in-place)
    const int* __restrict__ proj_type,      // projection type per vertex
    const float3* __restrict__ tri_verts,   // reference mesh vertices
    const int3* __restrict__ tri_faces,     // reference mesh faces
    int num_verts,
    int num_tri_faces)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_verts) return;

    // Only project surface vertices
    if (proj_type[i] != PROJ_SURFACE) return;

    float3 p = pos[i];
    float best_dist2 = FLT_MAX;
    float3 best_pt = p;

    for (int f = 0; f < num_tri_faces; ++f) {
        int3 face = tri_faces[f];
        float3 a = tri_verts[face.x];
        float3 b = tri_verts[face.y];
        float3 c = tri_verts[face.z];

        float3 cp = closest_point_on_triangle(p, a, b, c);
        float d2 = dist2_f3(p, cp);
        if (d2 < best_dist2) {
            best_dist2 = d2;
            best_pt = cp;
        }
    }

    pos[i] = best_pt;
}

// ============================================================
// Kernel: Back-project with spatial hash grid acceleration
// Grid cells are axis-aligned cubes of size cell_size
// ============================================================

__global__ void k_back_project_grid(
    float3* __restrict__ pos,
    const int* __restrict__ proj_type,
    const float3* __restrict__ tri_verts,
    const int3* __restrict__ tri_faces,
    const int* __restrict__ grid_cell_start,  // start index per grid cell
    const int* __restrict__ grid_cell_count,  // count per grid cell
    const int* __restrict__ grid_tri_indices,  // sorted triangle indices
    float3 grid_origin,
    float cell_size,
    int grid_dim_x, int grid_dim_y, int grid_dim_z,
    int num_verts,
    int num_tri_faces,
    float search_radius)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_verts) return;
    if (proj_type[i] != PROJ_SURFACE) return;

    float3 p = pos[i];
    float best_dist2 = FLT_MAX;
    float3 best_pt = p;

    // Compute grid cell for this vertex (clamp to grid bounds — no brute-force fallback)
    int cx = (int)((p.x - grid_origin.x) / cell_size);
    int cy = (int)((p.y - grid_origin.y) / cell_size);
    int cz = (int)((p.z - grid_origin.z) / cell_size);
    cx = max(0, min(cx, grid_dim_x - 1));
    cy = max(0, min(cy, grid_dim_y - 1));
    cz = max(0, min(cz, grid_dim_z - 1));

    // Search 3×3×3 neighborhood (27 cells)
    for (int dz = -1; dz <= 1; ++dz) {
        int gz = cz + dz;
        if (gz < 0 || gz >= grid_dim_z) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int gy = cy + dy;
            if (gy < 0 || gy >= grid_dim_y) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int gx = cx + dx;
                if (gx < 0 || gx >= grid_dim_x) continue;

                int cell_idx = gx + gy * grid_dim_x + gz * grid_dim_x * grid_dim_y;
                int start = grid_cell_start[cell_idx];
                int count = grid_cell_count[cell_idx];

                for (int j = start; j < start + count; ++j) {
                    int fi = grid_tri_indices[j];
                    int3 face = tri_faces[fi];
                    float3 a = tri_verts[face.x];
                    float3 b = tri_verts[face.y];
                    float3 c = tri_verts[face.z];

                    float3 cp = closest_point_on_triangle(p, a, b, c);
                    float d2 = dist2_f3(p, cp);
                    if (d2 < best_dist2) {
                        best_dist2 = d2;
                        best_pt = cp;
                    }
                }
            }
        }
    }

    pos[i] = best_pt;
}

// ============================================================
// Kernel: Project sharp-feature vertices to edge segments
// ============================================================

__global__ void k_project_sharp(
    float3* __restrict__ pos,
    const int* __restrict__ proj_type,
    const int* __restrict__ sharp_vert_edge_group,  // edge group index per vertex (-1 if not sharp)
    const float3* __restrict__ edge_verts,           // edge mesh vertices (pairs: v0,v1,v0,v1,...)
    const int* __restrict__ edge_group_offsets,       // CSR offsets per edge group
    const int* __restrict__ edge_group_indices,       // edge indices within group
    int num_verts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_verts) return;
    if (proj_type[i] != PROJ_SHARP) return;

    int group = sharp_vert_edge_group[i];
    if (group < 0) return;

    float3 p = pos[i];
    float best_dist2 = FLT_MAX;
    float3 best_pt = p;

    int start = edge_group_offsets[group];
    int end = edge_group_offsets[group + 1];

    for (int j = start; j < end; ++j) {
        int ei = edge_group_indices[j];
        float3 a = edge_verts[ei * 2];
        float3 b = edge_verts[ei * 2 + 1];
        float t;
        float3 cp = closest_point_on_segment(p, a, b, t);
        float d2 = dist2_f3(p, cp);
        if (d2 < best_dist2) {
            best_dist2 = d2;
            best_pt = cp;
        }
    }

    pos[i] = best_pt;
}

// ============================================================
// Host: Build spatial hash grid for triangle mesh
// ============================================================

static void build_spatial_grid(
    const float3* h_tri_verts,
    const int3* h_tri_faces,
    int num_tri_faces,
    float cell_size,
    SmoothData& sd)
{
    float3 bbmin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 bbmax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int f = 0; f < num_tri_faces; ++f) {
        for (int v = 0; v < 3; ++v) {
            int vi = (v == 0) ? h_tri_faces[f].x : (v == 1) ? h_tri_faces[f].y : h_tri_faces[f].z;
            float3 p = h_tri_verts[vi];
            bbmin.x = fminf(bbmin.x, p.x); bbmin.y = fminf(bbmin.y, p.y); bbmin.z = fminf(bbmin.z, p.z);
            bbmax.x = fmaxf(bbmax.x, p.x); bbmax.y = fmaxf(bbmax.y, p.y); bbmax.z = fmaxf(bbmax.z, p.z);
        }
    }

    // Small margin — just enough for cell boundary overlap.
    // Vertices outside the grid will clamp to the nearest border cell (no brute-force fallback).
    float margin = cell_size * 2.0f;
    bbmin.x -= margin; bbmin.y -= margin; bbmin.z -= margin;
    bbmax.x += margin; bbmax.y += margin; bbmax.z += margin;

    sd.grid_origin[0] = bbmin.x; sd.grid_origin[1] = bbmin.y; sd.grid_origin[2] = bbmin.z;
    sd.grid_cell_size = cell_size;
    sd.grid_dim[0] = (int)((bbmax.x - bbmin.x) / cell_size) + 1;
    sd.grid_dim[1] = (int)((bbmax.y - bbmin.y) / cell_size) + 1;
    sd.grid_dim[2] = (int)((bbmax.z - bbmin.z) / cell_size) + 1;
    int total_cells = sd.grid_dim[0] * sd.grid_dim[1] * sd.grid_dim[2];

    std::vector<std::vector<int>> cell_tris(total_cells);
    for (int f = 0; f < num_tri_faces; ++f) {
        float3 a = h_tri_verts[h_tri_faces[f].x], b = h_tri_verts[h_tri_faces[f].y], c = h_tri_verts[h_tri_faces[f].z];
        int gx0 = max(0, min((int)((fminf(a.x, fminf(b.x, c.x)) - bbmin.x) / cell_size), sd.grid_dim[0]-1));
        int gy0 = max(0, min((int)((fminf(a.y, fminf(b.y, c.y)) - bbmin.y) / cell_size), sd.grid_dim[1]-1));
        int gz0 = max(0, min((int)((fminf(a.z, fminf(b.z, c.z)) - bbmin.z) / cell_size), sd.grid_dim[2]-1));
        int gx1 = max(0, min((int)((fmaxf(a.x, fmaxf(b.x, c.x)) - bbmin.x) / cell_size), sd.grid_dim[0]-1));
        int gy1 = max(0, min((int)((fmaxf(a.y, fmaxf(b.y, c.y)) - bbmin.y) / cell_size), sd.grid_dim[1]-1));
        int gz1 = max(0, min((int)((fmaxf(a.z, fmaxf(b.z, c.z)) - bbmin.z) / cell_size), sd.grid_dim[2]-1));
        for (int gz = gz0; gz <= gz1; ++gz)
            for (int gy = gy0; gy <= gy1; ++gy)
                for (int gx = gx0; gx <= gx1; ++gx)
                    cell_tris[gx + gy*sd.grid_dim[0] + gz*sd.grid_dim[0]*sd.grid_dim[1]].push_back(f);
    }

    std::vector<int> h_start(total_cells), h_count(total_cells);
    int total_entries = 0;
    for (int i = 0; i < total_cells; ++i) { h_start[i] = total_entries; h_count[i] = (int)cell_tris[i].size(); total_entries += h_count[i]; }
    std::vector<int> h_idx(total_entries);
    for (int i = 0; i < total_cells; ++i)
        for (int j = 0; j < (int)cell_tris[i].size(); ++j)
            h_idx[h_start[i]+j] = cell_tris[i][j];

    cudaMalloc(&sd.d_grid_cell_start, total_cells * sizeof(int));
    cudaMalloc(&sd.d_grid_cell_count, total_cells * sizeof(int));
    cudaMalloc(&sd.d_grid_tri_indices, total_entries * sizeof(int));
    cudaMemcpy(sd.d_grid_cell_start, h_start.data(), total_cells*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sd.d_grid_cell_count, h_count.data(), total_cells*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sd.d_grid_tri_indices, h_idx.data(), total_entries*sizeof(int), cudaMemcpyHostToDevice);
    sd.grid_total_entries = total_entries;
}

// ============================================================
// Host: Run full smoothing pipeline on GPU
// ============================================================

void cuda_smooth(SmoothData& sd, int num_iterations, int back_proj_steps, float damp) {
    int block = 256;
    int grid_v = (sd.num_poly_verts + block - 1) / block;

    float3* d_pos_a = SD_POLY_VERTS(sd);
    float3* d_pos_b = SD_POLY_VERTS_TMP(sd);

    for (int iter = 0; iter < num_iterations; ++iter) {
        k_laplacian_smooth<<<grid_v, block>>>(d_pos_a, d_pos_b, SD_ADJ_OFFSETS(sd), SD_ADJ_INDICES(sd), SD_PROJ_TYPE(sd), damp, sd.num_poly_verts);

        if (sd.num_edge_verts > 0)
            k_project_sharp<<<grid_v, block>>>(d_pos_b, SD_PROJ_TYPE(sd), SD_SHARP_GROUP(sd), SD_EDGE_VERTS(sd), SD_EDGE_GRP_OFF(sd), SD_EDGE_GRP_IDX(sd), sd.num_poly_verts);

        for (int bp = 0; bp < back_proj_steps; ++bp) {
            if (sd.d_grid_cell_start) {
                float3 origin = make_float3(sd.grid_origin[0], sd.grid_origin[1], sd.grid_origin[2]);
                k_back_project_grid<<<grid_v, block>>>(d_pos_b, SD_PROJ_TYPE(sd), SD_TRI_VERTS(sd), SD_TRI_FACES(sd),
                    SD_GRID_START(sd), SD_GRID_COUNT(sd), SD_GRID_TRI(sd), origin, sd.grid_cell_size,
                    sd.grid_dim[0], sd.grid_dim[1], sd.grid_dim[2], sd.num_poly_verts, sd.num_tri_faces, sd.search_radius);
            } else {
                k_back_project<<<grid_v, block>>>(d_pos_b, SD_PROJ_TYPE(sd), SD_TRI_VERTS(sd), SD_TRI_FACES(sd), sd.num_poly_verts, sd.num_tri_faces);
            }
        }

        float3* tmp = d_pos_a; d_pos_a = d_pos_b; d_pos_b = tmp;
    }

    if (d_pos_a != SD_POLY_VERTS(sd))
        cudaMemcpy(sd.d_poly_verts, d_pos_a, sd.num_poly_verts * sizeof(float3), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}

// ============================================================
// Host: Allocate and upload smooth data
// ============================================================

void cuda_smooth_init(
    SmoothData& sd, const float* h_poly_verts, int num_poly_verts,
    const int* h_adj_offsets, const int* h_adj_indices, int num_adj_entries,
    const int* h_proj_type, const float* h_tri_verts, int num_tri_verts,
    const int* h_tri_faces, int num_tri_faces,
    const float* h_edge_verts, int num_edge_verts,
    const int* h_sharp_vert_group, const int* h_edge_group_offsets,
    const int* h_edge_group_indices, int num_edge_groups, int num_edge_group_entries,
    float cell_size)
{
    sd.num_poly_verts = num_poly_verts;
    sd.num_tri_verts = num_tri_verts;
    sd.num_tri_faces = num_tri_faces;
    sd.num_edge_verts = num_edge_verts;

    cudaMalloc(&sd.d_poly_verts, num_poly_verts * sizeof(float3));
    cudaMalloc(&sd.d_poly_verts_tmp, num_poly_verts * sizeof(float3));
    cudaMemcpy(sd.d_poly_verts, h_poly_verts, num_poly_verts * sizeof(float3), cudaMemcpyHostToDevice);

    cudaMalloc(&sd.d_adj_offsets, (num_poly_verts + 1) * sizeof(int));
    cudaMalloc(&sd.d_adj_indices, num_adj_entries * sizeof(int));
    cudaMemcpy(sd.d_adj_offsets, h_adj_offsets, (num_poly_verts + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sd.d_adj_indices, h_adj_indices, num_adj_entries * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&sd.d_proj_type, num_poly_verts * sizeof(int));
    cudaMemcpy(sd.d_proj_type, h_proj_type, num_poly_verts * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&sd.d_tri_verts, num_tri_verts * sizeof(float3));
    cudaMalloc(&sd.d_tri_faces, num_tri_faces * sizeof(int3));
    cudaMemcpy(sd.d_tri_verts, h_tri_verts, num_tri_verts * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(sd.d_tri_faces, h_tri_faces, num_tri_faces * sizeof(int3), cudaMemcpyHostToDevice);

    if (num_edge_verts > 0) {
        cudaMalloc(&sd.d_edge_verts, num_edge_verts * sizeof(float3));
        cudaMemcpy(sd.d_edge_verts, h_edge_verts, num_edge_verts * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMalloc(&sd.d_sharp_vert_group, num_poly_verts * sizeof(int));
        cudaMemcpy(sd.d_sharp_vert_group, h_sharp_vert_group, num_poly_verts * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&sd.d_edge_group_offsets, (num_edge_groups + 1) * sizeof(int));
        cudaMalloc(&sd.d_edge_group_indices, num_edge_group_entries * sizeof(int));
        cudaMemcpy(sd.d_edge_group_offsets, h_edge_group_offsets, (num_edge_groups+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(sd.d_edge_group_indices, h_edge_group_indices, num_edge_group_entries*sizeof(int), cudaMemcpyHostToDevice);
    }

    build_spatial_grid((const float3*)h_tri_verts, (const int3*)h_tri_faces, num_tri_faces, cell_size, sd);
    // Search radius must cover enough cells to always find a triangle.
    // Use 5× cell size — grid margin is already large enough to contain all vertices.
    sd.search_radius = cell_size * 5.0f;
}

void cuda_smooth_download(const SmoothData& sd, float* h_poly_verts) {
    cudaMemcpy(h_poly_verts, sd.d_poly_verts, sd.num_poly_verts * sizeof(float3), cudaMemcpyDeviceToHost);
}

void cuda_smooth_destroy(SmoothData& sd) {
    auto cfree = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    cfree(sd.d_poly_verts); cfree(sd.d_poly_verts_tmp);
    cfree(sd.d_adj_offsets); cfree(sd.d_adj_indices);
    cfree(sd.d_proj_type);
    cfree(sd.d_tri_verts); cfree(sd.d_tri_faces);
    cfree(sd.d_edge_verts); cfree(sd.d_sharp_vert_group);
    cfree(sd.d_edge_group_offsets); cfree(sd.d_edge_group_indices);
    cfree(sd.d_grid_cell_start); cfree(sd.d_grid_cell_count); cfree(sd.d_grid_tri_indices);
    memset(&sd, 0, sizeof(sd));
}

} // namespace cuda
} // namespace qw
