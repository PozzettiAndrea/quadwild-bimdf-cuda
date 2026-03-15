#ifndef QW_SMOOTH_KERNELS_H_
#define QW_SMOOTH_KERNELS_H_

// ============================================================
// CUDA smoothing kernels - host interface
//
// Replaces MultiCostraintSmooth() with GPU-accelerated version.
// Call flow:
//   1. Extract flat arrays from VCG meshes (CPU)
//   2. cuda_smooth_init() - upload to GPU
//   3. cuda_smooth() - run N iterations entirely on GPU
//   4. cuda_smooth_download() - download results
//   5. cuda_smooth_destroy() - free GPU memory
//   6. Write positions back to VCG mesh (CPU)
// ============================================================

#include <cstdint>

namespace qw {
namespace cuda {

// Projection type constants (must match smooth_mesh.h enums)
enum ProjType {
    PROJ_NONE = 0,
    PROJ_SURFACE = 1,
    PROJ_SHARP = 2,
    PROJ_CORNER = 3,
};

// Opaque pointers for GPU data (avoid exposing CUDA types in C++ headers)
struct SmoothData {
    void* d_poly_verts = nullptr;
    void* d_poly_verts_tmp = nullptr;
    int num_poly_verts = 0;

    void* d_adj_offsets = nullptr;
    void* d_adj_indices = nullptr;

    void* d_proj_type = nullptr;

    void* d_tri_verts = nullptr;
    void* d_tri_faces = nullptr;
    int num_tri_verts = 0;
    int num_tri_faces = 0;

    void* d_edge_verts = nullptr;
    void* d_sharp_vert_group = nullptr;
    void* d_edge_group_offsets = nullptr;
    void* d_edge_group_indices = nullptr;
    int num_edge_verts = 0;

    // Spatial grid (opaque)
    void* d_grid_cell_start = nullptr;
    void* d_grid_cell_count = nullptr;
    void* d_grid_tri_indices = nullptr;
    float grid_origin[3];
    float grid_cell_size;
    int grid_dim[3];
    int grid_total_entries;
    float search_radius = 0.0f;
};

// Initialize GPU data (upload all mesh data)
void cuda_smooth_init(
    SmoothData& sd,
    const float* h_poly_verts,   // x,y,z interleaved (num_poly_verts * 3)
    int num_poly_verts,
    const int* h_adj_offsets,    // CSR offsets (num_poly_verts + 1)
    const int* h_adj_indices,    // neighbor vertex indices
    int num_adj_entries,
    const int* h_proj_type,      // per-vertex projection type
    const float* h_tri_verts,    // reference mesh x,y,z interleaved
    int num_tri_verts,
    const int* h_tri_faces,      // v0,v1,v2 interleaved
    int num_tri_faces,
    const float* h_edge_verts,   // edge mesh vertex pairs
    int num_edge_verts,
    const int* h_sharp_vert_group,
    const int* h_edge_group_offsets,
    const int* h_edge_group_indices,
    int num_edge_groups,
    int num_edge_group_entries,
    float cell_size);

// Run smoothing iterations on GPU
void cuda_smooth(
    SmoothData& sd,
    int num_iterations,
    int back_proj_steps,
    float damp);

// Download smoothed positions back to host
void cuda_smooth_download(const SmoothData& sd, float* h_poly_verts);

// Free all GPU memory
void cuda_smooth_destroy(SmoothData& sd);

} // namespace cuda
} // namespace qw

#endif // QW_SMOOTH_KERNELS_H_
