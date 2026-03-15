#ifndef QW_CHART_KERNELS_H_
#define QW_CHART_KERNELS_H_

// ============================================================
// CUDA chart data computation - host interface
//
// GPU-accelerated components of computeChartData():
//   - Face label assignment
//   - Border face detection
//   - Boundary edge extraction
//   - Subside length computation
//   - Chart adjacency building
// ============================================================

#include <vector>
#include <array>

namespace qw {
namespace cuda {

// Compute face labels, border faces, and boundary edges on GPU
void cuda_compute_chart_data(
    const float* h_verts,          // vertex positions (x,y,z interleaved)
    int num_verts,
    const int* h_faces,            // face vertex indices (v0,v1,v2 interleaved)
    const int* h_face_adj,         // face adjacency (3 per face, -1 = boundary)
    int num_faces,
    const std::vector<std::vector<int>>& partitions,
    // Outputs:
    std::vector<int>& face_labels,
    std::vector<int>& border_faces,
    std::vector<std::array<int, 2>>& boundary_edge_patches,
    std::vector<std::array<int, 2>>& boundary_edge_verts);

// Compute subside lengths on GPU (parallel per-subside reduction)
void cuda_compute_subside_lengths(
    const float* h_verts,          // vertex positions
    int num_verts,
    const std::vector<std::vector<int>>& subside_vert_chains,
    std::vector<double>& lengths);

} // namespace cuda
} // namespace qw

#endif // QW_CHART_KERNELS_H_
