#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct RXMeshRemeshParams {
    float target_edge_length;     // 0 = auto (sqrt(area/10000))
    int   num_iterations;         // outer split+collapse+flip+smooth loops (default: 5)
    int   num_smooth_iters;       // inner smoothing sub-iterations (default: 5)
    float sharp_angle_degrees;    // dihedral angle threshold for feature edges (default: 35)
    int   patch_size;             // RXMesh patch size (default: 256)
    float over_alloc;             // RXMesh over-allocation factor (default: 4.0)
};

// Default parameters matching QuadWild's preprocessing
inline RXMeshRemeshParams rxmesh_remesh_default_params() {
    RXMeshRemeshParams p;
    p.target_edge_length  = 0;      // auto-compute
    p.num_iterations      = 15;     // QuadWild default
    p.num_smooth_iters    = 5;
    p.sharp_angle_degrees = 35.0f;  // QuadWild default
    p.patch_size          = 256;
    p.over_alloc          = 4.0f;
    return p;
}

/// Run GPU isotropic remeshing: split + collapse + flip + smooth.
/// Feature edges (dihedral angle > sharp_angle_degrees) are preserved.
///
/// Input:  V_in[nV_in*3], F_in[nF_in*3] (flat float/uint32 arrays)
/// Output: V_out, F_out allocated with malloc(), caller must free.
///         feature_edges_out[nE_feature*2] = vertex index pairs of feature edges
void rxmesh_remesh(
    const float*    V_in,  uint32_t nV_in,
    const uint32_t* F_in,  uint32_t nF_in,
    const RXMeshRemeshParams* params,
    float**    V_out,  uint32_t* nV_out,
    uint32_t** F_out,  uint32_t* nF_out,
    uint32_t** feature_edges_out, uint32_t* nE_feature);

#ifdef __cplusplus
}
#endif
