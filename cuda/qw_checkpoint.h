#ifndef QW_CHECKPOINT_H_
#define QW_CHECKPOINT_H_

// ============================================================
// Pipeline checkpoint system for QuadWild-BiMDF
//
// Modeled after QuadriFlow-cuda's checkpoint system.
// Allows saving/loading pipeline state at any stage boundary,
// enabling fast benchmarking and resuming without re-running
// the full pipeline.
//
// Usage:
//   Full run with saves:
//     ./quad_from_patches input.obj 0 setup.txt out.json \
//         -save-all -save-dir /tmp/checkpoints
//
//   Resume from stage:
//     ./quad_from_patches input.obj 0 setup.txt out.json \
//         -run-from post-flow -save-dir /tmp/checkpoints
//
//   Run single stage:
//     ./quad_from_patches input.obj 0 setup.txt out.json \
//         -run-from post-chartdata -run-to post-flow \
//         -save-dir /tmp/checkpoints
//
// Stage names (in pipeline order):
//   post-load         After loading mesh + patches + corners + features
//   post-chartdata    After computeChartData()
//   post-flow         After findSubdivisions() (BiMDF solve)
//   post-quadrangulate After quadrangulate() (mesh generation)
//   post-smooth       After MultiCostraintSmooth (final)
// ============================================================

#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <utility>

namespace qw {

// Stage indices (in pipeline order)
enum PipelineStage {
    STAGE_NONE = -1,
    STAGE_POST_LOAD = 0,
    STAGE_POST_CHARTDATA,
    STAGE_POST_FLOW,
    STAGE_POST_QUADRANGULATE,
    STAGE_POST_SMOOTH,
    STAGE_COUNT
};

// Convert stage name string to enum
PipelineStage stage_from_name(const char* name);

// Convert enum to display name
const char* stage_name(PipelineStage s);

// Print all stage names
void list_stages();

// Checkpoint file header (stored at start of each .qwc file)
struct CheckpointHeader {
    char magic[4];              // "QWC\0"
    int32_t version;            // format version (1)
    char stage[64];             // stage name string
    char input_mesh[256];       // input mesh path
    char setup_file[256];       // setup config file path
    float alpha;                // alpha parameter
    float scale_factor;         // scale factor
    int32_t fixed_chart_clusters;
    int32_t use_flow_solver;    // BiMDF vs ILP
    int64_t timestamp;          // unix timestamp
    int32_t num_vertices;       // input mesh stats
    int32_t num_faces;
    int32_t num_patches;
    int32_t num_subsides;
    char reserved[128];         // padding for future use
};

// ============================================================
// Checkpoint save/load for quad_from_patches pipeline
//
// The pipeline state at each stage consists of:
//   - Input mesh (TriangleMesh)
//   - Partitions, corners, features
//   - ChartData (after post-chartdata)
//   - ILP/flow result (after post-flow)
//   - Quad mesh + partitions + corners (after post-quadrangulate)
//   - Smoothed quad mesh (after post-smooth)
//
// Since VCG meshes are complex template types, we serialize
// the data needed to reconstruct them rather than the meshes
// directly:
//   - Vertex positions (float x,y,z)
//   - Face connectivity (vertex indices)
//   - Partition/corner/feature arrays
//   - ChartData structures
//   - ILP result vector
// ============================================================

// Forward declarations - actual serialization is in qw_checkpoint.cpp
// These functions work with raw data vectors extracted from VCG meshes

struct CheckpointData {
    // Input mesh data
    std::vector<float> tri_verts;     // x,y,z interleaved
    std::vector<int32_t> tri_faces;   // v0,v1,v2 interleaved

    // Partitions: vector of face index vectors
    std::vector<std::vector<size_t>> partitions;

    // Corners: vector of vertex index vectors
    std::vector<std::vector<size_t>> corners;

    // Features: pairs of (face_idx, edge_idx)
    std::vector<std::pair<size_t, size_t>> features;

    // Feature corners
    std::vector<size_t> feature_corners;

    // ChartData serialized components (available after post-chartdata)
    // We serialize the ChartData fields individually
    bool has_chartdata = false;
    std::vector<int32_t> chart_num_sides;       // num sides per chart
    std::vector<int32_t> chart_labels;          // label per chart
    std::vector<std::vector<size_t>> chart_faces;
    std::vector<std::vector<size_t>> chart_border_faces;
    std::vector<std::vector<size_t>> chart_adjacent;
    // Subsides
    std::vector<std::array<int32_t, 2>> subside_incident_charts;
    std::vector<std::array<int32_t, 2>> subside_incident_chart_subside_id;
    std::vector<std::array<int32_t, 2>> subside_incident_chart_side_id;
    std::vector<std::vector<size_t>> subside_vertices;
    std::vector<double> subside_lengths;
    std::vector<int32_t> subside_sizes;
    std::vector<bool> subside_is_on_border;
    // Chart sides (nested per chart)
    // For each chart, for each side: vertices, subsides, reversed flags, length, size
    std::vector<std::vector<std::vector<size_t>>> side_vertices;
    std::vector<std::vector<std::vector<size_t>>> side_subsides;
    std::vector<std::vector<std::vector<bool>>> side_reversed;
    std::vector<std::vector<double>> side_lengths;
    std::vector<std::vector<int32_t>> side_sizes;
    // Chart subsides (per chart)
    std::vector<std::vector<size_t>> chart_subsides;
    // Labels set
    std::vector<int32_t> label_set;

    // ILP/flow result (available after post-flow)
    bool has_flow_result = false;
    std::vector<int32_t> ilp_result;
    double flow_gap = 0.0;

    // Quad mesh data (available after post-quadrangulate)
    bool has_quad_mesh = false;
    std::vector<float> quad_verts;
    std::vector<int32_t> quad_face_sizes;  // VN per face
    std::vector<int32_t> quad_face_indices; // concatenated vertex indices
    std::vector<std::vector<size_t>> quad_partitions;
    std::vector<std::vector<size_t>> quad_corners;

    // Edge size (needed for smoothing)
    double edge_size = 0.0;
};

// Save checkpoint data to directory
void save_checkpoint(const CheckpointData& data, PipelineStage stage,
                     const char* dir, const CheckpointHeader& hdr);

// Load checkpoint data from directory
PipelineStage load_checkpoint(CheckpointData& data, const char* dir,
                              PipelineStage stage);

// Check if a checkpoint file exists
bool checkpoint_exists(const char* dir, PipelineStage stage);

} // namespace qw

#endif // QW_CHECKPOINT_H_
