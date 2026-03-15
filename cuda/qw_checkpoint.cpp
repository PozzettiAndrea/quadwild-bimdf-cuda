#include "qw_checkpoint.h"
#include "qw_serialize.h"

#include <cstring>
#include <ctime>
#include <sys/stat.h>

namespace qw {

// ============================================================
// Stage name mapping
// ============================================================

static const char* stage_names[] = {
    "post-load",
    "post-chartdata",
    "post-flow",
    "post-quadrangulate",
    "post-smooth",
};

PipelineStage stage_from_name(const char* name) {
    for (int i = 0; i < STAGE_COUNT; ++i) {
        if (strcmp(name, stage_names[i]) == 0) return (PipelineStage)i;
    }
    return STAGE_NONE;
}

const char* stage_name(PipelineStage s) {
    if (s >= 0 && s < STAGE_COUNT) return stage_names[s];
    return "unknown";
}

void list_stages() {
    printf("Pipeline stages:\n");
    for (int i = 0; i < STAGE_COUNT; ++i) {
        printf("  %d: %s\n", i, stage_names[i]);
    }
}

// ============================================================
// Checkpoint file path
// ============================================================

static std::string checkpoint_path(const char* dir, PipelineStage stage) {
    return std::string(dir) + "/" + stage_names[stage] + ".qwc";
}

bool checkpoint_exists(const char* dir, PipelineStage stage) {
    struct stat st;
    std::string path = checkpoint_path(dir, stage);
    return stat(path.c_str(), &st) == 0;
}

// ============================================================
// Save checkpoint
// ============================================================

void save_checkpoint(const CheckpointData& data, PipelineStage stage,
                     const char* dir, const CheckpointHeader& hdr_in) {
    mkdir(dir, 0755);

    std::string path = checkpoint_path(dir, stage);
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        printf("[CHECKPOINT] ERROR: Cannot open %s for writing\n", path.c_str());
        return;
    }

    // Write header
    CheckpointHeader hdr = hdr_in;
    memcpy(hdr.magic, "QWC", 4);
    hdr.version = 1;
    strncpy(hdr.stage, stage_names[stage], sizeof(hdr.stage) - 1);
    hdr.timestamp = (int64_t)time(nullptr);
    fwrite(&hdr, sizeof(hdr), 1, fp);

    // Write stage index
    int32_t stage_idx = (int32_t)stage;
    ser::Save(fp, stage_idx);

    // ---- Always save: input mesh data ----
    ser::Save(fp, data.tri_verts);
    ser::Save(fp, data.tri_faces);
    ser::Save(fp, data.partitions);
    ser::Save(fp, data.corners);
    ser::Save(fp, data.features);
    ser::Save(fp, data.feature_corners);
    ser::Save(fp, data.edge_size);

    // ---- ChartData (available after post-chartdata) ----
    bool has_cd = (stage >= STAGE_POST_CHARTDATA) && data.has_chartdata;
    ser::Save(fp, has_cd);
    if (has_cd) {
        ser::Save(fp, data.chart_num_sides);
        ser::Save(fp, data.chart_labels);
        ser::Save(fp, data.chart_faces);
        ser::Save(fp, data.chart_border_faces);
        ser::Save(fp, data.chart_adjacent);

        // Subsides
        int32_t n_subsides = (int32_t)data.subside_incident_charts.size();
        ser::Save(fp, n_subsides);
        for (int32_t i = 0; i < n_subsides; ++i) {
            ser::Save(fp, data.subside_incident_charts[i]);
            ser::Save(fp, data.subside_incident_chart_subside_id[i]);
            ser::Save(fp, data.subside_incident_chart_side_id[i]);
            ser::Save(fp, data.subside_vertices[i]);
            ser::Save(fp, data.subside_lengths[i]);
            ser::Save(fp, data.subside_sizes[i]);
            { bool tmp = data.subside_is_on_border[i]; ser::Save(fp, tmp); }
        }

        // Chart sides
        int32_t n_charts = (int32_t)data.side_vertices.size();
        ser::Save(fp, n_charts);
        for (int32_t c = 0; c < n_charts; ++c) {
            int32_t n_sides = (int32_t)data.side_vertices[c].size();
            ser::Save(fp, n_sides);
            for (int32_t s = 0; s < n_sides; ++s) {
                ser::Save(fp, data.side_vertices[c][s]);
                ser::Save(fp, data.side_subsides[c][s]);
                // Save bools as ints
                std::vector<int32_t> rev_ints;
                for (bool b : data.side_reversed[c][s]) rev_ints.push_back(b ? 1 : 0);
                ser::Save(fp, rev_ints);
                ser::Save(fp, data.side_lengths[c][s]);
                ser::Save(fp, data.side_sizes[c][s]);
            }
        }

        ser::Save(fp, data.chart_subsides);
        ser::Save(fp, data.label_set);
    }

    // ---- Flow/ILP result (available after post-flow) ----
    bool has_flow = (stage >= STAGE_POST_FLOW) && data.has_flow_result;
    ser::Save(fp, has_flow);
    if (has_flow) {
        ser::Save(fp, data.ilp_result);
        ser::Save(fp, data.flow_gap);
    }

    // ---- Quad mesh (available after post-quadrangulate) ----
    bool has_quad = (stage >= STAGE_POST_QUADRANGULATE) && data.has_quad_mesh;
    ser::Save(fp, has_quad);
    if (has_quad) {
        ser::Save(fp, data.quad_verts);
        ser::Save(fp, data.quad_face_sizes);
        ser::Save(fp, data.quad_face_indices);
        ser::Save(fp, data.quad_partitions);
        ser::Save(fp, data.quad_corners);
    }

    fclose(fp);

    // Print summary
    long file_size = 0;
    struct stat st;
    if (stat(path.c_str(), &st) == 0) file_size = st.st_size;
    printf("[CHECKPOINT] Saved '%s' to %s (%.1f MB)\n",
           stage_names[stage], path.c_str(), file_size / (1024.0 * 1024.0));
    printf("[CHECKPOINT]   input: %s | alpha=%.4f scale=%.2f clusters=%d flow=%d\n",
           hdr.input_mesh, hdr.alpha, hdr.scale_factor,
           hdr.fixed_chart_clusters, hdr.use_flow_solver);
    printf("[CHECKPOINT]   mesh: %d verts, %d faces, %d patches, %d subsides\n",
           hdr.num_vertices, hdr.num_faces, hdr.num_patches, hdr.num_subsides);
}

// ============================================================
// Load checkpoint
// ============================================================

PipelineStage load_checkpoint(CheckpointData& data, const char* dir,
                              PipelineStage stage) {
    std::string path = checkpoint_path(dir, stage);
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        printf("[CHECKPOINT] ERROR: Cannot open %s for reading\n", path.c_str());
        return STAGE_NONE;
    }

    // Read and validate header
    CheckpointHeader hdr;
    fread(&hdr, sizeof(hdr), 1, fp);
    if (memcmp(hdr.magic, "QWC", 4) != 0) {
        printf("[CHECKPOINT] ERROR: Invalid magic in %s (expected QWC)\n", path.c_str());
        fclose(fp);
        return STAGE_NONE;
    }
    if (hdr.version != 1) {
        printf("[CHECKPOINT] ERROR: Unsupported version %d in %s\n", hdr.version, path.c_str());
        fclose(fp);
        return STAGE_NONE;
    }

    int32_t stage_idx;
    ser::Read(fp, stage_idx);
    PipelineStage saved_stage = (PipelineStage)stage_idx;

    printf("[CHECKPOINT] Loading '%s' from %s\n", stage_names[saved_stage], path.c_str());
    printf("[CHECKPOINT]   saved: input=%s alpha=%.4f scale=%.2f flow=%d\n",
           hdr.input_mesh, hdr.alpha, hdr.scale_factor, hdr.use_flow_solver);
    printf("[CHECKPOINT]   mesh: %d verts, %d faces, %d patches\n",
           hdr.num_vertices, hdr.num_faces, hdr.num_patches);

    // ---- Input mesh data ----
    ser::Read(fp, data.tri_verts);
    ser::Read(fp, data.tri_faces);
    ser::Read(fp, data.partitions);
    ser::Read(fp, data.corners);
    ser::Read(fp, data.features);
    ser::Read(fp, data.feature_corners);
    ser::Read(fp, data.edge_size);

    // ---- ChartData ----
    bool has_cd;
    ser::Read(fp, has_cd);
    data.has_chartdata = has_cd;
    if (has_cd) {
        ser::Read(fp, data.chart_num_sides);
        ser::Read(fp, data.chart_labels);
        ser::Read(fp, data.chart_faces);
        ser::Read(fp, data.chart_border_faces);
        ser::Read(fp, data.chart_adjacent);

        int32_t n_subsides;
        ser::Read(fp, n_subsides);
        data.subside_incident_charts.resize(n_subsides);
        data.subside_incident_chart_subside_id.resize(n_subsides);
        data.subside_incident_chart_side_id.resize(n_subsides);
        data.subside_vertices.resize(n_subsides);
        data.subside_lengths.resize(n_subsides);
        data.subside_sizes.resize(n_subsides);
        data.subside_is_on_border.resize(n_subsides);
        for (int32_t i = 0; i < n_subsides; ++i) {
            ser::Read(fp, data.subside_incident_charts[i]);
            ser::Read(fp, data.subside_incident_chart_subside_id[i]);
            ser::Read(fp, data.subside_incident_chart_side_id[i]);
            ser::Read(fp, data.subside_vertices[i]);
            ser::Read(fp, data.subside_lengths[i]);
            ser::Read(fp, data.subside_sizes[i]);
            { bool tmp; ser::Read(fp, tmp); data.subside_is_on_border[i] = tmp; }
        }

        int32_t n_charts;
        ser::Read(fp, n_charts);
        data.side_vertices.resize(n_charts);
        data.side_subsides.resize(n_charts);
        data.side_reversed.resize(n_charts);
        data.side_lengths.resize(n_charts);
        data.side_sizes.resize(n_charts);
        for (int32_t c = 0; c < n_charts; ++c) {
            int32_t n_sides;
            ser::Read(fp, n_sides);
            data.side_vertices[c].resize(n_sides);
            data.side_subsides[c].resize(n_sides);
            data.side_reversed[c].resize(n_sides);
            data.side_lengths[c].resize(n_sides);
            data.side_sizes[c].resize(n_sides);
            for (int32_t s = 0; s < n_sides; ++s) {
                ser::Read(fp, data.side_vertices[c][s]);
                ser::Read(fp, data.side_subsides[c][s]);
                std::vector<int32_t> rev_ints;
                ser::Read(fp, rev_ints);
                data.side_reversed[c][s].clear();
                for (int32_t b : rev_ints) data.side_reversed[c][s].push_back(b != 0);
                ser::Read(fp, data.side_lengths[c][s]);
                ser::Read(fp, data.side_sizes[c][s]);
            }
        }

        ser::Read(fp, data.chart_subsides);
        ser::Read(fp, data.label_set);
    }

    // ---- Flow result ----
    bool has_flow;
    ser::Read(fp, has_flow);
    data.has_flow_result = has_flow;
    if (has_flow) {
        ser::Read(fp, data.ilp_result);
        ser::Read(fp, data.flow_gap);
    }

    // ---- Quad mesh ----
    bool has_quad;
    ser::Read(fp, has_quad);
    data.has_quad_mesh = has_quad;
    if (has_quad) {
        ser::Read(fp, data.quad_verts);
        ser::Read(fp, data.quad_face_sizes);
        ser::Read(fp, data.quad_face_indices);
        ser::Read(fp, data.quad_partitions);
        ser::Read(fp, data.quad_corners);
    }

    fclose(fp);
    return saved_stage;
}

} // namespace qw
