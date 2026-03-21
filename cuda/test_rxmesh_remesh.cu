// Standalone test for GPU isotropic remeshing via RXMesh
// Usage: ./test_rxmesh_remesh input.obj [output.obj]

#include "rxmesh_remesh.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

struct SimpleMesh {
    std::vector<float>    V;  // x,y,z interleaved
    std::vector<uint32_t> F;  // i,j,k interleaved
};

SimpleMesh load_obj(const char* path) {
    SimpleMesh m;
    FILE* fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            sscanf(line + 2, "%f %f %f", &x, &y, &z);
            m.V.push_back(x); m.V.push_back(y); m.V.push_back(z);
        } else if (line[0] == 'f' && line[1] == ' ') {
            // Handle f v1 v2 v3 and f v1/vt1 v2/vt2 v3/vt3
            uint32_t a, b, c;
            if (sscanf(line + 2, "%u/%*u/%*u %u/%*u/%*u %u/%*u/%*u", &a, &b, &c) == 3 ||
                sscanf(line + 2, "%u/%*u %u/%*u %u/%*u", &a, &b, &c) == 3 ||
                sscanf(line + 2, "%u %u %u", &a, &b, &c) == 3) {
                m.F.push_back(a-1); m.F.push_back(b-1); m.F.push_back(c-1);
            }
        }
    }
    fclose(fp);
    return m;
}

void save_obj(const char* path, const float* V, uint32_t nV,
              const uint32_t* F, uint32_t nF) {
    FILE* fp = fopen(path, "w");
    for (uint32_t i = 0; i < nV; i++)
        fprintf(fp, "v %.8g %.8g %.8g\n", V[i*3], V[i*3+1], V[i*3+2]);
    for (uint32_t i = 0; i < nF; i++)
        fprintf(fp, "f %u %u %u\n", F[i*3]+1, F[i*3+1]+1, F[i*3+2]+1);
    fclose(fp);
    fprintf(stderr, "Saved %s (%u verts, %u faces)\n", path, nV, nF);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s input.obj [output.obj]\n", argv[0]);
        return 1;
    }

    const char* input_path  = argv[1];
    const char* output_path = argc > 2 ? argv[2] : nullptr;

    fprintf(stderr, "Loading %s...\n", input_path);
    SimpleMesh mesh = load_obj(input_path);
    uint32_t nV = (uint32_t)(mesh.V.size() / 3);
    uint32_t nF = (uint32_t)(mesh.F.size() / 3);
    fprintf(stderr, "Input: %u verts, %u faces\n", nV, nF);

    RXMeshRemeshParams params = rxmesh_remesh_default_params();

    float*    V_out = nullptr;
    uint32_t* F_out = nullptr;
    uint32_t  nV_out = 0, nF_out = 0;
    uint32_t* feat_edges = nullptr;
    uint32_t  nE_feat = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    rxmesh_remesh(
        mesh.V.data(), nV,
        mesh.F.data(), nF,
        &params,
        &V_out, &nV_out,
        &F_out, &nF_out,
        &feat_edges, &nE_feat);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "\n=== GPU Remesh: %.1f ms ===\n", ms);
    fprintf(stderr, "Output: %u verts, %u faces\n", nV_out, nF_out);

    if (output_path && V_out && F_out) {
        save_obj(output_path, V_out, nV_out, F_out, nF_out);
    }

    free(V_out);
    free(F_out);
    free(feat_edges);
    return 0;
}
