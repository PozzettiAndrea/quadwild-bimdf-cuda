#ifndef QW_VCG_BRIDGE_H_
#define QW_VCG_BRIDGE_H_

// ============================================================
// VCG Mesh <-> Flat Array Bridge
//
// Extracts flat arrays from VCG mesh types for CUDA consumption,
// and writes results back from flat arrays to VCG meshes.
//
// Also handles:
//   - Building CSR vertex adjacency from face connectivity
//   - Extracting face adjacency (FF topology)
//   - Building projection basis data for smoothing
//   - Extracting edge mesh for sharp feature projection
// ============================================================

#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cassert>
#include <vcg/complex/complex.h>

namespace qw {
namespace bridge {

// ============================================================
// Extract flat vertex positions from VCG mesh
// Returns float array [x0,y0,z0, x1,y1,z1, ...]
// ============================================================

template <class MeshType>
void extract_vertex_positions(const MeshType& mesh, std::vector<float>& positions) {
    positions.resize(mesh.vert.size() * 3);
    for (size_t i = 0; i < mesh.vert.size(); ++i) {
        positions[i * 3 + 0] = (float)mesh.vert[i].cP()[0];
        positions[i * 3 + 1] = (float)mesh.vert[i].cP()[1];
        positions[i * 3 + 2] = (float)mesh.vert[i].cP()[2];
    }
}

// ============================================================
// Write flat positions back to VCG mesh
// ============================================================

template <class MeshType>
void write_vertex_positions(MeshType& mesh, const std::vector<float>& positions) {
    assert(positions.size() == mesh.vert.size() * 3);
    for (size_t i = 0; i < mesh.vert.size(); ++i) {
        mesh.vert[i].P()[0] = positions[i * 3 + 0];
        mesh.vert[i].P()[1] = positions[i * 3 + 1];
        mesh.vert[i].P()[2] = positions[i * 3 + 2];
    }
}

// ============================================================
// Extract triangle face indices from VCG TriangleMesh
// Returns int array [v0,v1,v2, v0,v1,v2, ...]
// ============================================================

template <class TriMesh>
void extract_tri_faces(const TriMesh& mesh, std::vector<int>& faces) {
    faces.resize(mesh.face.size() * 3);
    for (size_t i = 0; i < mesh.face.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            faces[i * 3 + j] = (int)(mesh.face[i].cV(j) - &mesh.vert[0]);
        }
    }
}

// ============================================================
// Extract face-face adjacency from VCG mesh (requires FF topology)
// Returns int array [adj0,adj1,adj2, ...] per face, -1 = boundary
// ============================================================

template <class MeshType>
void extract_face_adjacency(const MeshType& mesh, std::vector<int>& face_adj) {
    face_adj.resize(mesh.face.size() * 3);
    for (size_t i = 0; i < mesh.face.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            if (vcg::face::IsBorder(mesh.face[i], j)) {
                face_adj[i * 3 + j] = -1;
            } else {
                face_adj[i * 3 + j] = (int)(mesh.face[i].cFFp(j) - &mesh.face[0]);
            }
        }
    }
}

// ============================================================
// Build CSR vertex adjacency from polygon mesh
// For each vertex, lists its neighbor vertices (connected by edges)
// ============================================================

template <class PolyMesh>
void build_vertex_adjacency_csr(
    const PolyMesh& mesh,
    std::vector<int>& offsets,     // size = num_verts + 1
    std::vector<int>& indices)     // neighbor vertex indices
{
    int nv = (int)mesh.vert.size();

    // Collect all directed edges as (src, dst) pairs
    std::vector<std::pair<int,int>> edges;
    edges.reserve(mesh.face.size() * 8);

    for (size_t i = 0; i < mesh.face.size(); ++i) {
        int vn = mesh.face[i].VN();
        for (int j = 0; j < vn; ++j) {
            int v0 = (int)(mesh.face[i].cV(j) - &mesh.vert[0]);
            int v1 = (int)(mesh.face[i].cV((j + 1) % vn) - &mesh.vert[0]);
            edges.push_back({v0, v1});
            edges.push_back({v1, v0});
        }
    }

    // Sort + unique to deduplicate
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    // Build CSR from sorted edge list
    offsets.assign(nv + 1, 0);
    indices.clear();
    indices.reserve(edges.size());

    for (auto& [src, dst] : edges)
        offsets[src + 1]++;
    for (int i = 0; i < nv; ++i)
        offsets[i + 1] += offsets[i];

    indices.resize(edges.size());
    std::vector<int> pos(offsets.begin(), offsets.end());
    for (auto& [src, dst] : edges)
        indices[pos[src]++] = dst;
}

// ============================================================
// Extract poly mesh face data (variable face sizes)
// Returns: face_sizes[i] = VN for face i
//          face_indices = concatenated vertex indices
// ============================================================

template <class PolyMesh>
void extract_poly_faces(
    const PolyMesh& mesh,
    std::vector<int>& face_sizes,
    std::vector<int>& face_indices)
{
    face_sizes.resize(mesh.face.size());
    face_indices.clear();
    for (size_t i = 0; i < mesh.face.size(); ++i) {
        int vn = mesh.face[i].VN();
        face_sizes[i] = vn;
        for (int j = 0; j < vn; ++j) {
            face_indices.push_back((int)(mesh.face[i].cV(j) - &mesh.vert[0]));
        }
    }
}

// ============================================================
// Build projection type array for smoothing
// Maps smooth_mesh.h ProjType enum to cuda::ProjType
// ============================================================

// ProjType values (must match smooth_mesh.h)
enum ProjType {
    ProjNone = 0,
    ProjSuface = 1,   // note: matches smooth_mesh.h typo "ProjSuface"
    ProjSharp = 2,
    ProjCorner = 3,
};

// Build edge mesh flat arrays for sharp feature projection
// edge_verts: pairs of float3 endpoints [v0_x,v0_y,v0_z, v1_x,v1_y,v1_z, ...]
// sharp_vert_group: per poly vertex, which edge group (-1 = not sharp)
// edge_group_offsets/indices: CSR format grouping of edges
template <class PolyMesh, class TriMesh>
void build_sharp_feature_data(
    const PolyMesh& poly_mesh,
    const TriMesh& tri_mesh,
    const std::vector<std::pair<size_t, size_t>>& features,
    const std::vector<size_t>& tri_face_partition,
    const std::vector<size_t>& poly_face_partition,
    std::vector<float>& edge_verts,
    std::vector<int>& sharp_vert_group,
    std::vector<int>& edge_group_offsets,
    std::vector<int>& edge_group_indices)
{
    // Group feature edges by patch pair
    typedef std::pair<int, int> PatchPairKey;
    std::map<PatchPairKey, int> group_map;
    std::vector<std::vector<std::pair<size_t, size_t>>> grouped_edges;

    for (size_t i = 0; i < features.size(); ++i) {
        size_t fi = features[i].first;
        size_t ei = features[i].second;
        int part0 = (int)tri_face_partition[fi];
        int part1 = -1;
        if (!vcg::face::IsBorder(tri_mesh.face[fi], ei)) {
            size_t adj_fi = vcg::tri::Index(tri_mesh, tri_mesh.face[fi].cFFp(ei));
            part1 = (int)tri_face_partition[adj_fi];
        }
        PatchPairKey key(std::min(part0, part1), std::max(part0, part1));

        size_t v0 = vcg::tri::Index(tri_mesh, tri_mesh.face[fi].cV0(ei));
        size_t v1 = vcg::tri::Index(tri_mesh, tri_mesh.face[fi].cV1(ei));

        if (group_map.find(key) == group_map.end()) {
            group_map[key] = (int)grouped_edges.size();
            grouped_edges.push_back({});
        }
        grouped_edges[group_map[key]].push_back({v0, v1});
    }

    // Build flat edge arrays
    int num_groups = (int)grouped_edges.size();
    edge_group_offsets.resize(num_groups + 1);
    edge_group_indices.clear();
    edge_verts.clear();

    edge_group_offsets[0] = 0;
    int edge_idx = 0;
    for (int g = 0; g < num_groups; ++g) {
        for (auto& e : grouped_edges[g]) {
            // Store edge endpoints
            edge_verts.push_back((float)tri_mesh.vert[e.first].cP()[0]);
            edge_verts.push_back((float)tri_mesh.vert[e.first].cP()[1]);
            edge_verts.push_back((float)tri_mesh.vert[e.first].cP()[2]);
            edge_verts.push_back((float)tri_mesh.vert[e.second].cP()[0]);
            edge_verts.push_back((float)tri_mesh.vert[e.second].cP()[1]);
            edge_verts.push_back((float)tri_mesh.vert[e.second].cP()[2]);
            edge_group_indices.push_back(edge_idx++);
        }
        edge_group_offsets[g + 1] = (int)edge_group_indices.size();
    }

    // Map poly mesh vertices to edge groups
    // For each poly edge on a patch boundary, find the matching group
    sharp_vert_group.resize(poly_mesh.vert.size(), -1);

    for (size_t i = 0; i < poly_mesh.face.size(); ++i) {
        int vn = poly_mesh.face[i].VN();
        int my_part = (int)poly_face_partition[i];

        for (int j = 0; j < vn; ++j) {
            int other_part = -1;
            if (!vcg::face::IsBorder(poly_mesh.face[i], j)) {
                size_t adj_fi = vcg::tri::Index(poly_mesh, poly_mesh.face[i].cFFp(j));
                other_part = (int)poly_face_partition[adj_fi];
            }
            if (other_part != my_part) {
                PatchPairKey key(std::min(my_part, other_part), std::max(my_part, other_part));
                if (group_map.find(key) != group_map.end()) {
                    int group = group_map[key];
                    int v0 = (int)(poly_mesh.face[i].cV(j) - &poly_mesh.vert[0]);
                    int v1 = (int)(poly_mesh.face[i].cV((j + 1) % vn) - &poly_mesh.vert[0]);
                    sharp_vert_group[v0] = group;
                    sharp_vert_group[v1] = group;
                }
            }
        }
    }
}

} // namespace bridge
} // namespace qw

#endif // QW_VCG_BRIDGE_H_
