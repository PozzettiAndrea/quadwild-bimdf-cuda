#ifndef QW_SERIALIZE_H_
#define QW_SERIALIZE_H_

// ============================================================
// Binary serialization utilities for QuadWild checkpoint system
//
// Modeled after QuadriFlow-cuda's serialize.hpp.
// Supports: scalars, std::vector, std::set, std::array,
//           std::pair, std::map, std::string, Eigen matrices.
// ============================================================

#include <cstdio>
#include <cstdint>
#include <vector>
#include <set>
#include <map>
#include <array>
#include <string>

namespace qw {
namespace ser {

// ---- Scalars ----

inline void Save(FILE* fp, int32_t v) { fwrite(&v, sizeof(v), 1, fp); }
inline void Read(FILE* fp, int32_t& v) { fread(&v, sizeof(v), 1, fp); }

inline void Save(FILE* fp, uint32_t v) { fwrite(&v, sizeof(v), 1, fp); }
inline void Read(FILE* fp, uint32_t& v) { fread(&v, sizeof(v), 1, fp); }

inline void Save(FILE* fp, int64_t v) { fwrite(&v, sizeof(v), 1, fp); }
inline void Read(FILE* fp, int64_t& v) { fread(&v, sizeof(v), 1, fp); }

inline void Save(FILE* fp, uint64_t v) { fwrite(&v, sizeof(v), 1, fp); }
inline void Read(FILE* fp, uint64_t& v) { fread(&v, sizeof(v), 1, fp); }

inline void Save(FILE* fp, float v) { fwrite(&v, sizeof(v), 1, fp); }
inline void Read(FILE* fp, float& v) { fread(&v, sizeof(v), 1, fp); }

inline void Save(FILE* fp, double v) { fwrite(&v, sizeof(v), 1, fp); }
inline void Read(FILE* fp, double& v) { fread(&v, sizeof(v), 1, fp); }

inline void Save(FILE* fp, bool v) { int32_t i = v ? 1 : 0; Save(fp, i); }
inline void Read(FILE* fp, bool& v) { int32_t i; Read(fp, i); v = (i != 0); }

// ---- std::string ----

inline void Save(FILE* fp, const std::string& s) {
    int32_t n = (int32_t)s.size();
    Save(fp, n);
    if (n > 0) fwrite(s.data(), 1, n, fp);
}

inline void Read(FILE* fp, std::string& s) {
    int32_t n;
    Read(fp, n);
    s.resize(n);
    if (n > 0) fread(s.data(), 1, n, fp);
}

// ---- std::pair ----

template <class A, class B>
inline void Save(FILE* fp, const std::pair<A, B>& p) {
    Save(fp, p.first);
    Save(fp, p.second);
}

template <class A, class B>
inline void Read(FILE* fp, std::pair<A, B>& p) {
    Read(fp, p.first);
    Read(fp, p.second);
}

// ---- std::array ----

template <class T, size_t N>
inline void Save(FILE* fp, const std::array<T, N>& a) {
    for (size_t i = 0; i < N; ++i) Save(fp, a[i]);
}

template <class T, size_t N>
inline void Read(FILE* fp, std::array<T, N>& a) {
    for (size_t i = 0; i < N; ++i) Read(fp, a[i]);
}

// ---- std::vector ----

// Fast path for POD vectors (int, float, double)
inline void Save(FILE* fp, const std::vector<int32_t>& v) {
    int32_t n = (int32_t)v.size();
    Save(fp, n);
    if (n > 0) fwrite(v.data(), sizeof(int32_t), n, fp);
}

inline void Read(FILE* fp, std::vector<int32_t>& v) {
    int32_t n;
    Read(fp, n);
    v.resize(n);
    if (n > 0) fread(v.data(), sizeof(int32_t), n, fp);
}

inline void Save(FILE* fp, const std::vector<float>& v) {
    int32_t n = (int32_t)v.size();
    Save(fp, n);
    if (n > 0) fwrite(v.data(), sizeof(float), n, fp);
}

inline void Read(FILE* fp, std::vector<float>& v) {
    int32_t n;
    Read(fp, n);
    v.resize(n);
    if (n > 0) fread(v.data(), sizeof(float), n, fp);
}

inline void Save(FILE* fp, const std::vector<double>& v) {
    int32_t n = (int32_t)v.size();
    Save(fp, n);
    if (n > 0) fwrite(v.data(), sizeof(double), n, fp);
}

inline void Read(FILE* fp, std::vector<double>& v) {
    int32_t n;
    Read(fp, n);
    v.resize(n);
    if (n > 0) fread(v.data(), sizeof(double), n, fp);
}

// Generic vector (for nested vectors, pairs, etc.)
template <class T>
void Save(FILE* fp, const std::vector<T>& v) {
    int32_t n = (int32_t)v.size();
    Save(fp, n);
    for (auto& x : v) Save(fp, x);
}

template <class T>
void Read(FILE* fp, std::vector<T>& v) {
    int32_t n;
    Read(fp, n);
    v.resize(n);
    for (auto& x : v) Read(fp, x);
}

// size_t vectors (convert to/from int64_t for portability)
inline void Save(FILE* fp, const std::vector<size_t>& v) {
    int32_t n = (int32_t)v.size();
    Save(fp, n);
    for (auto& x : v) {
        int64_t val = (int64_t)x;
        Save(fp, val);
    }
}

inline void Read(FILE* fp, std::vector<size_t>& v) {
    int32_t n;
    Read(fp, n);
    v.resize(n);
    for (auto& x : v) {
        int64_t val;
        Read(fp, val);
        x = (size_t)val;
    }
}

// ---- std::set ----

template <class T>
void Save(FILE* fp, const std::set<T>& s) {
    std::vector<T> buf(s.begin(), s.end());
    Save(fp, buf);
}

template <class T>
void Read(FILE* fp, std::set<T>& s) {
    std::vector<T> buf;
    Read(fp, buf);
    s.clear();
    s.insert(buf.begin(), buf.end());
}

// ---- std::map ----

template <class K, class V>
void Save(FILE* fp, const std::map<K, V>& m) {
    int32_t n = (int32_t)m.size();
    Save(fp, n);
    for (auto& kv : m) {
        Save(fp, kv.first);
        Save(fp, kv.second);
    }
}

template <class K, class V>
void Read(FILE* fp, std::map<K, V>& m) {
    int32_t n;
    Read(fp, n);
    m.clear();
    for (int32_t i = 0; i < n; ++i) {
        K key;
        V val;
        Read(fp, key);
        Read(fp, val);
        m[key] = val;
    }
}

// ---- VCG mesh serialization helpers ----
// These work with any VCG mesh type by extracting/restoring vertex and face data

// Save VCG TriangleMesh (vertices + faces + topology)
template <class TriMesh>
void SaveTriMesh(FILE* fp, const TriMesh& mesh) {
    int32_t nv = (int32_t)mesh.vert.size();
    int32_t nf = (int32_t)mesh.face.size();
    Save(fp, nv);
    Save(fp, nf);

    // Save vertex positions
    for (int32_t i = 0; i < nv; ++i) {
        float p[3] = {(float)mesh.vert[i].cP()[0],
                      (float)mesh.vert[i].cP()[1],
                      (float)mesh.vert[i].cP()[2]};
        fwrite(p, sizeof(float), 3, fp);
    }

    // Save face indices
    for (int32_t i = 0; i < nf; ++i) {
        for (int j = 0; j < mesh.face[i].VN(); ++j) {
            int32_t vi = (int32_t)(mesh.face[i].cV(j) - &mesh.vert[0]);
            Save(fp, vi);
        }
    }
}

// Save VCG PolyMesh (variable face size)
template <class PolyMesh>
void SavePolyMesh(FILE* fp, const PolyMesh& mesh) {
    int32_t nv = (int32_t)mesh.vert.size();
    int32_t nf = (int32_t)mesh.face.size();
    Save(fp, nv);
    Save(fp, nf);

    // Save vertex positions
    for (int32_t i = 0; i < nv; ++i) {
        float p[3] = {(float)mesh.vert[i].cP()[0],
                      (float)mesh.vert[i].cP()[1],
                      (float)mesh.vert[i].cP()[2]};
        fwrite(p, sizeof(float), 3, fp);
    }

    // Save face vertex counts and indices
    for (int32_t i = 0; i < nf; ++i) {
        int32_t vn = mesh.face[i].VN();
        Save(fp, vn);
        for (int j = 0; j < vn; ++j) {
            int32_t vi = (int32_t)(mesh.face[i].cV(j) - &mesh.vert[0]);
            Save(fp, vi);
        }
    }
}

} // namespace ser
} // namespace qw

#endif // QW_SERIALIZE_H_
