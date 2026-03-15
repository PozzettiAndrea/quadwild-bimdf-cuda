# QuadWild-BiMDF-CUDA: Experiments & Architecture

## Overview

CUDA-accelerated version of [QuadWild](https://github.com/nicopietroni/quadwild) (Reliable Feature-Line Driven Quad-Remeshing, Siggraph 2021) with pipeline checkpointing modeled after QuadriFlow-cuda.

## Pipeline Architecture

### 3-Step Pipeline (quadwild executable)

```
Step 1: Remesh & Field Computation
  └─ Input: triangle mesh (.obj/.ply) + optional .sharp/.rosy
  └─ Output: mesh_rem.obj, mesh_rem.rosy, mesh_rem.sharp
  └─ Algorithm: Isotropic remeshing + 4-vector field computation

Step 2: Field Tracing
  └─ Input: remeshed mesh + field
  └─ Output: mesh_rem_p0.obj, .patch, .corners, .feature, .c_feature
  └─ Algorithm: Field-line tracing → patch decomposition

Step 3: Quadrangulation (quad_from_patches)
  └─ Input: partitioned mesh + patches + corners
  └─ Sub-stages:
     ├─ post-load:         Load mesh + partitions + features
     ├─ post-chartdata:    Compute ChartData (subsides, adjacency)
     ├─ post-flow:         Solve BiMDF min-cost flow (quantization)
     ├─ post-quadrangulate: Generate quad mesh from flow solution
     └─ post-smooth:       Constrained Laplacian smoothing (CUDA)
  └─ Output: *_quadrangulation.obj, *_quadrangulation_smooth.obj
```

### Checkpoint System

Binary `.qwc` format (like QuadriFlow-cuda's `.qfc`):
- 512-byte header with magic "QWC", version, metadata
- Stage-dependent data serialization
- Full state reconstruction from any checkpoint

#### quad_from_patches CLI flags:
```bash
-save-dir DIR         # Checkpoint directory
-save-at STAGE        # Save after specific stage
-save-all             # Save after every stage
-run-from STAGE       # Resume from checkpoint
-run-to STAGE         # Stop after reaching stage
-list-stages          # Print all stage names
-no-cuda-smooth       # Force CPU smoothing
```

#### quadwild CLI flags:
```bash
-run-from N           # Start from step N (1-3)
-run-to N             # Stop after step N (1-3)
-save-dir DIR         # (reserved for future use)
-list-stages          # Print all stages
```

### Usage Examples

```bash
# Full run with checkpoints at every stage
./quad_from_patches input.obj 0 setup.txt stats.json -save-all -save-dir /tmp/ckpt

# Resume from post-flow (skip expensive BiMDF solve)
./quad_from_patches input.obj 0 setup.txt stats.json -run-from post-flow -save-dir /tmp/ckpt

# Run only the flow solver stage
./quad_from_patches input.obj 0 setup.txt stats.json -run-from post-chartdata -run-to post-flow -save-dir /tmp/ckpt

# Run quadwild, skip step 1 (already have remeshed files)
./quadwild mesh.obj 3 -run-from 2

# Force CPU smoothing (for comparison)
./quad_from_patches input.obj 0 setup.txt stats.json -no-cuda-smooth
```

## CUDA Acceleration

### 1. Smoothing Kernels (`cuda/smooth_kernels.cu`)

**Replaces:** `MultiCostraintSmooth()` in `smooth_mesh.h`

**Kernels:**
- `k_laplacian_smooth` — Per-vertex neighbor averaging (CSR adjacency)
- `k_back_project_grid` — Closest-point projection to reference mesh with spatial hash grid
- `k_back_project` — Brute-force fallback for small meshes
- `k_project_sharp` — Sharp feature vertex projection to edge segments

**Data flow:**
1. Extract flat arrays from VCG meshes (CPU, one-time)
2. Upload to GPU (single H→D transfer)
3. Run 30 iterations entirely on GPU (no transfers between iterations)
4. Download results (single D→H transfer)
5. Write back to VCG mesh (CPU)

**Key optimization: Spatial hash grid**
- Grid cells: axis-aligned cubes of size `3 × avg_edge_length`
- Each triangle registered in all grid cells its AABB overlaps
- Closest-point query searches only nearby cells (±search_radius)
- Fallback to brute-force if no match found in grid

**Expected speedup:** 5-15× for meshes > 10K vertices
- Laplacian: embarrassingly parallel, memory-bandwidth limited
- Back-projection: O(1) per vertex with grid vs O(nF) brute-force

### 2. Chart Data Kernels (`cuda/chart_kernels.cu`)

**Accelerates:** `computeChartData()` in `qr_charts.cpp`

**Kernels:**
- `k_assign_face_labels` — Parallel face-to-partition mapping (one block per partition)
- `k_detect_border_faces` — Per-face border detection (check 3 adjacent faces)
- `k_extract_boundary_edges` — Atomic-based boundary edge extraction
- `k_compute_subside_lengths` — Per-subside length computation (vertex chain reduction)
- `k_build_adjacency_pairs` — Extract (patch_left, patch_right) pairs for adjacency

**Expected speedup:** 3-10× for meshes > 50K faces

### 3. VCG Bridge (`cuda/vcg_bridge.h`)

Header-only template library for extracting flat arrays from VCG mesh types:
- `extract_vertex_positions()` — VCG mesh → float[N×3]
- `extract_tri_faces()` — VCG mesh → int[F×3]
- `extract_face_adjacency()` — VCG FF topology → int[F×3]
- `build_vertex_adjacency_csr()` — VCG mesh → CSR (offsets, indices)
- `extract_poly_faces()` — Variable-size faces → (sizes, indices)
- `build_sharp_feature_data()` — Feature edges → grouped edge segments

## Build

```bash
cmake . -B build -DSATSUMA_ENABLE_BLOSSOM5=0
cmake --build build -j$(nproc)

# Outputs:
# build/Build/bin/quadwild
# build/Build/bin/quad_from_patches
```

**Requirements:**
- CUDA Toolkit (SM 86+ for RTX A6000)
- C++20 compiler
- All other dependencies are vendored

## File Structure

```
cuda/
├── CMakeLists.txt           # CUDA library build config
├── qw_checkpoint.h          # Checkpoint system header
├── qw_checkpoint.cpp        # Checkpoint save/load implementation
├── qw_serialize.h           # Binary serialization templates
├── smooth_kernels.h          # CUDA smoothing interface
├── smooth_kernels.cu         # CUDA smoothing kernels
├── chart_kernels.h           # CUDA chart computation interface
├── chart_kernels.cu          # CUDA chart computation kernels
└── vcg_bridge.h             # VCG mesh ↔ flat array conversion
```

## Pipeline Bottleneck Analysis

### Typical profile (Step 3, ~1000 patch mesh):

| Stage | Time | CUDA? | Notes |
|-------|------|-------|-------|
| Load mesh + partitions | ~0.1s | No | I/O bound |
| computeChartData | ~0.5s | Partial | GPU: labels, borders, lengths |
| findSubdivisions (BiMDF) | ~5-30s | No* | Satsuma solver, main bottleneck |
| quadrangulate | ~1-3s | No | UV mapping + mesh generation |
| MultiCostraintSmooth | ~2-8s | **Yes** | 30 iterations, fully on GPU |

*BiMDF solver is the primary bottleneck. CUDA acceleration would require
rewriting the satsuma shortest-path algorithm as GPU kernels. This is
the highest-impact future work target.

### Future CUDA targets:
1. **BiMDF flow solver** — Rewrite successive shortest paths on GPU
2. **UV parameterization** — GPU LSCM solver (libigl replacement)
3. **Remeshing** — GPU isotropic remesher (step 1)

## Checkpoint File Format (.qwc)

```
Offset  Size    Field
0       4       Magic: "QWC\0"
4       4       Version: 1
8       64      Stage name
72      256     Input mesh path
328     256     Setup file path
584     4       Alpha (float)
588     4       Scale factor (float)
592     4       Fixed chart clusters
596     4       Use flow solver flag
600     8       Timestamp (unix)
608     4       Num vertices
612     4       Num faces
616     4       Num patches
620     4       Num subsides
624     128     Reserved
752     ...     Serialized data (stage-dependent)
```

Serialization format:
- Scalars: raw binary (4 or 8 bytes)
- Vectors: int32 count + elements
- Nested vectors: recursive count + elements
- size_t: serialized as int64 for portability
