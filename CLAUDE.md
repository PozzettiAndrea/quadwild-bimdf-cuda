# QuadWild-BiMDF-CUDA

CUDA-accelerated version of [QuadWild](https://github.com/nicopietroni/quadwild) (Siggraph 2021) with experimental Penner coordinates pipeline.

## What Was Built

### CUDA Acceleration (`cuda/`)
- **15 BiMDF flow solver strategies** selectable via `-flow-strategy`:
  `satsuma`, `early-term`, `admm`, `phase1-only`, `pdhg`, `pdhg-direct`, `sa`, `suitor`, `pdhg-v2`, `hybrid`, `sinpen`, `pump`, `suitor-aug`, `adaptive`, `directed`
- **CUDA smoothing kernels** (`smooth_kernels.cu`): Laplacian + spatial hash back-projection, 7x kernel speedup
- **GPU PDHG LP solver** (`pdhg_solver.cu`): cuSPARSE SpMV, Ruiz scaling
- **GPU ADMM LP solver** (`admm_solver.cu`): consensus ADMM
- **GPU simulated annealing** (`sa_solver.cu`): 4096 parallel chains
- **GPU Suitor matching** (`suitor_solver.cu`): packed struct + 3-cycle search
- **Pipeline checkpoint system** (`qw_checkpoint.h/cpp`): save/resume at any stage
- **VCG mesh bridge** (`vcg_bridge.h`): sort-based CSR, flat array extraction

### Penner Coordinates Pipeline (`libs/penner-optimization/`)
- Vendored [penner-optimization](https://github.com/geometryprocessing/penner-optimization) library
- Custom `penner_quantize` executable for quad meshing
- Metric regularization fixes Newton NaN divergence
- Mesh doubling for boundary handling (dragon mesh)
- UV quantization script (`quantize_uv.py`)

### Bug Fixes
- `libs/xfield_tracer/tracing/edge_direction_table.h`: Fixed unguarded `map::at()` crash on dragon mesh
- `libs/lemon/CMakeLists.txt`: Fixed CMP0048 policy for newer CMake
- `libs/penner-optimization/src/holonomy/*/`: Fixed `has_priority` multiple definition

## Build

```bash
cmake . -B build_cuda -DSATSUMA_ENABLE_BLOSSOM5=0 \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11
cmake --build build_cuda -j$(nproc)

# Penner pipeline (separate build)
cd libs/penner-optimization
cmake -B build_penner -DUSE_PYBIND=OFF -DENABLE_VISUALIZATION=OFF \
    -DRENDER_TEXTURE=OFF -DBUILD_CURVATURE_METRIC_TESTS=OFF \
    -DCHECK_VALIDITY=OFF -DUSE_COMISO=ON -DUSE_SUITESPARSE=OFF \
    -DUSE_MULTIPRECISION=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DMPFR_INCLUDE_DIR=/usr/include \
    -DMPFR_LIBRARIES=/usr/lib/x86_64-linux-gnu/libmpfr.so
cmake --build build_penner -j$(nproc) --target penner_quantize
```

## Usage

```bash
# QuadWild pipeline with flow strategy selection
./quad_from_patches input.obj 0 setup.txt stats.json \
    -flow-strategy early-term

# Available strategies: satsuma early-term suitor phase1-only
#   pdhg pdhg-direct sa admm pdhg-v2 hybrid sinpen pump
#   suitor-aug adaptive directed

# Checkpoint system
./quad_from_patches input.obj 0 setup.txt stats.json \
    -save-all -save-dir /tmp/ckpt
./quad_from_patches input.obj 0 setup.txt stats.json \
    -run-from post-flow -save-dir /tmp/ckpt

# Penner pipeline (experimental)
./penner_quantize -i mesh.obj -o output.obj --max_itr 9
python3 quantize_uv.py output.obj quads.obj --scale 2.0
```

## Benchmark Results (Dragon, 15K verts)

| Strategy | Time | BiMDF Cost | Speedup |
|---|---|---|---|
| satsuma (baseline) | 123s | 16.69 | 1.0x |
| early-term | 42s | 17.04 | 2.9x |
| suitor | 9.4s | 18.12 | 13x |
| phase1-only | 3.2s | 19.71 | 38x |

## Penner Results (Gargoyle, 25K verts)

| Config | Time | Output |
|---|---|---|
| 0 Newton iters | 3.1s | Valid UV, 72% quads after quantize |
| 9 Newton iters (lambda0=0.1) | 6.3s | Valid UV, Newton converges |
| QuadWild comparison | 4.2s | 100% quads, 4.2% irregular |

## Status

- **Production ready:** 15 BiMDF strategies, CUDA smoothing, checkpoint system
- **Experimental:** Penner pipeline (works on gargoyle, dragon needs debugging)
- **Known issues:** Penner Newton error increases (needs proper intrinsic Delaunay preprocessing), dragon mesh doubling crashes in CoMISo cross field computation

## Research (boffins.md)

Extensive debate transcripts in `boffins.md` covering:
- GPU Blossom matching (impossible at this graph size)
- PDHG/ADMM LP solvers (LP gap too large for BiMDF)
- Penner coordinates (bypasses BiMDF entirely)
- 55+ papers surveyed in `/home/shadeform/relevant_papers_cuda_remeshing/`
