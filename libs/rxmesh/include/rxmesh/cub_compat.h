#pragma once
// Compatibility shim for CUB API changes in CUDA 13+
// cub::Sum, cub::Max, cub::Min were removed in CUDA 13.x
// cub::KeyValuePair still exists

#include <cub/cub.cuh>

#if CUDART_VERSION >= 13000

namespace cub {

struct Sum {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const {
        return a + b;
    }
};

struct Max {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const {
        return (a > b) ? a : b;
    }
};

struct Min {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const {
        return (a < b) ? a : b;
    }
};

}  // namespace cub

#endif  // CUDART_VERSION >= 13000
