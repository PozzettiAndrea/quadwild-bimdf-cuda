# Vendored RXMesh library for GPU isotropic remeshing
# Built separately with CUDA_SEPARABLE_COMPILATION ON
# (quadwild's main CUDA code uses OFF for performance)

cmake_minimum_required(VERSION 3.18)

# METIS (mesh partitioning, required by RXMesh)
set(IDXTYPEWIDTH 32 CACHE STRING "" FORCE)
set(REALTYPEWIDTH 32 CACHE STRING "" FORCE)
add_subdirectory(third_party/metis)
target_include_directories(metis INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/metis/include>)
target_compile_definitions(metis INTERFACE
    IDXTYPEWIDTH=${IDXTYPEWIDTH}
    REALTYPEWIDTH=${REALTYPEWIDTH})

# CUDA toolkit (need targets like CUDA::cudart, CUDA::cusparse, CUDA::cusolver)
find_package(CUDAToolkit REQUIRED)

# RXMesh core library
add_library(rxmesh_core
    # CPU sources
    include/rxmesh/rxmesh.cpp
    include/rxmesh/util/MshLoader.cpp
    include/rxmesh/util/MshSaver.cpp
    # CUDA sources
    include/rxmesh/attribute.cu
    include/rxmesh/hash_functions.cu
    include/rxmesh/lp_hashtable.cu
    include/rxmesh/patch_info.cu
    include/rxmesh/patch_lock.cu
    include/rxmesh/patch_scheduler.cu
    include/rxmesh/patch_stash.cu
    include/rxmesh/patcher/patcher.cu
    include/rxmesh/query.cu
    include/rxmesh/reduce_handle.cu
    include/rxmesh/rxmesh_dynamic.cu
    include/rxmesh/rxmesh_static.cu
)

set_target_properties(rxmesh_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "120"
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
    CXX_STANDARD 17
)

target_include_directories(rxmesh_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party   # glm, spdlog, cereal, rapidjson, CLI
)

# Eigen from parent project
target_link_libraries(rxmesh_core PUBLIC Eigen3::Eigen)

# METIS
target_link_libraries(rxmesh_core PUBLIC metis)

# CUDA libraries
target_link_libraries(rxmesh_core PUBLIC
    CUDA::cudart
    CUDA::cusparse
    CUDA::cusolver
)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rxmesh_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Required compile definitions
target_compile_definitions(rxmesh_core PUBLIC
    EIGEN_DEFAULT_DENSE_INDEX_TYPE=int
)

# CUDA compile flags matching RXMesh upstream
target_compile_options(rxmesh_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -Xcudafe=--diag_suppress=177
        -Xcudafe=--diag_suppress=174
        -Xcudafe=--diag_suppress=20054
        -lineinfo
        -w
    >
    $<$<COMPILE_LANGUAGE:CXX>:-w>
)

add_library(rxmesh::core ALIAS rxmesh_core)
