# CUDA acceleration library for QuadWild-BiMDF
#
# Always builds:
#   - Checkpoint/serialization system (pure C++)
#   - VCG bridge (header-only)
#
# With CUDA:
#   - GPU smoothing kernels
#   - GPU chart data computation kernels

set(QW_CUDA_SOURCES
    qw_checkpoint.cpp
)

set(QW_CUDA_HEADERS
    qw_checkpoint.h
    qw_serialize.h
    vcg_bridge.h
    smooth_kernels.h
    chart_kernels.h
)

if(QUADWILD_HAS_CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(qw_cuda
        ${QW_CUDA_SOURCES}
        smooth_kernels.cu
        chart_kernels.cu
        bimdf_cuda.cu
        admm_solver.cu
        pdhg_solver.cu
        sa_solver.cu
        suitor_solver.cu
    )

    set_target_properties(qw_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "86;89"
        POSITION_INDEPENDENT_CODE ON
    )

    target_compile_definitions(qw_cuda PUBLIC QUADWILD_HAS_CUDA=1)
    target_link_libraries(qw_cuda PUBLIC CUDA::cudart CUDA::cusparse)
    # Need satsuma headers for bimdf_bridge.h (header-only bridge)
    target_link_libraries(qw_cuda PUBLIC satsuma::satsuma)

    message(STATUS "Building qw_cuda WITH CUDA kernels (SM 86, 89)")
else()
    # CPU-only: just the checkpoint system
    add_library(qw_cuda
        ${QW_CUDA_SOURCES}
    )

    target_compile_definitions(qw_cuda PUBLIC QUADWILD_HAS_CUDA=0)

    message(STATUS "Building qw_cuda WITHOUT CUDA (checkpoint system only)")
endif()

target_compile_features(qw_cuda PUBLIC cxx_std_20)
target_include_directories(qw_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(quadwild::cuda ALIAS qw_cuda)
