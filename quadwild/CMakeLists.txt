add_executable(quadwild quadwild.cpp trace.cpp)
target_link_libraries(quadwild PRIVATE quadwild::quadretopology)
target_link_libraries(quadwild PRIVATE quadwild::lib_field_computation)
target_link_libraries(quadwild PRIVATE quadwild::xfield_tracer)
target_link_libraries(quadwild PRIVATE quadwild::quad_from_patches)
target_link_libraries(quadwild PRIVATE quadwild::cuda)

if(QUADWILD_HAS_CUDA)
    set_target_properties(quadwild PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

add_executable(cli_trace cli_trace.cpp trace.cpp)
target_link_libraries(cli_trace PRIVATE quadwild::xfield_tracer)
