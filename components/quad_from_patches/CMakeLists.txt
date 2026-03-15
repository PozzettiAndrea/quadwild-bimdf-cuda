add_library(lib_quad_from_patches
        load_save.cpp
        #quad_from_patches.cpp
        #quad_mesh_tracer.cpp
)
target_compile_definitions(lib_quad_from_patches PUBLIC COMISO_FIELD)
target_include_directories(lib_quad_from_patches PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lib_quad_from_patches PUBLIC Timekeeper::libTimekeeper)
target_link_libraries(lib_quad_from_patches PUBLIC quadwild::quadretopology)
target_link_libraries(lib_quad_from_patches PUBLIC quadwild::cuda)
add_library(quadwild::quad_from_patches ALIAS lib_quad_from_patches)

add_executable(quad_from_patches main.cpp)
target_link_libraries(quad_from_patches quadwild::quad_from_patches nlohmann_json::nlohmann_json)

if(QUADWILD_HAS_CUDA)
    set_target_properties(quad_from_patches PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
