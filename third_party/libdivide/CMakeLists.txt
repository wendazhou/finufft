add_library(libdivide INTERFACE)
target_include_directories(libdivide INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(libdivide INTERFACE LIBDIVIDE_SSE2 LIBDIVIDE_AVX2 LIBDIVIDE_AVX512)
