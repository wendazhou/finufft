add_library(ips4o INTERFACE)
target_include_directories(ips4o INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(ips4o INTERFACE "-mcx16")
target_link_libraries(ips4o INTERFACE atomic TBB::tbb)
