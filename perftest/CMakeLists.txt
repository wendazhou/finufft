# Each source test file is instantiated in single and double precision
set(PERFTESTS guru_timing_test manysmallprobs spreadtestnd)

foreach(TEST ${PERFTESTS})
  add_executable(${TEST} ${TEST}.cpp)
  add_executable(${TEST}f ${TEST}.cpp)

  target_compile_definitions(${TEST}f PRIVATE -DSINGLE)

  target_link_libraries(${TEST} PRIVATE finufft)
  target_link_libraries(${TEST}f PRIVATE finufft)
  enable_asan(${TEST})
  enable_asan(${TEST}f)
endforeach()

function(add_benchmark name source)
  add_executable(${name} ${source})
  target_compile_options(${name} PRIVATE SHELL:$<$<CONFIG:Release,RelWithDebInfo>:${FINUFFT_ARCH_FLAGS}>)
  target_link_libraries(${name} PRIVATE finufft benchmark::benchmark_main OpenMP::OpenMP_CXX finufft_test_utils)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
  enable_asan(${name})
endfunction()

function(add_benchmark_no_main name source)
  add_executable(${name} ${source})
  target_compile_options(${name} PRIVATE SHELL:$<$<CONFIG:Release,RelWithDebInfo>:${FINUFFT_ARCH_FLAGS}>)
  target_link_libraries(${name} PRIVATE finufft benchmark::benchmark OpenMP::OpenMP_CXX finufft_test_utils)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
  enable_asan(${name})
endfunction()

add_benchmark(spread_binsort_perf spread_binsort_perf.cpp)
add_benchmark(spread_gather_perf spread_gather_perf.cpp)
add_benchmark(spread_pack_bin spread_pack_bin.cpp)
add_benchmark(spread_blocked spread_blocked.cpp)
add_benchmark(spread_subproblem_1d_perf spread_subproblem_1d_perf.cpp)
add_benchmark(spread_subproblem_2d_perf spread_subproblem_2d_perf.cpp)
add_benchmark(spread_subproblem_2d_scaling spread_subproblem_2d_scaling.cpp)
add_benchmark(spread_subproblem_3d_perf spread_subproblem_3d_perf.cpp)
add_benchmark(spread_unpack_bin spread_unpack_bin.cpp)
add_benchmark(transform_type1 transform_type1.cpp)
add_benchmark(transform_type1_large transform_type1_large.cpp)
add_benchmark(transform_type1_xlarge transform_type1_xlarge.cpp)

add_benchmark(spread_binfullsort_perf spread_binfullsort_perf.cpp)
target_link_libraries(spread_binfullsort_perf PRIVATE ips4o)

add_benchmark_no_main(spread_full_2d spread_full_2d.cpp)
