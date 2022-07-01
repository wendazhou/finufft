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


add_executable(spread_gather_perf spread_gather_perf.cpp)
target_link_libraries(spread_gather_perf PRIVATE finufft benchmark::benchmark_main)
enable_asan(spread_gather_perf)

add_executable(spread_subproblem_1d_perf spread_subproblem_1d_perf.cpp)
target_link_libraries(spread_subproblem_1d_perf PRIVATE finufft benchmark::benchmark_main)
enable_asan(spread_subproblem_1d_perf)

add_executable(spread_subproblem_2d_perf spread_subproblem_2d_perf.cpp)
target_link_libraries(spread_subproblem_2d_perf PRIVATE finufft benchmark::benchmark_main)
enable_asan(spread_subproblem_2d_perf)

add_executable(spread_subproblem_3d_perf spread_subproblem_3d_perf.cpp)
target_link_libraries(spread_subproblem_3d_perf PRIVATE finufft benchmark::benchmark_main)
enable_asan(spread_subproblem_3d_perf)
