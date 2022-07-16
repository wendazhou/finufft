# Each source test file is instantiated in single and double precision
set(TESTS basicpassfail dumbinputs finufft1d_test finufft1dmany_test finufft2d_test finufft2dmany_test finufft3d_test finufft3dmany_test)

foreach(TEST ${TESTS})
  add_executable(${TEST} ${TEST}.cpp)
  add_executable(${TEST}f ${TEST}.cpp)

  target_compile_definitions(${TEST}f PRIVATE -DSINGLE)

  target_link_libraries(${TEST} PRIVATE finufft)
  target_link_libraries(${TEST}f PRIVATE finufft)
  enable_asan(${TEST})
  enable_asan(${TEST}f)
endforeach()

add_executable(testutils testutils.cpp)
target_link_libraries(testutils PRIVATE finufft ittnotify)
enable_asan(testutils)

add_test(NAME run_testutils COMMAND testutils)

# Ctest test definitions
function(add_tests_with_prec PREC REQ_TOL CHECK_TOL SUFFIX)

add_test(
  NAME run_basic_pass_fail_${PREC}
  COMMAND basicpassfail${SUFFIX})

add_test(
  NAME run_finufft1d_test_${PREC}
  COMMAND finufft1d_test${SUFFIX} 1e2 2e2 ${REQ_TOL} 0 2 0.0 ${CHECK_TOL})

add_test(
  NAME run_finufft1dmany_test_${PREC}
  COMMAND finufft1dmany_test${SUFFIX} 3 1e2 2e2 ${REQ_TOL} 0 0 0 2 0.0 ${CHECK_TOL})

add_test(
  NAME run_finufft2d_test_${PREC}
  COMMAND finufft2d_test${SUFFIX} 1e2 1e1 1e3 ${REQ_TOL} 0 2 0.0 ${CHECK_TOL})

add_test(
  NAME run_finufft2dmany_test_${PREC}
  COMMAND finufft2dmany_test${SUFFIX} 3 1e2 1e1 1e3 ${REQ_TOL} 0 0 0 2 0.0 ${CHECK_TOL})

add_test(
  NAME run_finufft3d_test_${PREC}
  COMMAND finufft3d_test${SUFFIX} 5 10 20 1e2 ${REQ_TOL} 0 2 0.0 ${CHECK_TOL})

add_test(
  NAME run_finufft3dmany_test_${PREC}
  COMMAND finufft3dmany_test${SUFFIX} 2 5 10 20 1e2 ${REQ_TOL} 0 0 0 2 0.0 ${CHECK_TOL})

add_test(
  NAME run_dumbinputs_${PREC}
  COMMAND dumbinputs${SUFFIX})

endfunction()

add_tests_with_prec(float 1e-5 2e-4 f)
add_tests_with_prec(double 1e-12 1e-11 "")

add_library(finufft_test_utils STATIC test_utils.cpp)
target_link_libraries(finufft_test_utils PRIVATE finufft OpenMP::OpenMP_CXX)
target_include_directories(finufft_test_utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
# Always enable optimization to avoid initialization taking too long
target_compile_options(finufft_test_utils PRIVATE -O3)

include(GoogleTest)

# Now add gtest-based tests
function (add_gtest_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE finufft gtest_main gmock finufft_test_utils)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
  enable_asan(${name})
  gtest_discover_tests(${name})
endfunction()

add_gtest_test(spread_add_wrapped_subgrid spread_add_wrapped_subgrid.cpp)
add_gtest_test(spread_bin_sort spread_bin_sort_test.cpp)
add_gtest_test(spread_bin_sort_int spread_bin_sort_int_test.cpp)
add_gtest_test(spread_full_test spread_full_test.cpp)
add_gtest_test(spread_gather_test spread_gather_test.cpp)
add_gtest_test(spread_poly_eval spread_poly_eval_test.cpp)
add_gtest_test(spread_subproblem_test spread_subproblem_test.cpp)
add_gtest_test(spread_sorted_integration spread_sorted_integration.cpp)
