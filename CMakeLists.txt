cmake_minimum_required(VERSION 3.19)

project(finufft VERSION 2.1.0 LANGUAGES C CXX)

set(GNU_LIKE_FRONTENDS AppleClang Clang GNU)
if(CMAKE_CXX_COMPILER_ID IN_LIST GNU_LIKE_FRONTENDS)
    # Set custom compiler flags for gcc-compatible compilers
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -funroll-loops")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG -funroll-loops")
endif()

include(CTest)
include(FetchContent)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.0.tar.gz
)

set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_INSTALL OFF)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG 1531ee0d634d8563da6a3f36ea2597c95d1c2d46
)


FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git 
    GIT_TAG 32754dc13d8be3cb272f029d135f36fbaea1e491)

# Function to create a new scope in order to set variables
set (HWY_ENABLE_INSTALL OFF)
set (HWY_ENABLE_EXAMPLES OFF)
set (BUILD_TESTING OFF)
FetchContent_MakeAvailable(highway)
set (BUILD_TESTING ON)

FetchContent_Declare(
    ittapi
    GIT_REPOSITORY https://github.com/intel/ittapi.git
    GIT_TAG 2428ed97aa977c66b30940081e7d3e9b1a3c740)
set(ITT_API_IPT_SUPPORT ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ittapi)

set(FINUFFT_ARCH_FLAGS "-march=native" CACHE STRING "Compiler flags for specifying target architecture.")
set(FINUFFT_FFTW_SUFFIX "OpenMP" CACHE STRING "Suffix for FFTW libraries (e.g. OpenMP, Threads etc.)")
set(FINUFFT_BUILD_EXAMPLES "OFF" CACHE BOOL "Whether to build the FINUFFT examples")
set(FINUFFT_BUILD_TESTS "OFF" CACHE BOOL "Whether to build the FINUFFT tests")
set(FINUFFT_BUILD_FORTRAN "OFF" CACHE BOOL "Whether to build the FINUFFT Fortran examples")
set(FINUFFT_BUILD_MATLAB "OFF" CACHE BOOL "Whether to build the FINUFFT Matlab interface")
set(FINUFFT_USE_OPENMP "ON" CACHE BOOL "Whether to use OpenMP for parallelization. If disabled, the finufft library will be single threaded. This does not affect the choice of FFTW library.")
set(FINUFFT_SANITIZER "address" CACHE STRING "If not empty, sanitizer to use. Only in effect for Debug configuration.")

if(FINUFFT_BUILD_MATLAB)
    # When building for matlab, we will fetch the OpenMP library used by matlab
    # instead of system default for compatibility.
    find_package(Matlab REQUIRED)
    find_library(matlab_iomp5_lib NAMES iomp5 HINTS ${Matlab_ROOT_DIR}/sys/os/ PATH_SUFFIXES glnxa64 maci64)
    find_library(pthreads_lib NAMES pthread CMAKE_FIND_ROOT_PATH_BOTH)

    # Create a "fake" imported library pointing to the matlab openmp implementation
    add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES IMPORTED_LOCATION ${matlab_iomp5_lib})
    target_link_libraries(OpenMP::OpenMP_CXX INTERFACE ${pthreads_lib})
    # Set the OpenMP flag.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_options(OpenMP::OpenMP_CXX INTERFACE -Xclang -fopenmp)
    else()
        target_compile_options(OpenMP::OpenMP_CXX INTERFACE -fopenmp)
    endif()
else()
    # For non-matlab builds, find system OpenMP
    if(FINUFFT_USE_OPENMP)
        find_package(OpenMP REQUIRED)
    endif()
endif()

# Find required packages
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(FFTW REQUIRED)


set(FINUFFT_FFTW_LIBRARIES "FFTW::Float" "FFTW::Double" "FFTW::Float${FINUFFT_FFTW_SUFFIX}" "FFTW::Double${FINUFFT_FFTW_SUFFIX}")

# This set of sources is compiled twice, once in single precision and once in double precision
# The single precision compilation is done with -DSINGLE
set(FINUFFT_PRECISION_DEPENDENT_SOURCES src/finufft.cpp src/simpleinterfaces.cpp src/spreadinterp.cpp src/utils.cpp fortran/finufftfort.cpp)

# Set of compilers which behave like gcc
set(FINUFFT_GNU_LIKE_COMPILERS AppleClang Clang GNU)

# Utility function to enable ASAN on debug builds
function(enable_asan target)
    if(${FINUFFT_SANITIZER} STREQUAL "")
        return()
    endif()

    if(CMAKE_CXX_COMPILER_ID IN_LIST FINUFFT_GNU_LIKE_COMPILERS)
        # Enable only on clang / gcc compilers.
        target_compile_options(${target} PRIVATE $<$<CONFIG:DEBUG>:-fsanitize=${FINUFFT_SANITIZER}>)
        target_link_options(${target} PRIVATE $<$<CONFIG:DEBUG>:-fsanitize=${FINUFFT_SANITIZER}>)
    endif()
endfunction()

# Use C++20
set(CMAKE_CXX_STANDARD 20)

# Utility function to set finufft compilation options.
function(set_finufft_options target)
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    enable_asan(${target})

    target_compile_options(${target} PRIVATE SHELL:$<$<CONFIG:Release,RelWithDebInfo>:${FINUFFT_ARCH_FLAGS}>)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Enable cx-limited-range on supported compilers
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release,RelWithDebInfo>:-fcx-limited-range>)
    endif()
    target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
    if(FINUFFT_USE_OPENMP)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    else()
        if(CMAKE_CXX_COMPILER_ID IN_LIST FINUFFT_GNU_LIKE_COMPILERS)
            # OpenMP disabled, suppress unknown pragma warnings to avoid spam.
            target_compile_options(${target} PRIVATE "-Wno-unknown-pragmas")
        endif()
    endif()

    target_link_libraries(${target} PUBLIC hwy ittnotify)
    target_link_libraries(${target} PRIVATE libdivide ips4o)
endfunction()

find_package(TBB REQUIRED)
add_subdirectory(third_party)

# Main finufft libraries
add_library(finufft_f32 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
target_compile_definitions(finufft_f32 PRIVATE SINGLE)
set_finufft_options(finufft_f32)
target_link_libraries(finufft_f32 PUBLIC ${FINUFFT_FFTW_LIBRARIES})

add_library(finufft_f64 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
set_finufft_options(finufft_f64)
target_link_libraries(finufft_f64 PUBLIC ${FINUFFT_FFTW_LIBRARIES})

add_library(finufft_obj OBJECT
    src/utils_precindep.cpp
    contrib/legendre_rule_fast.cpp
    src/memory.cpp
    src/plan.cpp
    src/tracing.cpp
    src/precomputed_poly_kernel_data.cpp
    src/precomputed_poly_kernel_data_helper.cpp
    src/kernels/dispatch.cpp
    src/kernels/legacy/interpolation_legacy.cpp
    src/kernels/legacy/spread_kernel_legacy.cpp
    src/kernels/legacy/spread_legacy.cpp
    src/kernels/fft/fftw.cpp
    src/kernels/reference/sort_bin_counting.cpp
    src/kernels/reference/spread_bin_sort_int.cpp
    src/kernels/reference/spread_bin_sort_reference.cpp
    src/kernels/reference/spread_blocked.cpp
    src/kernels/reference/spread_blocked_st.cpp
    src/kernels/reference/spread_kernel.cpp
    src/kernels/reference/spreading_reference.cpp
    src/kernels/reference/synchronized_accumulate_reference.cpp)
set_finufft_options(finufft_obj)

add_library(finufft SHARED)
set_finufft_options(finufft)
target_link_libraries(finufft PUBLIC finufft_f32 finufft_f64 finufft_obj)
target_include_directories(finufft PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(MSVC)
    # Visibility not currently handled well, simply export all symbols
    # TODO: fix visibility in shared library.
    set_property(TARGET finufft PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_library(finufft_static STATIC)
set_finufft_options(finufft)
target_link_libraries(finufft_static PUBLIC finufft_f32 finufft_f64 finufft_obj)
target_include_directories(finufft_static PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")


# Utility function to define a new set of objects for the given kernels
# These need to be compiled with architecture flags which vary across the different libraries,
# hence the need to define separate object libraries for each of them.
function(add_finufft_kernel target)
    cmake_parse_arguments(PARSE_ARGV 1 FINUFFT_KERNEL "" "" "SOURCES;FLAGS")

    add_library(${target} OBJECT ${FINUFFT_KERNEL_SOURCES})
    set_finufft_options(${target})
    target_compile_options(${target} PRIVATE ${FINUFFT_KERNEL_FLAGS})
    target_link_libraries(finufft PUBLIC ${target})
    enable_asan(${target})
endfunction()


add_finufft_kernel(finufft_avx512
    SOURCES
        src/kernels/avx512/gather_fold_avx512.cpp
        src/kernels/avx512/sort_bin_counting.cpp
        src/kernels/avx512/spread_avx512.cpp
        src/kernels/avx512/spread_1d_avx512_impl.cpp
        src/kernels/avx512/spread_2d_avx512_impl.cpp
        src/kernels/avx512/spread_3d_avx512_impl.cpp
        src/kernels/avx512/spread_bin_sort_int_impl.cpp
    FLAGS -mavx512f -mavx512vl -mavx512dq -mbmi -mbmi2)
add_finufft_kernel(finufft_avx2 SOURCES src/kernels/avx2/gather_fold_avx2.cpp FLAGS -mavx -mavx2 -mfma)
add_finufft_kernel(finufft_hwy
    SOURCES
        src/kernels/hwy/gather_fold_hwy_impl.cpp
        src/kernels/hwy/spread_bin_sort_hwy_impl.cpp)
target_include_directories(finufft_hwy PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/hwy")

# Add tests defined in their own directory
if(BUILD_TESTING AND FINUFFT_BUILD_TESTS)

# We also fetch googletest and googlebenchmark
FetchContent_MakeAvailable(googletest)
FetchContent_MakeAvailable(benchmark)

add_subdirectory(test)
add_subdirectory(perftest)
endif()

if(FINUFFT_BUILD_EXAMPLES)
add_subdirectory(examples)
endif()

if(FINUFFT_BUILD_FORTRAN)
enable_language(Fortran)
add_subdirectory(fortran)
endif()

if(FINUFFT_BUILD_MATLAB)
add_subdirectory(matlab)
endif()
