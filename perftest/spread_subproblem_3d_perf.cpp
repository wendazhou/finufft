#include "spread_subproblem_perf.h"
#include <benchmark/benchmark.h>

BENCHMARK(benchmark_avx512<float, 3>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_reference<float, 3>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_legacy<float, 3>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_direct<float, 3>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_avx512<double, 2>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_reference<double, 3>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_legacy<double, 3>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_direct<double, 3>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
