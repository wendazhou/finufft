#include "spread_subproblem_perf.h"

#include <benchmark/benchmark.h>

BENCHMARK(benchmark_avx512<float, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_reference<float, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_legacy<float, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_direct<float, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_avx512<double, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_reference<double, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_legacy<double, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_direct<double, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
