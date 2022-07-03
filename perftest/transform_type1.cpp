#include <benchmark/benchmark.h>

#include "transform_benchmark.h"

BENCHMARK(benchmark_type1<float, 1>)->Arg(32678)->Unit(benchmark::kMillisecond);
BENCHMARK(benchmark_type1<float, 2>)->Arg(128)->Unit(benchmark::kMillisecond);
BENCHMARK(benchmark_type1<float, 3>)->Arg(32)->Unit(benchmark::kMillisecond);

BENCHMARK(benchmark_type1<double, 1>)->Arg(32678)->Unit(benchmark::kMillisecond);
BENCHMARK(benchmark_type1<double, 2>)->Arg(128)->Unit(benchmark::kMillisecond);
BENCHMARK(benchmark_type1<double, 3>)->Arg(32)->Unit(benchmark::kMillisecond);
