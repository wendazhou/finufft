#include <benchmark/benchmark.h>
#include "transform_benchmark.h"

BENCHMARK(benchmark_type1<float, 1>)->Arg(1 << 20)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(benchmark_type1<float, 2>)->Arg(1 << 10)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(benchmark_type1<float, 3>)->Arg(1 << 7)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(benchmark_type1<double, 1>)->Arg(1 << 20)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(benchmark_type1<double, 2>)->Arg(1 << 10)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(benchmark_type1<double, 3>)->Arg(1 << 7)->Unit(benchmark::kMillisecond)->UseRealTime();
