/** Benchmark for full spreading (without FFT) in type-1 transform.
 *
 * This benchmark is intended to probe the performance of the spreading
 * part of the type-1 transform in 2D. In particular, we are testing
 * out a new pipeline based on:
 * - full packed sorting
 * - cache-aware subproblem blocking
 *
 */

#include <cstring>

#include <benchmark/benchmark.h>

#include "../src/kernels/spreading.h"

#include <tcb/span.hpp>

#include <ips4o/ips4o.hpp>

#include "../src/kernels/avx512/spread_avx512.h"
#include "../src/kernels/avx512/spread_bin_sort_int.h"
#include "../src/kernels/reference/gather_fold_reference.h"
#include "../src/kernels/reference/spread_bin_sort_int.h"
#include "../src/kernels/reference/spread_blocked.h"
#include "../src/kernels/reference/synchronized_accumulate_reference.h"

#include "../src/tracing.h"

#include "../test/spread_test_utils.h"

using namespace finufft::spreading;

namespace {


void bm_spread_2d(benchmark::State &state) {
    std::size_t target_size = state.range(0);
    std::size_t kernel_width = state.range(1);

    auto num_points = target_size * target_size;

    finufft::TimerRoot root("bench_full_spread");
    auto timer = root.make_timer("full_spread");

    auto points = make_random_point_collection<2, float>(num_points, 0, {-3 * M_PI, 3 * M_PI});
    auto kernel_spec = specification_from_width(kernel_width, 2);
    auto output = finufft::allocate_aligned_array<float>(2 * target_size * target_size, 64);

    auto spread_functor = make_avx512_blocked_spread_functor<float, 2>(
        kernel_spec, std::array<std::size_t, 2>{target_size, target_size}, FoldRescaleRange::Pi, timer);

    for (auto _ : state) {
        spread_functor(points, output.get());
        benchmark::DoNotOptimize(output[0]);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_points);

    // Report additional subtimings
    auto results = root.report("full_spread");
    for (auto &name_and_time : results) {
        auto name = std::get<0>(name_and_time);

        if (name.empty()) {
            continue;
        }

        auto time = std::chrono::duration<double>(std::get<1>(name_and_time)).count();
        if (time == 0) {
            continue;
        }

        state.counters[name] = benchmark::Counter(time, benchmark::Counter::kIsRate);
    }
}

} // namespace

BENCHMARK(bm_spread_2d)
    ->ArgsProduct({{1 << 10, 1 << 11, 1 << 12}, {4, 6, 8}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
