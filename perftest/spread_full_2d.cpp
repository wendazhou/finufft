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
#include "../src/kernels/legacy/spread_legacy.h"
#include "../src/kernels/reference/gather_fold_reference.h"
#include "../src/kernels/reference/spread_bin_sort_int.h"
#include "../src/kernels/reference/spread_blocked.h"
#include "../src/kernels/reference/synchronized_accumulate_reference.h"

#include "../src/tracing.h"

#include "../test/spread_test_utils.h"

using namespace finufft::spreading;

namespace {

void report_timers(benchmark::State &state, finufft::TimerRoot const &timer_root) {
    // Report additional subtimings
    auto results = timer_root.report("full_spread");
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

void bm_spread_2d(
    benchmark::State &state, SpreadFunctor<float, 2> const &spread_functor,
    finufft::TimerRoot const &timer_root) {
    std::size_t target_size = state.range(0);
    auto num_points = target_size * target_size;

    auto points = make_random_point_collection<2, float>(num_points, 0, {-3 * M_PI, 3 * M_PI});
    auto output = finufft::allocate_aligned_array<float>(2 * target_size * target_size, 64);

    for (auto _ : state) {
        spread_functor(points, output.get());
        benchmark::DoNotOptimize(output[0]);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_points);

    // Report additional subtimings
    report_timers(state, timer_root);
}

void bm_avx512(benchmark::State &state) {
    std::size_t target_size = state.range(0);
    std::size_t kernel_width = state.range(1);

    auto kernel_spec = specification_from_width(kernel_width, 2);

    finufft::TimerRoot root("bench_full_spread");
    auto timer = root.make_timer("full_spread");

    auto spread_functor = avx512::get_blocked_spread_functor<float, 2>(
        kernel_spec,
        std::array<std::size_t, 2>{target_size, target_size},
        FoldRescaleRange::Pi,
        timer);

    bm_spread_2d(state, spread_functor, root);
}

void bm_legacy(benchmark::State &state) {
    std::size_t target_size = state.range(0);
    std::size_t kernel_width = state.range(1);

    auto kernel_spec = specification_from_width(kernel_width, 2);

    finufft::TimerRoot root("bench_full_spread");
    auto timer = root.make_timer("full_spread");

    auto spread_functor = legacy::make_spread_functor<float, 2>(
        kernel_spec,
        FoldRescaleRange::Pi,
        std::array<std::size_t, 2>{target_size, target_size},
        timer);

    bm_spread_2d(state, spread_functor, root);
}

} // namespace

BENCHMARK(bm_avx512)
    ->ArgsProduct({{1 << 10, 1 << 11, 1 << 12}, {4, 6, 8}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bm_legacy)
    ->ArgsProduct({{1 << 10, 1 << 11, 1 << 12}, {4, 6, 8}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
