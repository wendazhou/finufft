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

struct SpreadTimers {
    finufft::Timer sort_packed;
    finufft::Timer spread_blocked;

    SortPackedTimers sort_packed_timers;
    reference::SpreadBlockedTimers spread_blocked_timers;

    SpreadTimers(finufft::Timer &timer)
        : sort_packed(timer.make_timer("sp")), spread_blocked(timer.make_timer("sb")),
          sort_packed_timers(sort_packed), spread_blocked_timers(spread_blocked) {}
};

/** Compute grid size from cache size.
 *
 * Note, the current arrangement is only reasonable for 1D and 2D problems.
 * For 3D problems - better to target baseline size on L2 cache instead of L1.
 *
 */
template <typename T, std::size_t Dim> std::array<std::size_t, Dim> get_grid_size() {
    std::array<std::size_t, Dim> grid_size;

    std::size_t cache_size = 1 << 15;         // 32 kiB cache (L1D cache size)
    std::size_t element_size = sizeof(T) * 2; // 2x real numbers per point

    // Baseline size
    grid_size[0] = (1 << 15) / element_size;

    for (std::size_t i = 1; i < Dim; ++i) {
        grid_size[i] = 32;
        grid_size[0] /= 32;
    }

    return grid_size;
}

/** Compute grid sp
 *
 */
template <typename T>
std::vector<finufft::spreading::grid_specification<2>>
make_bin_grids(IntGridBinInfo<T, 2> const &info) {
    std::vector<grid_specification<2>> grids(info.num_bins_total());

    std::size_t idx = 0;

    for (std::size_t j = 0; j < info.num_bins[1]; ++j) {
        for (std::size_t i = 0; i < info.num_bins[0]; ++i) {
            grids[idx].extents[0] = info.grid_size[0];
            grids[idx].extents[1] = info.grid_size[1];

            grids[idx].offsets[0] = info.global_offset[0] + i * info.bin_size[0];
            grids[idx].offsets[1] = info.global_offset[1] + j * info.bin_size[1];

            idx += 1;
        }
    }

    return grids;
}

template <typename T, std::size_t Dim>
void spread(
    nu_point_collection<Dim, const T> const &points, kernel_specification const &kernel_spec,
    std::array<std::size_t, Dim> const &size, T *output, SpreadTimers &timer) {

    SpreaderMemoryInput<Dim, T> points_sorted(points.num_points);
    auto packed = finufft::allocate_aligned_array<PointBin<T, Dim>>(points.num_points, 64);

    auto spread_subproblem = get_subproblem_polynomial_avx512_functor<T, Dim>(kernel_spec);
    auto grid_size = get_grid_size<T, Dim>();

    IntGridBinInfo<T, Dim> info(size, grid_size, spread_subproblem.target_padding());

    auto bin_counts = finufft::allocate_aligned_array<size_t>(info.num_bins_total() + 1, 64);
    std::memset(bin_counts.get(), 0, (info.num_bins_total() + 1) * sizeof(size_t));

    {
        finufft::ScopedTimerGuard guard(timer.sort_packed);
        auto sort_packed = avx512::get_sort_functor<T, Dim>(&timer.sort_packed_timers);
        sort_packed(points, FoldRescaleRange::Pi, points_sorted, bin_counts.get() + 1, info);
    }

    // Compute bin boundaries
    {
        std::partial_sum(
            bin_counts.get(), bin_counts.get() + info.num_bins_total() + 1, bin_counts.get());
    }

    // Spread points
    auto spread_blocked = reference::make_omp_spread_blocked(
        std::move(spread_subproblem),
        get_reference_block_locking_accumulator<T, Dim>(),
        timer.spread_blocked_timers);

    {
        finufft::ScopedTimerGuard guard(timer.spread_blocked);
        spread_blocked(points_sorted, info, bin_counts.get(), output);
    }
}


void bm_spread_2d(benchmark::State &state) {
    std::size_t target_size = state.range(0);
    std::size_t kernel_width = state.range(1);

    auto num_points = target_size * target_size;

    finufft::TimerRoot root("bench_full_spread");
    auto timer = root.make_timer("full_spread");
    SpreadTimers timers(timer);

    auto points = make_random_point_collection<2, float>(num_points, 0, {-3 * M_PI, 3 * M_PI});
    auto kernel_spec = specification_from_width(kernel_width, 2);
    auto output = finufft::allocate_aligned_array<float>(2 * target_size * target_size, 64);

    for (auto _ : state) {
        spread<float, 2>(points, kernel_spec, {target_size, target_size}, output.get(), timers);
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
