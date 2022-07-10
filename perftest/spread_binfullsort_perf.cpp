/** @file
 *
 * Exploration of full sorting (rather than index sorting) to optimize data movement.
 * Currently the gather stage is the main bottleneck in spreading. We are hoping
 * that ips4o is better able to make use of data locality in the sorting stage,
 * and hence improve total throughput of sorting.
 *
 */

#include <array>
#include <cstdint>

#include <ips4o/ips4o.hpp>

#include "../src/kernels/avx512/spread_bin_sort_int.h"
#include "../src/kernels/reference/gather_fold_reference.h"
#include "../src/kernels/reference/spread_bin_sort_int.h"

#include "../src/tracing.h"

#include <benchmark/benchmark.h>

#include "../test/spread_test_utils.h"

using namespace finufft::spreading;
using namespace finufft::spreading::reference;

namespace {

struct SortPackedTimers {
    finufft::Timer pack;
    finufft::Timer sort;
    finufft::Timer unpack;

    SortPackedTimers(finufft::Timer &timer)
        : pack(timer.make_timer("pack")), sort(timer.make_timer("sort")),
          unpack(timer.make_timer("unpack")) {}
    SortPackedTimers(SortPackedTimers const &) = default;
    SortPackedTimers(SortPackedTimers &&) = default;
};

/** Sorts points by bin index in packed format.
 *
 */
template <typename T, std::size_t Dim, typename FoldRescale>
void sort_packed(
    nu_point_collection<Dim, const T> const& points, nu_point_collection<Dim, T> const& output,
    uint32_t *bin_index, IntGridBinInfo<T, Dim> const &info, SortPackedTimers &timers) {

    auto packed = finufft::allocate_aligned_array<PointBin<T, Dim>>(points.num_points, 64);

    // Compute bins
    {
        finufft::ScopedTimerGuard guard(timers.pack);
        finufft::spreading::avx512::compute_bins_and_pack(
            points, FoldRescaleRange::Pi, info, packed.get());
    }

    {
        finufft::ScopedTimerGuard guard(timers.sort);
        ips4o::parallel::sort(packed.get(), packed.get() + points.num_points);
    }

    // Unpack to output.
    {
        finufft::ScopedTimerGuard guard(timers.unpack);
        unpack_bins_to_points(packed.get(), output, bin_index);
    }
}

template <typename T, std::size_t Dim> void bench_sort_packed(benchmark::State &state) {
    auto num_points = state.range(0);

    auto points = make_random_point_collection<Dim, T>(num_points, 1, {-3 * M_PI, 3 * M_PI});
    auto output = finufft::spreading::SpreaderMemoryInput<Dim, T>(num_points);
    auto output_bin_index = finufft::allocate_aligned_array<uint32_t>(num_points, 64);

    std::array<std::size_t, Dim> extents;
    extents.fill(1024);

    // Set-up standard grid size for 32kb L1 cache.
    // Set-up padding for avx512 2d functor
    std::array<std::size_t, Dim> grid_size;
    std::array<KernelWriteSpec<T>, Dim> padding;

    grid_size[0] = 1 << 15;
    padding[0].grid_left = 0;
    padding[0].grid_right = 16;
    padding[0].offset = 4;

    for (std::size_t i = 1; i < Dim; ++i) {
        grid_size[0] /= 16;

        grid_size[i] = 16;
        padding[i].grid_left = 0;
        padding[i].grid_right = 8;
        padding[i].offset = 4;
    }

    IntGridBinInfo<T, Dim> info(extents, grid_size, padding);

    finufft::TimerRoot timer_root("benchmark");
    auto timer = timer_root.make_timer("sort_packed");
    SortPackedTimers timers(timer);

    for (auto _ : state) {
        sort_packed<T, Dim, FoldRescalePi<T>>(points, output, output_bin_index.get(), info, timers);
        benchmark::DoNotOptimize(output.coordinates[0][0]);
        benchmark::DoNotOptimize(output_bin_index[0]);
        benchmark::DoNotOptimize(points.coordinates[0][0]);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_points);

    // Report additional subtimings
    auto results = timer_root.report("sort_packed");
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

BENCHMARK(bench_sort_packed<float, 2>)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 24)
    ->Unit(benchmark::kMillisecond);
