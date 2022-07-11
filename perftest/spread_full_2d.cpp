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

#include "../src/spreading.h"

#include <tcb/span.hpp>

#include <ips4o/ips4o.hpp>

#include "../src/kernels/avx512/spread_avx512.h"
#include "../src/kernels/avx512/spread_bin_sort_int.h"
#include "../src/kernels/reference/gather_fold_reference.h"
#include "../src/kernels/reference/spread_bin_sort_int.h"
#include "../src/kernels/reference/synchronized_accumulate_reference.h"

#include "../src/tracing.h"

#include "../test/spread_test_utils.h"

using namespace finufft::spreading;

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

struct SpreadBlockedTimers {
    SpreadBlockedTimers(finufft::Timer &timer)
        : gather(timer.make_timer("gather")), subproblem(timer.make_timer("subproblem")),
          accumulate(timer.make_timer("accumulate")) {}

    finufft::Timer gather;
    finufft::Timer subproblem;
    finufft::Timer accumulate;
};

struct SpreadTimers {
    finufft::Timer sort_packed;
    finufft::Timer spread_blocked;

    SortPackedTimers sort_packed_timers;
    SpreadBlockedTimers spread_blocked_timers;

    finufft::Timer compute_bin_boundaries;
    finufft::Timer make_grids;

    SpreadTimers(finufft::Timer &timer)
        : sort_packed(timer.make_timer("sp")), spread_blocked(timer.make_timer("sb")),
          sort_packed_timers(sort_packed), spread_blocked_timers(spread_blocked),
          compute_bin_boundaries(timer.make_timer("cbb")), make_grids(timer.make_timer("mg")) {}
};

/** Sorts points by bin index in packed format.
 *
 */
template <typename T, std::size_t Dim>
void sort_packed(
    nu_point_collection<Dim, const T> const &points, nu_point_collection<Dim, T> const &output,
    uint32_t *bin_index, IntBinInfo<T, Dim> const &info, SortPackedTimers &timers) {

    auto packed = finufft::allocate_aligned_array<PointBin<T, Dim>>(points.num_points, 64);

    // Compute bins
    {
        finufft::ScopedTimerGuard guard(timers.pack);
        avx512::compute_bins_and_pack(points, FoldRescaleRange::Pi, info, packed.get());
    }

    {
        finufft::ScopedTimerGuard guard(timers.sort);
        ips4o::parallel::sort(packed.get(), packed.get() + points.num_points);
    }

    // Unpack to output.
    {
        finufft::ScopedTimerGuard guard(timers.unpack);
        reference::unpack_bins_to_points(packed.get(), output, bin_index);
    }
}

void spread_blocked(
    finufft::spreading::nu_point_collection<2, const float> const &input, std::size_t num_blocks,
    tcb::span<const finufft::spreading::grid_specification<2>> grids,
    std::size_t const *point_block_boundaries, std::array<std::size_t, 2> const &sizes,
    float *output, SpreadSubproblemFunctor<float, 2> const &spread_subproblem,
    SynchronizedAccumulateFactory<float, 2> const &accumulate_factory,
    SpreadBlockedTimers &timers_ref) {

    typedef float T;
    const std::size_t Dim = 2;

    std::size_t max_grid_size = 0;
    std::size_t max_num_points = 0;

    // Determine amount of memory to allocate
    for (std::size_t i = 0; i < num_blocks; ++i) {
        max_grid_size = std::max(max_grid_size, std::size_t(grids[i].num_elements()));
        max_num_points =
            std::max(max_num_points, point_block_boundaries[i + 1] - point_block_boundaries[i]);
    }

    auto const &padding_info = spread_subproblem.target_padding();
    auto const &accumulate_subgrid = accumulate_factory(output, sizes);

#pragma omp parallel
    {
        auto subgrid_output = finufft::allocate_aligned_array<float>(2 * max_grid_size, 64);
        auto local_points = finufft::spreading::SpreaderMemoryInput<2, float>(max_num_points);
        SpreadBlockedTimers timers(timers_ref);

#pragma omp for
        for (std::size_t i = 0; i < num_blocks; ++i) {
            // Set number of points for subproblem
            auto block_num_points = point_block_boundaries[i + 1] - point_block_boundaries[i];
            local_points.num_points = block_num_points;

            auto &grid = grids[i];

            // Zero local memory
            std::memset(subgrid_output.get(), 0, 2 * grid.num_elements() * sizeof(float));

            // Gather local points
            {
                finufft::ScopedTimerGuard guard(timers.gather);

                // Copy points to local buffer
                std::memcpy(
                    local_points.strengths,
                    input.strengths + point_block_boundaries[i],
                    block_num_points * sizeof(float) * 2);
                for (std::size_t dim = 0; dim < Dim; ++dim) {
                    std::memcpy(
                        local_points.coordinates[dim],
                        input.coordinates[dim] + point_block_boundaries[i],
                        block_num_points * sizeof(float));
                }

                auto num_points_padded = finufft::spreading::round_to_next_multiple(
                    block_num_points, spread_subproblem.num_points_multiple());

                // Pad the input points to the required multiple, using a pad coordinate derived
                // from the subgrid. The pad coordinate is given by the leftmost valid coordinate in
                // the subgrid.
                std::array<T, Dim> pad_coordinate;
                for (std::size_t i = 0; i < Dim; ++i) {
                    pad_coordinate[i] =
                        padding_info[i].min_valid_value(grid.offsets[i], grid.extents[i]);
                }
                finufft::spreading::pad_nu_point_collection(
                    local_points, num_points_padded, pad_coordinate);
            }

            // Spread to local subgrid
            {
                finufft::ScopedTimerGuard guard(timers.subproblem);
                spread_subproblem(local_points, grid, subgrid_output.get());
            }

            // Accumulate to main grid
            {
                finufft::ScopedTimerGuard guard(timers.accumulate);
                accumulate_subgrid(subgrid_output.get(), grid);
            }
        }
    }
}

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
make_bin_grids(reference::IntGridBinInfo<T, 2> const &info) {
    std::vector<finufft::spreading::grid_specification<2>> grids(info.num_bins_total());

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
    auto bin_idx = finufft::allocate_aligned_array<uint32_t>(points.num_points, 64);

    auto spread_subproblem = get_subproblem_polynomial_avx512_functor<T, Dim>(kernel_spec);
    auto grid_size = get_grid_size<T, Dim>();

    reference::IntGridBinInfo<T, Dim> info(size, grid_size, spread_subproblem.target_padding());

    {
        finufft::ScopedTimerGuard guard(timer.sort_packed);
        sort_packed<T, Dim>(points, points_sorted, bin_idx.get(), info, timer.sort_packed_timers);
    }

    // Compute bin boundaries
    auto bin_boundaries =
        finufft::allocate_aligned_array<std::size_t>(info.num_bins_total() + 1, 64);

    {
        finufft::ScopedTimerGuard guard(timer.compute_bin_boundaries);

        // Note: can we make this faster / fuse into unpack step?
        // Currently ~8% of total time!!
        std::memset(bin_boundaries.get(), 0, sizeof(std::size_t) * (info.num_bins_total() + 1));
        for (std::size_t i = 0; i < points.num_points; ++i) {
            bin_boundaries[bin_idx[i] + 1] += 1;
        }

        std::partial_sum(
            bin_boundaries.get(),
            bin_boundaries.get() + info.num_bins_total() + 1,
            bin_boundaries.get());
    }

    // Spread points
    timer.make_grids.start();
    auto const &grids = make_bin_grids(info);
    timer.make_grids.end();

    {
        finufft::ScopedTimerGuard guard(timer.spread_blocked);
        spread_blocked(
            points_sorted,
            info.num_bins_total(),
            grids,
            bin_boundaries.get(),
            size,
            output,
            spread_subproblem,
            get_reference_block_locking_accumulator<T, Dim>(),
            timer.spread_blocked_timers);
    }
}

template void spread<float, 2>(
    nu_point_collection<2, const float> const &, kernel_specification const &,
    std::array<std::size_t, 2> const &, float *, SpreadTimers &);

void bm_spread_2d(benchmark::State &state) {
    std::size_t target_size = state.range(0);
    std::size_t kernel_width= state.range(1);

    auto num_points = target_size * target_size;

    finufft::TimerRoot root("bench_full_spread");
    auto timer = root.make_timer("full_spread");
    SpreadTimers timers(timer);

    auto points = make_random_point_collection<2, float>(num_points, 0, {-3 * M_PI, 3 * M_PI});
    auto kernel_spec = specification_from_width(kernel_width, 2);
    auto output = finufft::allocate_aligned_array<float>(2 * target_size * target_size, 64);

    for (auto _ : state) {
        spread<float, 2>(points, kernel_spec, {target_size, target_size}, output.get(), timers);
        benchmark::DoNotOptimize(points.coordinates[0]);
        benchmark::DoNotOptimize(points.coordinates[1]);
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
