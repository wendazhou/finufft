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
#include <libdivide.h>

#include "../src/kernels/reference/gather_fold_reference.h"
#include "../src/kernels/reference/spread_bin_sort_int.h"

#include "../src/tracing.h"

#include <benchmark/benchmark.h>

#include "../test/spread_test_utils.h"

using namespace finufft::spreading;
using namespace finufft::spreading::reference;

namespace {

/** Computation of bin grid index based on integer indices, accelerated by libdivide
 *
 *
 */
template <typename T, std::size_t Dim>
std::size_t compute_bin_index_single_dividers(
    std::array<T, Dim> const &coords, IntBinInfo<T, Dim> const &info,
    std::array<libdivide::divider<uint32_t>, Dim> const &dividers) {
    std::size_t bin_index = 0;

    for (std::size_t j = 0; j < Dim; ++j) {
        auto coord = coords[j];
        auto coord_grid_left = std::ceil(coord - info.offset[j]);
        std::uint32_t coord_grid = std::uint32_t(coord_grid_left - info.global_offset[j]);

        std::size_t bin_index_j = coord_grid / dividers[j];

        bin_index += bin_index_j * info.bin_index_stride[j];
    }

    return bin_index;
}

template <typename T, std::size_t Dim, typename FoldRescale> struct ComputeAndPackSingle {
    IntBinInfo<T, Dim> const &info;
    FoldRescale fold_rescale;
    std::array<T, Dim> extents_f;
    std::array<libdivide::divider<uint32_t>, Dim> dividers;

    ComputeAndPackSingle(IntBinInfo<T, Dim> const &info, FoldRescale fold_rescale)
        : info(info), fold_rescale(fold_rescale) {
        std::copy(info.size.begin(), info.size.end(), extents_f.begin());
        for (std::size_t j = 0; j < Dim; ++j) {
            dividers[j] = libdivide::divider<uint32_t>(info.bin_size[j]);
        }
    }

    PointBin<T, Dim> operator()(nu_point_collection<Dim, const T> const& input, std::size_t i) const {
        PointBin<T, Dim> p;
        p.bin = 0;

        for (std::size_t j = 0; j < Dim; ++j) {
            auto coord = fold_rescale(input.coordinates[j][i], extents_f[j]);

            // Pack basic data
            p.coordinates[j] = coord;
            p.strength[0] = input.strengths[2 * i];
            p.strength[1] = input.strengths[2 * i + 1];

            // Compute bin index
            auto coord_grid_left = std::ceil(coord - info.offset[j]);
            std::uint32_t coord_grid = std::uint32_t(coord_grid_left - info.global_offset[j]);
            std::size_t bin_index_j = coord_grid / dividers[j];
            p.bin += bin_index_j * info.bin_index_stride[j];
        }

        return p;
    }
};

/** Computes bin index, and collects folded coordinates into
 * given buffer.
 *
 * @param[out] points_with_bin The buffer to write the folded coordinates to, of length
 * `num_points`
 * @param coordinates The points to fold and bin
 * @param fold_rescale Fold-rescale functor to use
 * @param info Bin information
 *
 */
template <typename T, std::size_t Dim, typename FoldRescale>
void compute_bins_and_pack_impl(
    PointBin<T, Dim> *points_with_bin, nu_point_collection<Dim, const T> input,
    FoldRescale &&fold_rescale, IntBinInfo<T, Dim> const &info) {

    ComputeAndPackSingle<T, Dim, FoldRescale> compute_and_pack(info, fold_rescale);

    for (std::size_t i = 0; i < input.num_points; ++i) {
        points_with_bin[i] = compute_and_pack(input, i);
    }
}

template <typename T, std::size_t Dim>
void compute_bins_and_pack(
    nu_point_collection<Dim, const T> input, FoldRescaleRange range, IntBinInfo<T, Dim> const &info,
    PointBin<T, Dim> *output) {
    if (range == FoldRescaleRange::Pi) {
        compute_bins_and_pack_impl(output, input, FoldRescalePi<T>{}, info);
    } else {
        compute_bins_and_pack_impl(output, input, FoldRescaleIdentity<T>{}, info);
    }
}

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
    nu_point_collection<Dim, const T> points, nu_point_collection<Dim, T> output,
    IntGridBinInfo<T, Dim> const &info, SortPackedTimers &timers) {

    auto packed = finufft::allocate_aligned_array<PointBin<T, Dim>>(points.num_points, 64);

    // Compute bins
    {
        finufft::ScopedTimerGuard guard(timers.pack);
        compute_bins_and_pack(points, FoldRescaleRange::Pi, info, packed.get());
    }

    {
        finufft::ScopedTimerGuard guard(timers.sort);
        ips4o::parallel::sort(packed.get(), packed.get() + points.num_points);
    }

    // Unpack to output.
    {
        finufft::ScopedTimerGuard guard(timers.unpack);
        for (std::size_t i = 0; i < points.num_points; ++i) {
            auto const &p = packed[i];

            for (std::size_t j = 0; j < Dim; ++j) {
                output.coordinates[j][i] = p.coordinates[j];
            }

            output.strengths[2 * i] = p.strength[0];
            output.strengths[2 * i + 1] = p.strength[1];
        }
    }
}

template <typename T, std::size_t Dim> void bench_sort_packed(benchmark::State &state) {
    auto num_points = state.range(0);

    auto points = make_random_point_collection<Dim, T>(num_points, 1, {-3 * M_PI, 3 * M_PI});
    auto output = finufft::spreading::SpreaderMemoryInput<Dim, T>(num_points);

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
        sort_packed<T, Dim, FoldRescalePi<T>>(points, output, info, timers);
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
