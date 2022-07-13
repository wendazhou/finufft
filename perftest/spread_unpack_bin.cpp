/** @file
 *
 * Micro-benchmarks for unpacking.
 *
 */

#include <benchmark/benchmark.h>

#include <cassert>
#include <cstring>

#include "../test/spread_test_utils.h"

#include "../src/kernels/avx512/spread_bin_sort_int.h"
#include "../src/kernels/reference/spread_bin_sort_int.h"

using namespace finufft::spreading;

namespace {

template <typename T, std::size_t Dim, typename Fn>
void benchmark_unpack(benchmark::State &state, Fn &&unpack) {
    auto num_points = state.range(0);
    auto points = make_random_point_collection<Dim, T>(num_points, 0, {-3 * M_PI, 3 * M_PI});

    auto packed = finufft::allocate_aligned_array<PointBin<T, Dim>>(points.num_points, 64);
    auto unpacked = finufft::spreading::SpreaderMemoryInput<Dim, T>(num_points);
    auto bin_idx = finufft::allocate_aligned_array<std::uint32_t>(num_points, 64);

    std::array<std::size_t, Dim> target_size;
    target_size.fill(4096);

    std::array<std::size_t, Dim> bin_size;
    bin_size.fill(32);
    bin_size[0] = 128;

    std::array<T, Dim> offset;
    offset.fill(4);

    IntBinInfo<T, Dim> info(target_size, bin_size, offset);
    reference::compute_bins_and_pack<float, 2>(points, FoldRescaleRange::Pi, info, packed.get());

    for (auto _ : state) {
        unpack(packed.get(), unpacked, bin_idx.get());
        benchmark::DoNotOptimize(unpacked.coordinates[0][0]);
        benchmark::DoNotOptimize(unpacked.coordinates[1][0]);
        benchmark::DoNotOptimize(bin_idx[0]);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * sizeof(PointBin<T, Dim>));
}

template <typename T, std::size_t Dim, typename Fn>
void benchmark_unpack_sorted(benchmark::State &state, Fn &&unpack) {
    auto num_points = state.range(0);
    auto points = make_random_point_collection<Dim, T>(num_points, 0, {-3 * M_PI, 3 * M_PI});

    auto packed_holder = finufft::allocate_aligned_array<PointBin<T, Dim>>(points.num_points, 64);
    auto packed = packed_holder.get();
    auto unpacked = finufft::spreading::SpreaderMemoryInput<Dim, T>(num_points);

    std::array<std::size_t, Dim> target_size;
    target_size.fill(4096);

    std::array<std::size_t, Dim> bin_size;
    bin_size.fill(32);
    bin_size[0] = 128;

    std::array<T, Dim> offset;
    offset.fill(4);

    IntBinInfo<T, Dim> info(target_size, bin_size, offset);
    reference::compute_bins_and_pack<float, 2>(points, FoldRescaleRange::Pi, info, packed);
    std::sort(packed, packed + points.num_points);

    if (!std::is_sorted(packed, packed + points.num_points)) {
        state.SkipWithError("Points are not sorted");
        return;
    }

    if (std::any_of(packed, packed + points.num_points, [&](const PointBin<T, Dim> &p) {
            return p.bin >= info.num_bins_total();
        })) {
        state.SkipWithError("Invalid bin");
        return;
    }

    auto bin_counts_holder = finufft::allocate_aligned_array<size_t>(info.num_bins_total(), 64);
    auto bin_counts = bin_counts_holder.get();
    std::memset(bin_counts, 0, info.num_bins_total() * sizeof(size_t));

    for (auto _ : state) {
        unpack(packed, unpacked, bin_counts);
        benchmark::DoNotOptimize(unpacked.coordinates[0][0]);
        benchmark::DoNotOptimize(unpacked.coordinates[1][0]);
        benchmark::DoNotOptimize(bin_counts[0]);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * sizeof(PointBin<T, Dim>));
}

void bm_unpack_reference(benchmark::State &state) {
    benchmark_unpack<float, 2>(
        state, &finufft::spreading::reference::unpack_bins_to_points<float, 2>);
}

void bm_unpack_sorted_reference(benchmark::State &state) {
    benchmark_unpack_sorted<float, 2>(
        state, &finufft::spreading::reference::unpack_sorted_bins_to_points<float, 2>);
}

} // namespace

BENCHMARK(bm_unpack_reference)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(bm_unpack_sorted_reference)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMicrosecond);
