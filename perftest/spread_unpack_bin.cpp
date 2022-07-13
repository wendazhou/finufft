/** @file
 *
 * Micro-benchmarks for unpacking.
 *
 */

#include <benchmark/benchmark.h>

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
        benchmark::DoNotOptimize(packed[0]);
        benchmark::DoNotOptimize(unpacked.coordinates[0][0]);
        benchmark::DoNotOptimize(unpacked.coordinates[1][0]);
        benchmark::DoNotOptimize(bin_idx[0]);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * sizeof(PointBin<T, Dim>));
}

void bm_pack_reference(benchmark::State &state) {
    benchmark_unpack<float, 2>(
        state, &finufft::spreading::reference::unpack_bins_to_points<float, 2>);
}

} // namespace

BENCHMARK(bm_pack_reference)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMicrosecond);
