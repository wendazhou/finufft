/** @file
 *
 * Micro-benchmarks for packing and bin index computation.
 *
 */

#include <benchmark/benchmark.h>

#include "../test/spread_test_utils.h"

#include "../src/kernels/avx512/spread_bin_sort_int.h"
#include "../src/kernels/reference/spread_bin_sort_int.h"

using namespace finufft::spreading;

namespace {

template <typename T, std::size_t Dim, typename Fn>
void benchmark_pack(benchmark::State &state, Fn &&compute_bins_and_pack) {
    auto num_points = state.range(0);
    auto points = testing::make_random_point_collection<Dim, T>(num_points, 0, {-3 * M_PI, 3 * M_PI});

    auto packed = finufft::allocate_aligned_array<PointBin<T, Dim>>(points.num_points, 64);

    std::array<std::size_t, Dim> target_size;
    target_size.fill(4096);

    std::array<std::size_t, Dim> bin_size;
    bin_size.fill(32);
    bin_size[0] = 128;

    std::array<T, Dim> offset;
    offset.fill(4);

    IntBinInfo<T, Dim> info(target_size, bin_size, offset);

    for (auto _ : state) {
        compute_bins_and_pack(points, FoldRescaleRange::Pi, info, packed.get());
        benchmark::DoNotOptimize(points.coordinates[0][0]);
        benchmark::DoNotOptimize(packed[0]);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * sizeof(PointBin<T, Dim>));
}

void bm_pack_reference(benchmark::State &state) {
    benchmark_pack<float, 2>(state, &finufft::spreading::reference::compute_bins_and_pack<float, 2>);
}

void bm_pack_avx512(benchmark::State &state) {
    benchmark_pack<float, 2>(state, &finufft::spreading::avx512::compute_bins_and_pack<float, 2>);
}

} // namespace

BENCHMARK(bm_pack_reference)->RangeMultiplier(4)->Range(1 << 10, 1 << 24)->Unit(benchmark::kMicrosecond);
BENCHMARK(bm_pack_avx512)->RangeMultiplier(4)->Range(1 << 10, 1 << 24)->Unit(benchmark::kMicrosecond);
