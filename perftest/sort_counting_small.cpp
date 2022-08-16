#include <benchmark/benchmark.h>

#include "../src/kernels/avx512/sort_bin_counting.h"
#include "../src/kernels/avx512/spread_bin_sort_int.h"
#include "../src/kernels/reference/sort_bin_counting.h"

#include "../test/spread_test_utils.h"

namespace {

template <typename T, std::size_t Dim>
void benchmark_sort(
    benchmark::State &state, finufft::spreading::SortPointsFunctor<T, Dim> const &fn,
    finufft::spreading::IntBinInfo<T, Dim> const &info) {
    auto num_points = state.range(0);

    auto points = make_random_point_collection<Dim, T>(num_points, 0, {-M_PI, M_PI});
    auto output = finufft::spreading::SpreaderMemoryInput<Dim, T>(num_points);
    auto num_points_per_bin =
        finufft::allocate_aligned_array<std::size_t>(info.num_bins_total(), 64);

    for (auto _ : state) {
        fn(points,
           finufft::spreading::FoldRescaleRange::Pi,
           output,
           num_points_per_bin.get(),
           info);
        benchmark::DoNotOptimize(output.coordinates[0]);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * sizeof(T) * (2 + Dim));
}

template <typename T>
void benchmark_sort_2d_small(
    benchmark::State &state, finufft::spreading::SortPointsFunctor<T, 2> const &fn) {
    finufft::spreading::IntBinInfo<T, 2> info({1024, 1024}, {128, 32}, {4, 4});
    benchmark_sort(state, fn, info);
}

void bm_counting_direct(benchmark::State &state) {
    benchmark_sort_2d_small<float>(
        state,
        &finufft::spreading::reference::nu_point_counting_sort_direct_singlethreaded<float, 2>);
}

void bm_counting_blocked(benchmark::State &state) {
    benchmark_sort_2d_small<float>(
        state,
        &finufft::spreading::reference::nu_point_counting_sort_blocked_singlethreaded<float, 2>);
}

void bm_counting_direct_avx512(benchmark::State &state) {
    benchmark_sort_2d_small<float>(
        state,
        &finufft::spreading::avx512::nu_point_counting_sort_direct_singlethreaded<float, 2>);
}

void bm_ips4o_packed(benchmark::State &state) {
    benchmark_sort_2d_small<float>(state, finufft::spreading::avx512::get_sort_functor<float, 2>());
}

} // namespace

BENCHMARK(bm_counting_direct)->Range(1 << 16, 1 << 24)->Unit(benchmark::kMillisecond);
BENCHMARK(bm_counting_direct_avx512)->Range(1 << 16, 1 << 24)->Unit(benchmark::kMillisecond);
BENCHMARK(bm_counting_blocked)->Range(1 << 16, 1 << 24)->Unit(benchmark::kMillisecond);
BENCHMARK(bm_ips4o_packed)->Range(1 << 16, 1 << 24)->Unit(benchmark::kMillisecond);
