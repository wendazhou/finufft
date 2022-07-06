#include <benchmark/benchmark.h>

#include "../src/tracing.h"

#include "../src/kernels/hwy/spread_bin_sort_hwy.h"
#include "../src/kernels/legacy/spread_bin_sort_legacy.h"
#include "../src/kernels/reference/spread_bin_sort_reference.h"
#include "../src/spreading.h"

#include "../test/spread_test_utils.h"

namespace {
template <typename T, std::size_t Dim>
void benchmark_binsort(
    benchmark::State &state, finufft::spreading::BinSortFunctor<T, Dim> &&functor) {
    std::size_t num_points = std::size_t(1) << state.range(0);

    auto points = make_random_point_collection<Dim, T>(num_points, 0, {-3 * M_PI, 3 * M_PI});
    finufft::spreading::nu_point_collection<Dim, const T> points_view = points;

    auto output = finufft::allocate_aligned_array<int64_t>(points.num_points, 64);

    std::array<T, Dim> extents;
    extents.fill(1024);

    std::array<T, Dim> bin_sizes;
    bin_sizes.fill(16);

    for (auto _ : state) {
        functor(
            output.get(),
            points.num_points,
            points_view.coordinates,
            extents,
            bin_sizes,
            finufft::spreading::FoldRescaleRange::Pi);
    }

    state.SetItemsProcessed(state.iterations() * points.num_points);
}

template <typename T, std::size_t Dim> void bm_binsort_legacy(benchmark::State &state) {
    benchmark_binsort(state, finufft::spreading::get_bin_sort_functor_legacy<T, Dim>());
}

template <typename T, std::size_t Dim> void bm_binsort_reference(benchmark::State &state) {
    benchmark_binsort(state, finufft::spreading::get_bin_sort_functor_reference<T, Dim>());
}

template <typename T, std::size_t Dim> void bm_binsort_highway(benchmark::State &state) {
    finufft::TimerRoot root("benchmark_binsort_highway");
    auto timer = root.make_timer("");
    benchmark_binsort(state, finufft::spreading::highway::get_bin_sort_functor<T, Dim>(&timer));

    auto results = root.report("/binsort_highway");
    for (auto &name_and_time : results) {
        auto name = std::get<0>(name_and_time);

        if (name.empty()) {
            continue;
        }

        auto time = std::chrono::duration<double>(std::get<1>(name_and_time)).count();
        if (time == 0) {
            continue;
        }

        state.counters[name] = benchmark::Counter(time / state.iterations());
    }
}

} // namespace

#define MAKE_BENCHMARKS(fn, type)                                                                  \
    BENCHMARK(fn<type, 1>)->Arg(20)->Arg(24)->UseRealTime()->Unit(benchmark::kMillisecond);        \
    BENCHMARK(fn<type, 2>)->Arg(20)->Arg(24)->UseRealTime()->Unit(benchmark::kMillisecond);        \
    BENCHMARK(fn<type, 3>)->Arg(20)->Arg(24)->UseRealTime()->Unit(benchmark::kMillisecond);

MAKE_BENCHMARKS(bm_binsort_highway, float)
MAKE_BENCHMARKS(bm_binsort_reference, float)
MAKE_BENCHMARKS(bm_binsort_legacy, float)

MAKE_BENCHMARKS(bm_binsort_highway, double)
MAKE_BENCHMARKS(bm_binsort_reference, double)
MAKE_BENCHMARKS(bm_binsort_legacy, double)

#undef MAKE_BENCHMARKS
