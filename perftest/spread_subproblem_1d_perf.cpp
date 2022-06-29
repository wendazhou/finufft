#include <benchmark/benchmark.h>

#include "../src/kernels/avx512/spread_axv512.h"
#include "../src/kernels/reference/spreading_reference.h"
#include "../src/spreading.h"
#include "../test/spread_test_utils.h"

namespace fs = finufft::spreading;

namespace {

template <typename T, std::size_t Dim>
void benchmark_spread_subroblem(
    benchmark::State &state, fs::SpreadSubproblemFunctor<T, Dim> const &functor) {
    auto num_points = state.range(0);
    num_points = fs::round_to_next_multiple(num_points, functor.num_points_multiple());

    // Currently use density with ~ 1 uniform point / 1 non-uniform point.
    auto extent = num_points;
    extent = fs::round_to_next_multiple(extent, functor.extent_multiple());

    // Fill grid specification
    fs::grid_specification<Dim> grid;
    std::fill(grid.offsets.begin(), grid.offsets.end(), 3);
    std::fill(grid.extents.begin(), grid.extents.end(), extent);

    auto padding = functor.target_padding();

    auto input =
        make_spread_subproblem_input<T>(num_points, 0, grid, padding.first, padding.second);
    sort_point_collection(input);

    auto output = fs::allocate_aligned_array<T>(2 * grid.num_elements(), 64);

    auto input_view = input.cast(fs::UniqueArrayToConstPtr{});

    for (auto _ : state) {
        benchmark::DoNotOptimize(input_view.strengths[0]);
        functor(input_view, grid, output.get());
        benchmark::DoNotOptimize(output[0]);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_points);
}

template <typename T, std::size_t Dim, typename Factory>
void benchmark_for_width(benchmark::State &state, Factory const &factory) {
    auto kernel_spec = specification_from_width(state.range(1), 2.0);
    benchmark_spread_subroblem<T, Dim>(state, factory(kernel_spec));
}

template <typename T, std::size_t Dim> void benchmark_legacy(benchmark::State &state) {
    benchmark_for_width<T, Dim>(state, [](fs::kernel_specification const &kernel_spec) {
        return fs::SpreadSubproblemLegacyFunctor{kernel_spec};
    });
}

template <typename T, std::size_t Dim> void benchmark_direct(benchmark::State &state) {
    benchmark_for_width<T, Dim>(state, [](fs::kernel_specification const &kernel_spec) {
        return fs::SpreadSubproblemDirectReference{kernel_spec};
    });
}

template <typename T, std::size_t Dim> void benchmark_reference(benchmark::State &state) {
    benchmark_for_width<T, Dim>(state, [](fs::kernel_specification const &kernel_spec) {
        return fs::get_subproblem_polynomial_reference_functor<T, Dim>(kernel_spec);
    });
}

template <typename T, std::size_t Dim> void benchmark_avx512(benchmark::State &state) {
    benchmark_for_width<T, Dim>(state, [](fs::kernel_specification const &kernel_spec) {
        return fs::get_subproblem_polynomial_avx512_functor<T, Dim>(kernel_spec);
    });
}

} // namespace

BENCHMARK(benchmark_avx512<float, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_reference<float, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_legacy<float, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_direct<float, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_reference<double, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_legacy<double, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_direct<double, 1>)
    ->ArgsProduct({{1 << 12}, {4, 5, 6, 7, 8}})
    ->Unit(benchmark::kMicrosecond);
