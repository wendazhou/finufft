/** Benchmark to investigate the performance of spread subproblem as a function of grid size.
 * 
 * In the subproblem evaluation, the size of the grid under consideration is of crucial importance
 * due to the fact that we require repeated reads and writes to it.
 * This sequence of benchmark explores the performance of the subproblem as a function of that grid size.
 * 
 */

#include "../src/kernels/avx512/spread_avx512.h"
#include "spread_subproblem_perf.h"

#include <benchmark/benchmark.h>

namespace {

void benchmark_scaling_avx512(benchmark::State &state) {
    auto width = state.range(0);
    auto total_points = state.range(1);
    auto dim_y = state.range(2);

    auto kernel_spec = specification_from_width(width, 2.0);
    auto functor = fs::get_subproblem_polynomial_avx512_functor<float, 2>(kernel_spec);

    auto padding = functor.target_padding();

    fs::grid_specification<2> grid_spec;
    grid_spec.offsets.fill(0);
    grid_spec.extents[0] =
        fs::round_to_next_multiple(total_points / dim_y, functor.extent_multiple()[0]);
    grid_spec.extents[1] = fs::round_to_next_multiple(dim_y, functor.extent_multiple()[1]);

    auto num_points = fs::round_to_next_multiple(total_points, functor.num_points_multiple());

    benchmark_spread_subproblem<float, 2>(state, functor, num_points, grid_spec, false);
}

} // namespace

BENCHMARK(benchmark_scaling_avx512)
    ->ArgsProduct({{4, 8}, {1 << 10, 1 << 11, 3 * 1 << 10, 1 << 12, 1 << 13, 1 << 14}, {16, 24, 32, 48}});
