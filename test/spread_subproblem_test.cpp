#include "../src/spreading.h"

#include <numeric>

#include <gtest/gtest.h>

#include "spread_test_utils.h"

namespace finufft {
namespace spreadinterp {
int setup_spreader(
    finufft_spread_opts &opts, double eps, double upsampfac, int kerevalmeth, int debug,
    int showwarn, int dim);
}

} // namespace finufft

namespace {

template <typename T, std::size_t Dim>
finufft::spreading::nu_point_collection<Dim, finufft::spreading::aligned_unique_array<T>>
make_spread_subproblem_input(
    std::size_t num_points, uint32_t seed, finufft::spreading::grid_specification<Dim> const &grid,
    int kernel_width) {
    std::array<std::pair<T, T>, Dim> range;

    for (std::size_t i = 0; i < Dim; ++i) {
        range[i].first = grid.offsets[i] + 0.5 * kernel_width;
        range[i].second = grid.offsets[i] + grid.extents[i] - 0.5 * kernel_width - 1;
    }

    return make_random_point_collection<Dim, T>(num_points, seed, range);
}

template <std::size_t Dim, typename Fn1, typename Fn2>
void adjust_problem_parameters(
    std::size_t &num_points, finufft::spreading::grid_specification<Dim> &grid, Fn1 const &fn1,
    Fn2 const &fn2) {
    auto num_points_multiple = std::lcm(fn1.num_points_multiple, fn2.num_points_multiple);
    auto extent_multiple = std::lcm(fn1.extent_multiple, fn2.extent_multiple);

    num_points = finufft::spreading::round_to_next_multiple(num_points, num_points_multiple);
    for (std::size_t i = 0; i < Dim; ++i) {
        grid.extents[i] =
            finufft::spreading::round_to_next_multiple(grid.extents[i], extent_multiple);
    }
}

template <std::size_t Dim, typename T> struct evaluation_result {
    finufft::spreading::aligned_unique_array<T> output_reference;
    finufft::spreading::aligned_unique_array<T> output;
    finufft::spreading::grid_specification<Dim> grid;
};

template <std::size_t Dim, typename T, typename Fn>
evaluation_result<Dim, T> evaluate_subproblem_implementation(Fn &&fn, uint32_t seed) {
    // Get the kernel specification
    finufft_spread_opts opts;
    finufft::spreadinterp::setup_spreader(opts, 1e-5, 2, 1, 0, 0, 1);
    finufft::spreading::kernel_specification kernel{opts.ES_c, opts.ES_beta, opts.nspread};

    auto reference_fn = finufft::spreading::spread_subproblem_reference;

    auto offset = 3;
    auto size = 20;

    std::size_t num_points = 100;
    finufft::spreading::grid_specification<Dim> grid;
    for (std::size_t i = 0; i < Dim; ++i) {
        grid.offsets[i] = offset;
        grid.extents[i] = size;
    }

    adjust_problem_parameters(num_points, grid, fn, reference_fn);

    auto input = make_spread_subproblem_input<T>(100, seed, grid, opts.nspread);

    auto output = finufft::spreading::allocate_aligned_array<T>(2 * grid.num_elements(), 64);
    auto output_reference =
        finufft::spreading::allocate_aligned_array<T>(2 * grid.num_elements(), 64);

    reference_fn(
        input.cast(finufft::spreading::UniqueArrayToConstPtr{}),
        grid,
        output_reference.get(),
        kernel);
    fn(input.cast(finufft::spreading::UniqueArrayToConstPtr{}), grid, output.get(), kernel);

    return {std::move(output_reference), std::move(output), grid};
}

} // namespace

TEST(SpreadSubproblem, SpreadSubproblem1df32) {
    // Get the kernel specification
    auto result = evaluate_subproblem_implementation<1, float>(
        finufft::spreading::spread_subproblem_reference, 0);

    for (std::size_t i = 0; i < 2 * result.grid.num_elements(); ++i) {
        EXPECT_NEAR(result.output_reference[i], result.output[i], std::abs(result.output[i]) * 1e-3);
    }
}


TEST(SpreadSubproblem, SpreadSubproblem1df64) {
    // Get the kernel specification
    auto result = evaluate_subproblem_implementation<1, double>(
        finufft::spreading::spread_subproblem_reference, 0);

    for (std::size_t i = 0; i < 2 * result.grid.num_elements(); ++i) {
        EXPECT_NEAR(result.output_reference[i], result.output[i], std::abs(result.output[i]) * 1e-3);
    }
}
