#include "../src/kernels/reference/spreading_reference.h"
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
    T padding_left, T padding_right) {
    std::array<std::pair<T, T>, Dim> range;

    for (std::size_t i = 0; i < Dim; ++i) {
        range[i].first = grid.offsets[i] + padding_left;
        range[i].second = grid.offsets[i] + grid.extents[i] - padding_right - 1;
    }

    return make_random_point_collection<Dim, T>(num_points, seed, range);
}

/** Adjusts the number of points and the extents of the grid according to alignment requirements.
 *
 * Implementations may specify alignment requirements, so that the total number of points and the
 * dimensions of the grid are multiples of some specified value. This function adjusts the given
 * specification to match these requirements.
 *
 */
template <std::size_t Dim, typename Fn1, typename Fn2>
void adjust_problem_parameters(
    std::size_t &num_points, finufft::spreading::grid_specification<Dim> &grid, Fn1 const &fn1,
    Fn2 const &fn2) {
    auto num_points_multiple = std::lcm(fn1.num_points_multiple(), fn2.num_points_multiple());
    auto extent_multiple = std::lcm(fn1.extent_multiple(), fn2.extent_multiple());

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

/** Evaluates an implementation of the subproblem spread compared to the reference implementation.
 *
 */
template <std::size_t Dim, typename T, typename Fn>
evaluation_result<Dim, T>
evaluate_subproblem_implementation(Fn &&fn_factory, std::size_t num_points, uint32_t seed) {
    // Get the kernel specification
    finufft_spread_opts opts;
    finufft::spreadinterp::setup_spreader(opts, 1e-5, 2, 0, 0, 0, 1);
    finufft::spreading::kernel_specification kernel{opts.ES_beta, opts.nspread};

    auto reference_fn = finufft::spreading::SpreadSubproblemLegacyFunctor{kernel};
    auto fn = fn_factory(kernel);

    // Arbitrary grid specification in all dimensions
    auto offset = 3;
    auto size = 20;

    finufft::spreading::grid_specification<Dim> grid;
    for (std::size_t i = 0; i < Dim; ++i) {
        grid.offsets[i] = offset;
        grid.extents[i] = size;
    }

    // Adjust grid and number of points according to padding requirements of target.
    adjust_problem_parameters(num_points, grid, fn, reference_fn);
    auto padding_ref = reference_fn.target_padding();
    auto padding = fn.target_padding();
    padding.first = std::max(padding_ref.first, padding.first);
    padding.second = std::max(padding_ref.second, padding.second);

    // Create subproblem input.
    // Note that input is not sorted as this is functionality, not performance test.
    auto input =
        make_spread_subproblem_input<T>(num_points, seed, grid, padding.first, padding.second);
    std::fill_n(input.strengths.get(), 2 * num_points, 1.0);

    // Allocate output arrays.
    auto output = finufft::spreading::allocate_aligned_array<T>(2 * grid.num_elements(), 64);
    auto output_reference =
        finufft::spreading::allocate_aligned_array<T>(2 * grid.num_elements(), 64);

    auto input_view = input.cast(finufft::spreading::UniqueArrayToConstPtr{});

    reference_fn(input_view, grid, output_reference.get());
    fn(input_view, grid, output.get());

    return {std::move(output_reference), std::move(output), grid};
}

} // namespace

TEST(SpreadSubproblem, SpreadSubproblem1df32) {
    // Get the kernel specification
    auto result = evaluate_subproblem_implementation<1, float>(
        [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemLegacyFunctor{k};
        },
        100,
        0);

    auto error_level = compute_max_relative_threshold(
        1e-5,
        result.output_reference.get(),
        result.output_reference.get() + 2 * result.grid.num_elements());
    for (std::size_t i = 0; i < 2 * result.grid.num_elements(); ++i) {
        EXPECT_NEAR(result.output_reference[i], result.output[i], error_level);
    }
}

TEST(SpreadSubproblem, ReferenceDirect1Df32) {
    // Get the kernel specification
    auto result = evaluate_subproblem_implementation<1, float>(
        [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference{k};
        },
        100,
        0);

    auto error_level = compute_max_relative_threshold(
        1e-5,
        result.output_reference.get(),
        result.output_reference.get() + 2 * result.grid.num_elements());
    for (std::size_t i = 0; i < 2 * result.grid.num_elements(); ++i) {
        ASSERT_NEAR(result.output_reference[i], result.output[i], error_level) << "i = " << i;
    }
}

TEST(SpreadSubproblem, ReferenceDirect1Df64) {
    // Get the kernel specification
    auto result = evaluate_subproblem_implementation<1, double>(
        [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference{k};
        },
        100,
        0);

    auto error_level = compute_max_relative_threshold(
        1e-5,
        result.output_reference.get(),
        result.output_reference.get() + 2 * result.grid.num_elements());
    for (std::size_t i = 0; i < 2 * result.grid.num_elements(); ++i) {
        ASSERT_NEAR(result.output_reference[i], result.output[i], error_level) << "i = " << i;
    }
}

TEST(SpreadSubproblem, ReferenceDirect2Df32) {
    // Get the kernel specification
    auto result = evaluate_subproblem_implementation<2, float>(
        [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference{k};
        },
        1,
        0);

    auto error_level = compute_max_relative_threshold(
        1e-5,
        result.output_reference.get(),
        result.output_reference.get() + 2 * result.grid.num_elements());
    for (std::size_t i = 0; i < 2 * result.grid.num_elements(); ++i) {
        ASSERT_NEAR(result.output_reference[i], result.output[i], error_level) << "i = " << i;
    }
}
