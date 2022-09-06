#pragma once

#include <array>
#include <cstring>
#include <numeric>

#include "../src/kernels/reference/spread_subproblem_reference.h"
#include "../src/kernels/spreading.h"

#include "spread_test_utils.h"

namespace finufft {
namespace spreading {
namespace testing {

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

    std::array<std::size_t, Dim> extent_multiple;
    {
        auto fn1_extent_multiple = fn1.extent_multiple();
        auto fn2_extent_multiple = fn2.extent_multiple();
        for (std::size_t i = 0; i < Dim; ++i) {
            extent_multiple[i] = std::lcm(fn1_extent_multiple[i], fn2_extent_multiple[i]);
        }
    }

    num_points = finufft::round_to_next_multiple(num_points, num_points_multiple);
    for (std::size_t i = 0; i < Dim; ++i) {
        grid.extents[i] = finufft::round_to_next_multiple(grid.extents[i], extent_multiple[i]);
    }
}

/** Adjusts grid padding to be able to be able to call both reference
 * and target implementation on the same grid specification.
 *
 */
template <typename T, std::size_t Dim>
void unify_padding(
    tcb::span<finufft::spreading::KernelWriteSpec<T>, Dim> target_padding,
    tcb::span<const finufft::spreading::KernelWriteSpec<T>, Dim> reference_padding) {
    for (std::size_t i = 0; i < Dim; ++i) {
        if (target_padding[i].offset != reference_padding[i].offset) {
            // don't try to unify different offsets yet (they probably correspond to different
            // problems).
            ADD_FAILURE() << "Target padding offset mismatch for dimension " << i << ": "
                          << target_padding[i].offset << " != " << reference_padding[i].offset;
        }

        target_padding[i].grid_left =
            std::max(target_padding[i].grid_left, reference_padding[i].grid_left);
        target_padding[i].grid_right =
            std::max(target_padding[i].grid_right, reference_padding[i].grid_right);
    }
}

template <typename T, std::size_t Dim> struct evaluation_result {
    finufft::aligned_unique_array<T> output_reference;
    finufft::aligned_unique_array<T> output;
    finufft::spreading::grid_specification<Dim> grid;
};

/** Evaluates an implementation of the subproblem spread compared to the reference implementation.
 *
 * @param fn_factory Target implementation functor factory
 * @param num_points Number of points to spread
 * @param grid Grid specification - may be modified to accomodate requirements of implementations
 * @param width Width of the kernel, other parameters of kernel are derived accordingly
 * @param point_factory Callable which creates the points to spread
 * 
 */
template <std::size_t Dim, typename T, typename Fn, typename PointFactory>
evaluation_result<T, Dim> evaluate_subproblem_implementation(
    Fn &&fn_factory, std::size_t num_points, finufft::spreading::subgrid_specification<Dim> grid,
    int width, PointFactory &&point_factory) {
    // Get the kernel specification
    auto kernel_spec = specification_from_width(width, 2.0);

    auto reference_fn =
        finufft::spreading::get_subproblem_polynomial_reference_functor<T, Dim>(kernel_spec);
    auto fn = fn_factory(kernel_spec);

    // Adjust grid and number of points according to padding requirements of target.
    std::size_t num_points_initial = num_points;
    finufft::spreading::testing::adjust_problem_parameters(num_points, grid, fn, reference_fn);
    std::array<finufft::spreading::KernelWriteSpec<T>, Dim> const &padding_ref =
        reference_fn.target_padding();
    std::array<finufft::spreading::KernelWriteSpec<T>, Dim> padding = fn.target_padding();
    unify_padding<T, Dim>(padding, padding_ref);

    // Create subproblem input.
    // Note that input is not sorted as this is functionality, not performance test.
    auto input = point_factory(num_points, grid, padding);

    std::memset(
        input.strengths + 2 * num_points_initial,
        0,
        2 * sizeof(T) * (num_points - num_points_initial));

    // Allocate output arrays.
    auto output = finufft::allocate_aligned_array<T>(2 * grid.max_elements(), 64);
    auto output_reference = finufft::allocate_aligned_array<T>(2 * grid.max_elements(), 64);

    // Zero output arrays.
    std::memset(output.get(), 0, 2 * sizeof(T) * grid.max_elements());
    std::memset(output_reference.get(), 0, 2 * sizeof(T) * grid.max_elements());

    // Invoke reference and target implementation.
    reference_fn(input, grid, output_reference.get());
    fn(input, grid, output.get());

    return {std::move(output_reference), std::move(output), grid};
}

} // namespace testing

} // namespace spreading

} // namespace finufft
