#pragma once

#include <array>
#include <numeric>

#include "../src/kernels/spreading.h"

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

} // namespace testing

} // namespace spreading

} // namespace finufft
