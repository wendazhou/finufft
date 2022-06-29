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
evaluation_result<Dim, T> evaluate_subproblem_implementation(
    Fn &&fn_factory, std::size_t num_points, uint32_t seed, int width) {
    // Get the kernel specification
    auto kernel_spec = specification_from_width(width, 2.0);

    auto reference_fn = finufft::spreading::SpreadSubproblemLegacyFunctor{kernel_spec};
    auto fn = fn_factory(kernel_spec);

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

template <typename T, std::size_t Dim, typename Fn>
void test_subproblem_implementation(int width, Fn &&factory) {
    auto result = evaluate_subproblem_implementation<Dim, T>(factory, 100, 0, width);
    auto error_level = compute_max_relative_threshold(
        std::pow(10, -width + 1),
        result.output_reference.get(),
        result.output_reference.get() + 2 * result.grid.num_elements());

    for (std::size_t i = 0; i < 2 * result.grid.num_elements(); ++i) {
        ASSERT_NEAR(result.output_reference[i], result.output[i], error_level)
            << "i = " << i << "; "
            << "width = " << width;
    }
}

} // namespace

TEST(SpreadSubproblem, SpreadSubproblem1df32) {
    test_subproblem_implementation<float, 1>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemLegacyFunctor{k};
        });
}

TEST(SpreadSubproblem, ReferenceDirect1Df32) {
    test_subproblem_implementation<float, 1>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference{k};
        });
}

TEST(SpreadSubproblem, ReferenceDirect1Df64) {
    test_subproblem_implementation<double, 1>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference{k};
        });
}

TEST(SpreadSubproblem, ReferenceDirect2Df32) {
    test_subproblem_implementation<float, 2>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference{k};
        });
}

TEST(SpreadSubproblem, ReferenceDirect2Df64) {
    test_subproblem_implementation<double, 2>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference{k};
        });
}

TEST(SpreadSubproblem, ReferenceDirect3Df32) {
    test_subproblem_implementation<float, 3>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference{k};
        });
}

TEST(SpreadSubproblem, ReferenceDirect3Df64) {
    test_subproblem_implementation<double, 3>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference{k};
        });
}

TEST(SpreadSubproblem, ReferencePoly1Df32) {
    test_subproblem_implementation<float, 1>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_reference_functor<float, 1>(k);
        });
}

TEST(SpreadSubproblem, ReferencePoly1Df64) {
    test_subproblem_implementation<double, 1>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_reference_functor<double, 1>(k);
        });
}

TEST(SpreadSubproblem, ReferencePoly2Df32) {
    test_subproblem_implementation<float, 2>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_reference_functor<float, 2>(k);
        });
}

TEST(SpreadSubproblem, ReferencePoly2Df64) {
    test_subproblem_implementation<double, 2>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_reference_functor<double, 2>(k);
        });
}

TEST(SpreadSubproblem, ReferencePoly3Df32) {
    test_subproblem_implementation<float, 3>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_reference_functor<float, 3>(k);
        });
}

TEST(SpreadSubproblem, ReferencePoly3Df64) {
    test_subproblem_implementation<double, 3>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_reference_functor<double, 3>(k);
        });
}
