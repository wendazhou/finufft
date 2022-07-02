/** @file
 *
 * This file contains unit tests for implementations of the spreading subproblem.
 *
 */

#include "../src/kernels/avx512/spread_avx512.h"
#include "../src/kernels/legacy/spread_subproblem_legacy.h"
#include "../src/kernels/reference/spread_subproblem_reference.h"
#include "../src/spreading.h"

#include "../src/kernels/dispatch.h"

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

    std::array<std::size_t, Dim> extent_multiple;
    {
        auto fn1_extent_multiple = fn1.extent_multiple();
        auto fn2_extent_multiple = fn2.extent_multiple();
        for (std::size_t i = 0; i < Dim; ++i) {
            extent_multiple[i] = std::lcm(fn1_extent_multiple[i], fn2_extent_multiple[i]);
        }
    }

    num_points = finufft::spreading::round_to_next_multiple(num_points, num_points_multiple);
    for (std::size_t i = 0; i < Dim; ++i) {
        grid.extents[i] =
            finufft::spreading::round_to_next_multiple(grid.extents[i], extent_multiple[i]);
    }
}

template <std::size_t Dim, typename T> struct evaluation_result {
    finufft::aligned_unique_array<T> output_reference;
    finufft::aligned_unique_array<T> output;
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

    auto reference_fn =
        finufft::spreading::get_subproblem_polynomial_reference_functor<T, Dim>(kernel_spec);
    auto fn = fn_factory(kernel_spec);

    // Arbitrary grid specification in all dimensions
    auto offset = 3;
    auto size = 24;

    finufft::spreading::grid_specification<Dim> grid;
    for (std::size_t i = 0; i < Dim; ++i) {
        grid.offsets[i] = offset;
        grid.extents[i] = size;
    }

    // Adjust grid and number of points according to padding requirements of target.
    adjust_problem_parameters(num_points, grid, fn, reference_fn);
    std::array<finufft::spreading::KernelWriteSpec<T>, Dim> padding_ref = reference_fn.target_padding();
    std::array<finufft::spreading::KernelWriteSpec<T>, Dim> padding = fn.target_padding();
    for (std::size_t i = 0; i < Dim; ++i) {
        if (padding[i].offset != padding_ref[i].offset) {
            // don't try to unify different offsets yet (they probably correspond to different problems).
            ADD_FAILURE() << "Target padding offset mismatch for dimension " << i << ": "
                          << padding[i].offset << " != " << padding_ref[i].offset;
        }

        padding[i].grid_left = std::max(padding[i].grid_left, padding_ref[i].grid_left);
        padding[i].grid_right = std::max(padding[i].grid_right, padding_ref[i].grid_right);
    }

    // Create subproblem input.
    // Note that input is not sorted as this is functionality, not performance test.
    auto input = make_spread_subproblem_input<T>(num_points, seed, grid, padding);

    // Allocate output arrays.
    auto output = finufft::allocate_aligned_array<T>(2 * grid.num_elements(), 64);
    auto output_reference = finufft::allocate_aligned_array<T>(2 * grid.num_elements(), 64);

    reference_fn(input, grid, output_reference.get());
    fn(input, grid, output.get());

    return {std::move(output_reference), std::move(output), grid};
}

/** Calls the given implementation of the subproblem with data
 * at the boundary of the grid. This function is mostly intended
 * to test the implementation for memory correctness, however,
 * it might only be effective when run under address-sanitizer
 * or a similar tool.
 *
 */
template <std::size_t Dim, typename T, typename Fn>
void evaluate_subproblem_limits(int width, Fn &&factory) {
    auto kernel_spec = specification_from_width(width, 2.0);
    auto fn = factory(kernel_spec);

    // Arbitrary grid specification
    auto offset = 5;
    auto extent = 100;

    std::array<std::size_t, Dim> extent_multiple = fn.extent_multiple();
    finufft::spreading::grid_specification<Dim> grid;
    for (std::size_t i = 0; i < Dim; ++i) {
        grid.offsets[i] = offset;
        grid.extents[i] = finufft::spreading::round_to_next_multiple(extent, extent_multiple[i]);
    }

    auto num_points = finufft::spreading::round_to_next_multiple(2, fn.num_points_multiple());
    auto padding = fn.target_padding();

    finufft::spreading::SpreaderMemoryInput<Dim, T> input(num_points);

    std::array<double, Dim> min_x;
    std::array<double, Dim> max_x;

    for (std::size_t i = 0; i < Dim; ++i) {
        min_x[i] = padding[i].min_valid_value(grid.offsets[i], grid.extents[i]);
        max_x[i] = padding[i].max_valid_value(grid.offsets[i], grid.extents[i]);
    }

    for (std::size_t i = 0; i < Dim; ++i) {
        std::fill_n(input.coordinates[i], num_points / 2, min_x[i]);
        std::fill_n(input.coordinates[i] + num_points / 2, num_points - num_points / 2, max_x[i]);
    }
    std::fill_n(input.strengths, 2 * num_points, 1.0);

    auto output = finufft::allocate_aligned_array<T>(2 * grid.num_elements(), 64);
    fn(input, grid, output.get());
}

template <typename T, std::size_t Dim, typename Fn>
void evaluate_subproblem_implementation_with_points(int width, int num_points, Fn &&factory) {
    auto result = evaluate_subproblem_implementation<Dim, T>(factory, num_points, 0, width);

    // Note: check correct error level computation for the test
    // Currently more lax in higher dimension due to (potentially bad?)
    // AVX512 implementation in 2D.
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

template <typename T, std::size_t Dim, typename Fn>
void test_subproblem_implementation(int width, Fn &&factory) {
    // Evaluate memory correctness first
    evaluate_subproblem_limits<Dim, T>(width, factory);

    {
        SCOPED_TRACE("num_points = 1");
        evaluate_subproblem_implementation_with_points<T, Dim>(width, 1, factory);
    }

    {
        SCOPED_TRACE("num_points = 100");
        evaluate_subproblem_implementation_with_points<T, Dim>(width, 100, factory);
    }
}

} // namespace

TEST(SpreadSubproblem, SpreadSubproblem1df32) {
    test_subproblem_implementation<float, 1>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemLegacyFunctor<float, 1>{k};
        });
}

TEST(SpreadSubproblem, ReferenceDirect1Df32) {
    test_subproblem_implementation<float, 1>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference<float, 1>{k};
        });
}

TEST(SpreadSubproblem, ReferenceDirect1Df64) {
    test_subproblem_implementation<double, 1>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference<double, 1>{k};
        });
}

TEST(SpreadSubproblem, ReferenceDirect2Df32) {
    test_subproblem_implementation<float, 2>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference<float, 2>{k};
        });
}

TEST(SpreadSubproblem, ReferenceDirect2Df64) {
    test_subproblem_implementation<double, 2>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference<double, 2>{k};
        });
}

TEST(SpreadSubproblem, ReferenceDirect3Df32) {
    test_subproblem_implementation<float, 3>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference<float, 3>{k};
        });
}

TEST(SpreadSubproblem, ReferenceDirect3Df64) {
    test_subproblem_implementation<double, 3>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::SpreadSubproblemDirectReference<double, 3>{k};
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

TEST(SpreadSubproblem, Avx512_1D_f32) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    test_subproblem_implementation<float, 1>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_1d_fp32_functor(k);
        });
}

TEST(SpreadSubproblem, Avx512_1D_f32_Short) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    test_subproblem_implementation<float, 1>(
        4, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_1d_fp32_functor(k);
        });
}

TEST(SpreadSubproblem, Avx512_1D_f64) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    test_subproblem_implementation<double, 1>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_1d_fp64_functor(k);
        });
}

TEST(SpreadSubproblem, Avx512_2D_f32) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    test_subproblem_implementation<float, 2>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_2d_fp32_functor(k);
        });
}

TEST(SpreadSubproblem, Avx512_2D_f64) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    test_subproblem_implementation<double, 2>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_2d_fp64_functor(k);
        });
}

TEST(SpreadSubproblem, Avx512_3D_f32) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    test_subproblem_implementation<float, 3>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_3d_fp32_functor(k);
        });
}
