/** @file
 *
 * This file contains unit tests for implementations of the spreading subproblem.
 *
 */

#include <cstring>

#include "../src/kernels/avx512/spread_avx512.h"
#include "../src/kernels/legacy/spread_subproblem_legacy.h"
#include "../src/kernels/reference/spread_subproblem_reference.h"
#include "../src/kernels/spreading.h"

#include "../src/kernels/dispatch.h"

#include <numeric>

#include <gtest/gtest.h>

#include "spread_subproblem_test_utils.h"
#include "spread_test_utils.h"

namespace fst = finufft::spreading::testing;

namespace finufft {
namespace spreadinterp {
int setup_spreader(
    finufft_spread_opts &opts, double eps, double upsampfac, int kerevalmeth, int debug,
    int showwarn, int dim);
}

} // namespace finufft

namespace {

template <std::size_t Dim, typename T, typename Fn>
fst::evaluation_result<T, Dim> evaluate_subproblem_implementation(
    Fn &&fn_factory, std::size_t num_points, uint32_t seed, int width) {
    finufft::spreading::subgrid_specification<Dim> grid;

    // Arbitrary grid specification in all dimensions
    auto offset = 3;
    auto size = 24;

    for (std::size_t i = 0; i < Dim; ++i) {
        grid.offsets[i] = offset;
        grid.extents[i] = size;
    }

    {
        std::size_t stride = 1;
        for (std::size_t i = 0; i < Dim; ++i) {
            grid.strides[i] = stride;
            stride *= grid.extents[i];

            // Generate a stride setup with a non-contiguous stride
            if (i == 0) {
                stride *= 3;
            }
        }
    }

    return fst::evaluate_subproblem_implementation<Dim, T>(
        std::forward<Fn>(fn_factory),
        num_points,
        grid,
        width,
        [seed](auto num_points, auto grid, auto padding) {
            return fst::make_spread_subproblem_input<T, Dim>(num_points, seed, grid, padding);
        });
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
    auto kernel_spec = fst::specification_from_width(width, 2.0);
    auto fn = factory(kernel_spec);

    // Arbitrary grid specification
    auto offset = 5;
    auto extent = 100;

    std::array<std::size_t, Dim> extent_multiple = fn.extent_multiple();
    finufft::spreading::grid_specification<Dim> grid;
    for (std::size_t i = 0; i < Dim; ++i) {
        grid.offsets[i] = offset;
        grid.extents[i] = finufft::round_to_next_multiple(extent, extent_multiple[i]);
    }

    auto num_points = finufft::round_to_next_multiple(2, fn.num_points_multiple());
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

    auto error_level = fst::compute_max_relative_threshold(
        std::pow(10, -width + 1),
        result.output_reference.get(),
        result.output_reference.get() + 2 * result.grid.num_elements());

    for (std::size_t i = 0; i < 2 * result.grid.num_elements(); ++i) {
        ASSERT_NEAR(result.output_reference[i], result.output[i], error_level)
            << "i = " << i << "; "
            << "width = " << width;
    }
}

/** Note: this test uses bit-exact comparisons, and might be brittle.
 *
 */
template <typename T, std::size_t Dim, typename Fn>
void evaluate_subproblem_implementation_bitexact(int width, int num_points, Fn &&factory) {
    auto result = evaluate_subproblem_implementation<Dim, T>(factory, num_points, 0, width);

    for (std::size_t i = 0; i < 2 * result.grid.num_elements(); ++i) {
        ASSERT_EQ(result.output_reference[i], result.output[i]) << "i = " << i << "; "
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

TEST(SpreadSubproblem, Avx512_1D_f32_bitexact) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    evaluate_subproblem_implementation_bitexact<float, 1>(
        5, 1, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_1d_fp32_functor(k);
        });

    evaluate_subproblem_implementation_bitexact<float, 1>(
        5, 16, [](finufft::spreading::kernel_specification const &k) {
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

TEST(SpreadSubproblem, Avx512_2D_f32_bitexact) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    evaluate_subproblem_implementation_bitexact<float, 2>(
        5, 1, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_2d_fp32_functor(k);
        });

    evaluate_subproblem_implementation_bitexact<float, 2>(
        5, 16, [](finufft::spreading::kernel_specification const &k) {
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

TEST(SpreadSubproblem, Avx512_3D_f64) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    test_subproblem_implementation<double, 3>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_3d_fp64_functor(k);
        });
}
