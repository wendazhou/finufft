#include <gtest/gtest.h>

#include "spread_subproblem_test_utils.h"

#include "../src/kernels/avx512/spread_avx512.h"
#include "../src/kernels/dispatch.h"

namespace fst = finufft::spreading::testing;

namespace {

template <typename T, std::size_t Dim>
finufft::spreading::SpreaderMemoryInput<Dim, T> make_spreader_input_unaligned(
    std::size_t num_points, int seed, finufft::spreading::subgrid_specification<Dim> grid,
    tcb::span<const finufft::spreading::KernelWriteSpec<T>, Dim> padding) {

    auto input = fst::make_spread_subproblem_input<T, Dim>(num_points + 1, seed, grid, padding);

    return input;
}

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
            return make_spreader_input_unaligned<T, Dim>(num_points, seed, grid, padding);
        });
}

template <typename T, std::size_t Dim, typename Fn>
void test_subproblem_implementation_unaligned(int width, Fn &&factory) {
    {
        SCOPED_TRACE("num_points = 1");
        evaluate_subproblem_implementation_with_points<T, Dim>(width, 1, factory);
    }

    {
        SCOPED_TRACE("num_points = 100");
        evaluate_subproblem_implementation_with_points<T, Dim>(width, 100, factory);
    }
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

} // namespace

TEST(SpreadSubproblemUnaligned, Avx512_1D_f32) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    test_subproblem_implementation_unaligned<float, 1>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_1d_fp32_functor(k);
        });
}

TEST(SpreadSubproblemUnaligned, Avx512_1D_f32_Short) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    test_subproblem_implementation_unaligned<float, 1>(
        4, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_1d_fp32_functor(k);
        });
}

TEST(SpreadSubproblemUnaligned, Avx512_1D_f64) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    test_subproblem_implementation_unaligned<double, 1>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_1d_fp64_functor(k);
        });
}

TEST(SpreadSubproblemUnaligned, Avx512_2D_f32) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    test_subproblem_implementation_unaligned<float, 2>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_2d_fp32_functor(k);
        });
}

TEST(SpreadSubproblemUnaligned, Avx512_2D_f64) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    test_subproblem_implementation_unaligned<double, 2>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_2d_fp64_functor(k);
        });
}

TEST(SpreadSubproblemUnaligned, Avx512_3D_f32) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    test_subproblem_implementation_unaligned<float, 3>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_3d_fp32_functor(k);
        });
}

TEST(SpreadSubproblemUnaligned, Avx512_3D_f64) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        GTEST_SKIP() << "Skipping AVX512 test";
    }

    test_subproblem_implementation_unaligned<double, 3>(
        5, [](finufft::spreading::kernel_specification const &k) {
            return finufft::spreading::get_subproblem_polynomial_avx512_3d_fp64_functor(k);
        });
}
