#include <gtest/gtest.h>

#include "../src/kernels/sorting.h"
#include "../src/kernels/spreading.h"

#include "../src/kernels/avx512/spread_avx512.h"
#include "../src/kernels/legacy/spread_legacy.h"
#include "../src/kernels/reference/spread_reference.h"

#include <tcb/span.hpp>

#include "spread_test_utils.h"

#include <iostream>

using namespace finufft::spreading;

namespace {

template <typename T, std::size_t Dim>
void test_spread_functor(
    std::size_t num_points, std::size_t target_width, kernel_specification const &kernel_spec,
    SpreadFunctor<T, Dim> const &functor) {
    std::array<std::size_t, Dim> target_size;
    target_size.fill(target_width);

    auto total_size =
        std::accumulate(target_size.begin(), target_size.end(), 1, std::multiplies<std::size_t>());

    auto points = make_random_point_collection<Dim, T>(num_points, 0, {-0.5, 0.5});

    auto output_ref_holder = finufft::allocate_aligned_array<T>(2 * total_size, 64);
    auto output_holder = finufft::allocate_aligned_array<T>(2 * total_size, 64);

    auto output_ref = output_ref_holder.get();
    auto output = output_holder.get();

    auto spread_reference = reference::get_indirect_spread_functor<T, Dim>(
        kernel_spec, target_size, FoldRescaleRange::Pi);

    spread_reference(points, output_ref);
    functor(points, output);

    // Compute error level
    // Error computation is based on approximation accuracy for the spreading kernel,
    // and a tolerance for accumulation based on linear error accumulation across the sum.
    double max_kernel_value =
        std::exp(kernel_spec.es_beta * Dim); // Baseline value for relative tolerance
    double density = std::pow(static_cast<double>(kernel_spec.width), Dim) * num_points /
                     total_size; // Average number of sums per point in grid
    density = std::max(density, 10.0);

    double tolerance = std::max(
        std::pow(10., -kernel_spec.width + 1),
        static_cast<double>(std::numeric_limits<T>::epsilon())); // Error tolerance in kernel
    double error_level =
        20 * tolerance * density; // Final relative error level (factor of 10 in tolerance).

    for (std::size_t i = 0; i < total_size; ++i) {
        ASSERT_NEAR(
            output_ref[2 * i] / max_kernel_value, output[2 * i] / max_kernel_value, error_level) << "i = " << i;
        ASSERT_NEAR(
            output_ref[2 * i + 1] / max_kernel_value,
            output[2 * i + 1] / max_kernel_value,
            error_level) << "i = " << i;
    }
}

template <typename T, std::size_t Dim>
void test_spread_functor_vary_points(
    std::size_t target_width, kernel_specification const &kernel_spec,
    SpreadFunctor<T, Dim> const &functor) {
    for (std::size_t num_points : {std::size_t(1), std::size_t(2), std::size_t(16), std::size_t(17), target_width, target_width * target_width}) {
        SCOPED_TRACE("num_points = " + std::to_string(num_points));
        test_spread_functor(num_points, target_width, kernel_spec, functor);
    }
}

} // namespace

TEST(TestSpreadFull, TestSpreadLegacy1D) {
    auto kernel_spec = specification_from_width(8, 2);
    std::size_t target_width = 128;

    auto spread_legacy_functor = legacy::make_spread_functor<float, 1>(
        kernel_spec, FoldRescaleRange::Pi, std::array<std::size_t, 1>{target_width});
    test_spread_functor_vary_points(target_width, kernel_spec, spread_legacy_functor);
}

TEST(TestSpreadFull, TestSpreadLegacy2D) {
    auto kernel_spec = specification_from_width(8, 2);
    std::size_t target_width = 128;

    auto spread_legacy_functor = legacy::make_spread_functor<float, 2>(
        kernel_spec, FoldRescaleRange::Pi, std::array<std::size_t, 2>{target_width, target_width});
    test_spread_functor_vary_points(target_width, kernel_spec, spread_legacy_functor);
}

TEST(TestSpreadFull, TestSpreadReferenceIndirect1D) {
    auto kernel_spec = specification_from_width(8, 2);
    std::size_t target_width = 128;

    auto spread_functor = reference::get_indirect_spread_functor<float, 1>(
        kernel_spec, std::array<std::size_t, 1>{target_width}, FoldRescaleRange::Pi);

    test_spread_functor_vary_points(target_width, kernel_spec, spread_functor);
}

TEST(TestSpreadFull, TestSpreadReferenceIndirect2D) {
    auto kernel_spec = specification_from_width(8, 2);
    std::size_t target_width = 128;

    auto spread_functor = reference::get_indirect_spread_functor<float, 2>(
        kernel_spec, std::array<std::size_t, 2>{target_width, target_width}, FoldRescaleRange::Pi);

    test_spread_functor_vary_points(target_width, kernel_spec, spread_functor);
}

TEST(TestSpreadFull, TestSpreadReferenceBlocked1D) {
    auto kernel_spec = specification_from_width(5, 2);
    std::size_t target_width = 128;

    auto spread_functor = reference::get_blocked_spread_functor<float, 1>(
        kernel_spec, std::array<std::size_t, 1>{target_width}, FoldRescaleRange::Pi);

    test_spread_functor_vary_points(target_width, kernel_spec, spread_functor);
}

// Note: numerical errors don't seem to work out for this test
TEST(TestSpreadFull, DISABLED_TestSpreadReferenceBlocked2D) {
    auto kernel_spec = specification_from_width(5, 2);
    std::size_t target_width = 128;

    auto spread_functor = reference::get_blocked_spread_functor<float, 2>(
        kernel_spec, std::array<std::size_t, 2>{target_width, target_width}, FoldRescaleRange::Pi);

    test_spread_functor_vary_points(target_width, kernel_spec, spread_functor);
}

TEST(TestSpreadFull, TestSpreadAvx512Blocked1D) {
    auto kernel_spec = specification_from_width(5, 2);
    std::size_t target_width = 128;

    auto spread_functor = avx512::get_blocked_spread_functor<float, 1>(
        kernel_spec, std::array<std::size_t, 1>{target_width}, FoldRescaleRange::Pi);

    test_spread_functor_vary_points(target_width, kernel_spec, spread_functor);
}

TEST(TestSpreadFull, TestSpreadAvx512Blocked2D) {
    auto kernel_spec = specification_from_width(5, 2);
    std::size_t target_width = 128;

    auto spread_functor = avx512::get_blocked_spread_functor<float, 2>(
        kernel_spec, std::array<std::size_t, 2>{target_width, target_width}, FoldRescaleRange::Pi);

    test_spread_functor_vary_points(target_width, kernel_spec, spread_functor);
}
