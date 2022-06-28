#include <algorithm>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "../src/precomputed_poly_kernel_data.h"
#include "../src/kernels/reference/spreading_reference.h"

#include "spread_test_utils.h"

namespace {

template<typename T>
void evaluate_kernel_grid(T* output, T offset, T beta, std::size_t width, std::size_t out_width) {
    T x = offset / width;

    for(std::size_t i = 0; i < width; ++i) {
        T center = 2 * (i + 0.5) / width - 1;
        T z = std::clamp(center + x, static_cast<T>(-1), static_cast<T>(1));
        output[i] = std::exp(beta * std::sqrt(1 - z * z));
    }

    for(std::size_t i = width; i < out_width; ++i) {
        output[i] = 0;
    }
}

}


TEST(KernelPolyEvaluation, KernelPolySingle) {
    const std::size_t degree = 8;
    const std::size_t out_width = 16;

    std::array<float, out_width> output;
    std::array<float, out_width> expected;

    for (auto const& poly_kernel_data : finufft::detail::precomputed_poly_kernel_data_table) {
        if (poly_kernel_data.degree != degree) {
            continue;
        }

        finufft::spreading::PolynomialBatch<float, out_width, degree> kernel(poly_kernel_data.coefficients, poly_kernel_data.width);
        float beta = poly_kernel_data.beta_1000 / 1000.;

        for (std::size_t i = 0; i < 100; ++i) {
            float x = 2 * (i / 100. + 0.5) - 1;
            kernel(output.data(), x);
            evaluate_kernel_grid(expected.data(), x, beta, poly_kernel_data.width, out_width);

            auto error_level = 1e-2 * std::exp(beta);
            ASSERT_THAT(output, ::testing::Pointwise(::testing::FloatNear(error_level), expected)) << "beta = " << beta;
        }
    }
}

