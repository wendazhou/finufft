#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../src/constants.h"
#include "../src/kernels/avx512/plan.h"
#include "../src/kernels/legacy/plan.h"

#include "spread_test_utils.h"

#include <finufft.h>

namespace {

template <typename T>
void exact_nuft_type1(
    finufft::spreading::nu_point_collection<1, const T> const &points, T *output,
    std::size_t modes) {
    for (std::size_t mode = 0; mode < modes; ++mode) {
        T freq = mode;
        if (freq > modes / 2) {
            freq -= modes;
        }

        for (std::size_t i = 0; i < points.num_points; ++i) {
            T phase = freq * points.coordinates[0][i];
            output[2 * mode] += points.strengths[i] * std::cos(phase);
            output[2 * mode + 1] += points.strengths[i] * std::sin(phase);
        }
    }
}

} // namespace

TEST(ExactNuftType1, TestSinglePoint1DF64) {
    auto points = finufft::spreading::testing::make_random_point_collection<1, double>(
        1, 0, {-3 * finufft::constants::pi_v<double>, 3 * finufft::constants::pi_v<double>});
    points.coordinates[0][0] = 0.0;
    points.strengths[0] = 1.0;
    points.strengths[1] = 0.0;
    points.num_points = 1;

    std::size_t modes = 16;

    auto output_exact_holder = finufft::allocate_aligned_array<double>(2 * modes, 64);
    auto output_fast_holder = finufft::allocate_aligned_array<double>(2 * modes, 64);

    tcb::span<double> output_exact = {output_exact_holder.get(), 2 * modes};
    tcb::span<double> output_fast = {output_fast_holder.get(), 2 * modes};

    std::fill(output_exact.begin(), output_exact.end(), 0.0);
    std::fill(output_fast.begin(), output_fast.end(), 0.0);

    finufft::Type1TransformConfiguration<1> config = {
        .max_num_points_ = points.num_points, .modes_ = {modes}, .max_threads_ = 1};
    auto plan = finufft::legacy::make_type1_plan<double, 1>(config);

    exact_nuft_type1<double>(points, output_exact.data(), modes);
    plan(points.num_points, points.coordinates, points.strengths, output_fast.data());

    for (std::size_t i = 0; i < 2 * modes; ++i) {
        EXPECT_NEAR(output_exact[i], output_fast[i], 1e-5) << "i = " << i;
    }
}

TEST(ExactNuftType1, TestSinglePoint1DF64Guru) {
    auto points = finufft::spreading::testing::make_random_point_collection<1, double>(
        1, 0, {-3 * finufft::constants::pi_v<double>, 3 * finufft::constants::pi_v<double>});
    points.coordinates[0][0] = 0.0;
    points.strengths[0] = 1.0;
    points.strengths[1] = 0.0;
    points.num_points = 1;

    std::size_t modes = 16;

    auto output_exact_holder = finufft::allocate_aligned_array<double>(2 * modes, 64);
    auto output_fast_holder = finufft::allocate_aligned_array<double>(2 * modes, 64);

    tcb::span<double> output_exact = {output_exact_holder.get(), 2 * modes};
    tcb::span<double> output_fast = {output_fast_holder.get(), 2 * modes};

    std::fill(output_exact.begin(), output_exact.end(), 0.0);
    std::fill(output_fast.begin(), output_fast.end(), 0.0);

    exact_nuft_type1<double>(points, output_exact.data(), modes);

    finufft_opts opts;
    finufft_default_opts(&opts);

    int ier = finufft1d1(
        points.num_points,
        points.coordinates[0],
        reinterpret_cast<std::complex<double>*>(points.strengths),
        +1,
        1e-6,
        modes,
        reinterpret_cast<std::complex<double>*>(output_fast.data()),
        &opts);


    for (std::size_t i = 0; i < 2 * modes; ++i) {
        EXPECT_NEAR(output_exact[i], output_fast[i], 1e-5) << "i = " << i;
    }
}
