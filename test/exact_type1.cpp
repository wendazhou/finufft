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
        if (freq >= modes / 2) {
            freq -= modes;
        }

        for (std::size_t i = 0; i < points.num_points; ++i) {
            T phase = freq * points.coordinates[0][i];
            output[2 * mode] += points.strengths[i] * std::cos(phase);
            output[2 * mode + 1] += points.strengths[i] * std::sin(phase);
        }
    }
}

template <typename T>
void run_type1_transform_plan_legacy(
    finufft::spreading::nu_point_collection<1, const T> const &points, T *output,
    std::size_t modes) {
    finufft::Type1TransformConfiguration<1> config = {
        .max_num_points_ = points.num_points, .modes_ = {modes}, .max_threads_ = 1};
    auto plan = finufft::legacy::make_type1_plan<T, 1>(config);
    plan(points.num_points, points.coordinates, points.strengths, output);
}

template <typename T>
void run_type1_transform_plan_avx512(
    finufft::spreading::nu_point_collection<1, const T> const& points, T* output,
    std::size_t modes) {

    finufft::Type1TransformConfiguration<1> config = {
        .max_num_points_ = points.num_points, .modes_ = {modes}, .max_threads_ = 1};

    auto plan = finufft::avx512::make_type1_plan<T, 1>(config);
    plan(points.num_points, points.coordinates, points.strengths, output);
}

void run_type1_transform_simple(
    finufft::spreading::nu_point_collection<1, const double> const &points, double *output,
    std::size_t modes) {
    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.modeord = 1;

    int ier = finufft1d1(
        points.num_points,
        const_cast<double *>(points.coordinates[0]),
        reinterpret_cast<std::complex<double> *>(const_cast<double *>(points.strengths)),
        +1,
        1e-6,
        modes,
        reinterpret_cast<std::complex<double> *>(output),
        &opts);
}

template <typename T, typename Fn>
void test_type1_transform_single_point(
    std::size_t num_modes, T coordinate, Fn &&do_transform, double tolerance) {
    auto points = finufft::spreading::testing::make_random_point_collection<1, T>(
        1, 0, {-3 * finufft::constants::pi_v<T>, 3 * finufft::constants::pi_v<T>});
    points.coordinates[0][0] = coordinate;
    points.strengths[0] = 1.0;
    points.strengths[1] = 0.0;
    points.num_points = 1;

    std::size_t modes = 16;

    auto output_exact_holder = finufft::allocate_aligned_array<T>(2 * modes, 64);
    auto output_fast_holder = finufft::allocate_aligned_array<T>(2 * modes, 64);

    tcb::span<T> output_exact = {output_exact_holder.get(), 2 * modes};
    tcb::span<T> output_fast = {output_fast_holder.get(), 2 * modes};

    std::fill(output_exact.begin(), output_exact.end(), 0.0);
    std::fill(output_fast.begin(), output_fast.end(), 0.0);

    exact_nuft_type1<T>(points, output_exact.data(), modes);
    do_transform(points, output_fast.data(), modes);

    for (std::size_t i = 0; i < 2 * modes; ++i) {
        EXPECT_NEAR(output_exact[i], output_fast[i], 1e-5) << "i = " << i;
    }
}

class ExactNuftType1Test : public ::testing::TestWithParam<std::tuple<std::size_t, double>> {
};

} // namespace

TEST_P(ExactNuftType1Test, SinglePoint1DF64) {
    auto param = GetParam();
    auto num_modes = std::get<0>(param);
    auto coordinate = std::get<1>(param);

    test_type1_transform_single_point<double>(
        num_modes, coordinate, run_type1_transform_plan_legacy<double>, 1e-5);
}

TEST_P(ExactNuftType1Test, SinglePoint1DF32) {
    auto param = GetParam();
    auto num_modes = std::get<0>(param);
    auto coordinate = std::get<1>(param);

    test_type1_transform_single_point<float>(
        num_modes, coordinate, run_type1_transform_plan_legacy<float>, 1e-5);
}

TEST_P(ExactNuftType1Test, SinglePoint1DF32Avx512) {
    auto param = GetParam();
    auto num_modes = std::get<0>(param);
    auto coordinate = std::get<1>(param);

    test_type1_transform_single_point<float>(
        num_modes, coordinate, run_type1_transform_plan_avx512<float>, 1e-5);
}

TEST_P(ExactNuftType1Test, SinglePoint1DF64Guru) {
    auto param = GetParam();
    auto num_modes = std::get<0>(param);
    auto coordinate = std::get<1>(param);

    test_type1_transform_single_point<double>(
        num_modes, coordinate, run_type1_transform_simple, 1e-5);
}

INSTANTIATE_TEST_SUITE_P(
    ExactNuftType1Test,
    ExactNuftType1Test,
    ::testing::Combine(
        ::testing::Values(16),
        ::testing::Values(0, 1)));
