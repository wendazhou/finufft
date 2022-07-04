// Integration test comparing spread sorted to current implementation.

#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include <finufft/defs.h>
#include <finufft_spread_opts.h>

#include "spread_test_utils.h"

#include "../src/kernels/avx512/spread_avx512.h"
#include "../src/kernels/legacy/spread_legacy.h"
#include "../src/kernels/reference/spread_reference.h"
#include "../src/spreading_default.h"

namespace finufft {
namespace spreadinterp {

int spreadSorted(
    BIGINT *sort_indices, BIGINT N1, BIGINT N2, BIGINT N3, float *data_uniform, BIGINT M, float *kx,
    float *ky, float *kz, float *data_nonuniform, finufft_spread_opts opts, int did_sort);
int spreadSorted(
    BIGINT *sort_indices, BIGINT N1, BIGINT N2, BIGINT N3, float *data_uniform, BIGINT M,
    double *kx, double *ky, double *kz, double *data_nonuniform, finufft_spread_opts opts,
    int did_sort);

int spreadSortedOriginal(
    BIGINT *sort_indices, BIGINT N1, BIGINT N2, BIGINT N3, float *data_uniform, BIGINT M, float *kx,
    float *ky, float *kz, float *data_nonuniform, finufft_spread_opts opts, int did_sort);
int spreadSortedOriginal(
    BIGINT *sort_indices, BIGINT N1, BIGINT N2, BIGINT N3, double *data_uniform, BIGINT M,
    double *kx, double *ky, double *kz, double *data_nonuniform, finufft_spread_opts opts,
    int did_sort);

int setup_spreader(
    finufft_spread_opts &opts, double eps, double upsampfac, int kerevalmeth, int debug,
    int showwarn, int dim);

} // namespace spreadinterp

} // namespace finufft

namespace {

enum class ConfigType {
    legacy = 0,
    reference = 1,
    avx512 = 2,
};

class SpreadSortedIntegrationTest
    : public ::testing::TestWithParam<std::tuple<double, ConfigType>> {};

template <typename T, std::size_t Dim>
finufft::spreading::SpreadFunctorConfiguration<T, Dim>
get_configuration(ConfigType type, finufft_spread_opts const &opts) {
    switch (type) {
    case ConfigType::legacy:
        return finufft::spreading::get_spread_configuration_legacy<T, Dim>(opts);
    case ConfigType::reference:
        return finufft::spreading::get_spread_configuration_reference<T, Dim>(opts);
    case ConfigType::avx512:
        return finufft::spreading::get_spread_configuration_avx512<T, Dim>(opts);
    default:
        throw std::runtime_error("Unknown configuration type");
    }
}

template <typename T>
void run_spread_sorted_original(
    int64_t const *sort_index, finufft::spreading::nu_point_collection<1, T const> input,
    finufft_spread_opts const &opts, std::array<int64_t, 1> sizes, T *output) {
    finufft::spreadinterp::spreadSortedOriginal(
        const_cast<int64_t *>(sort_index),
        sizes[0],
        1,
        1,
        output,
        input.num_points,
        const_cast<T *>(input.coordinates[0]),
        nullptr,
        nullptr,
        const_cast<T *>(input.strengths),
        opts,
        1);
}

template <typename T>
void run_spread_sorted_original(
    int64_t const *sort_index, finufft::spreading::nu_point_collection<2, T const> input,
    finufft_spread_opts const &opts, std::array<int64_t, 2> sizes, T *output) {
    finufft::spreadinterp::spreadSortedOriginal(
        const_cast<int64_t *>(sort_index),
        sizes[0],
        sizes[1],
        1,
        output,
        input.num_points,
        const_cast<T *>(input.coordinates[0]),
        const_cast<T *>(input.coordinates[1]),
        nullptr,
        const_cast<T *>(input.strengths),
        opts,
        1);
}

template <typename T>
void run_spread_sorted_original(
    int64_t const *sort_index, finufft::spreading::nu_point_collection<3, T const> input,
    finufft_spread_opts const &opts, std::array<int64_t, 3> sizes, T *output) {
    finufft::spreadinterp::spreadSortedOriginal(
        const_cast<int64_t *>(sort_index),
        sizes[0],
        sizes[1],
        sizes[2],
        output,
        input.num_points,
        const_cast<T *>(input.coordinates[0]),
        const_cast<T *>(input.coordinates[1]),
        const_cast<T *>(input.coordinates[2]),
        const_cast<T *>(input.strengths),
        opts,
        1);
}

template <typename T, std::size_t Dim>
std::pair<std::vector<T>, std::vector<T>> run_spread_interp(double eps, ConfigType type) {
    double upsampfac = 2.0;

    std::array<int64_t, Dim> sizes;
    // Cannot make too small or some kernels may require multiple wraps,
    // which is not supported by accumulation process
    sizes.fill(35);
    auto output_size = std::accumulate(sizes.begin(), sizes.end(), 1ll, std::multiplies<int64_t>());

    auto num_points = 100;

    auto input = make_random_point_collection<Dim, T>(num_points, 0, {-3 * M_PI, 3 * M_PI});
    auto permutation = make_random_permutation(num_points, 1);

    auto output_expected = std::vector<T>(2 * output_size);
    auto output = std::vector<T>(2 * output_size);

    finufft_spread_opts opts;
    finufft::spreadinterp::setup_spreader(opts, eps, upsampfac, 1, 0, 0, 1);
    opts.spread_direction = 1;

    auto config = get_configuration<T, Dim>(type, opts);

    finufft::spreading::spread_with_configuration(
        permutation.get(), input, sizes, output.data(), opts, config);

    run_spread_sorted_original<T>(permutation.get(), input, opts, sizes, output_expected.data());

    return {std::move(output_expected), std::move(output)};
}

} // namespace

TEST_P(SpreadSortedIntegrationTest, SpreadSorted1DF32) {
    auto params = GetParam();
    auto eps = std::get<0>(params);
    auto config_type = std::get<1>(params);

    auto result = run_spread_interp<float, 1>(eps, config_type);
    auto &output_expected = result.first;
    auto &output = result.second;

    auto tolerance =
        compute_max_relative_threshold(eps, output_expected.begin(), output_expected.end());

    for (std::size_t i = 0; i < output_expected.size(); ++i) {
        ASSERT_NEAR(output_expected[i], output[i], tolerance) << "i = " << i;
    }
}

TEST_P(SpreadSortedIntegrationTest, SpreadSorted1DF64) {
    auto params = GetParam();
    auto eps = std::get<0>(params);
    auto config_type = std::get<1>(params);

    auto result = run_spread_interp<double, 1>(eps, config_type);
    auto &output_expected = result.first;
    auto &output = result.second;

    auto tolerance =
        compute_max_relative_threshold(eps, output_expected.begin(), output_expected.end());

    for (std::size_t i = 0; i < output_expected.size(); ++i) {
        ASSERT_NEAR(output_expected[i], output[i], tolerance) << "i = " << i;
    }
}

TEST_P(SpreadSortedIntegrationTest, SpreadSorted2DF32) {
    auto params = GetParam();
    auto eps = std::get<0>(params);
    auto config_type = std::get<1>(params);

    auto result = run_spread_interp<float, 2>(eps, config_type);
    auto &output_expected = result.first;
    auto &output = result.second;

    auto tolerance =
        compute_max_relative_threshold(eps, output_expected.begin(), output_expected.end());

    for (std::size_t i = 0; i < output_expected.size(); ++i) {
        ASSERT_NEAR(output_expected[i], output[i], tolerance) << "i = " << i;
    }
}

TEST_P(SpreadSortedIntegrationTest, SpreadSorted2DF64) {
    auto params = GetParam();
    auto eps = std::get<0>(params);
    auto config_type = std::get<1>(params);

    auto result = run_spread_interp<double, 2>(eps, config_type);
    auto &output_expected = result.first;
    auto &output = result.second;

    auto tolerance =
        compute_max_relative_threshold(eps, output_expected.begin(), output_expected.end());

    for (std::size_t i = 0; i < output_expected.size(); ++i) {
        ASSERT_NEAR(output_expected[i], output[i], tolerance) << "i = " << i;
    }
}

TEST_P(SpreadSortedIntegrationTest, SpreadSorted3DF32) {
    auto params = GetParam();
    auto eps = std::get<0>(params);
    auto config_type = std::get<1>(params);

    auto result = run_spread_interp<float, 3>(eps, config_type);
    auto &output_expected = result.first;
    auto &output = result.second;

    // TODO: check tolerance again once polynomial weights are standardized.
    auto tolerance =
        compute_max_relative_threshold(eps * 10, output_expected.begin(), output_expected.end());

    for (std::size_t i = 0; i < output_expected.size(); ++i) {
        ASSERT_NEAR(output_expected[i], output[i], tolerance) << "i = " << i;
    }
}


TEST_P(SpreadSortedIntegrationTest, SpreadSorted3DF64) {
    auto params = GetParam();
    auto eps = std::get<0>(params);
    auto config_type = std::get<1>(params);

    auto result = run_spread_interp<double, 3>(eps, config_type);
    auto &output_expected = result.first;
    auto &output = result.second;

    // TODO: check tolerance again once polynomial weights are standardized.
    auto tolerance =
        compute_max_relative_threshold(eps * 10, output_expected.begin(), output_expected.end());

    for (std::size_t i = 0; i < output_expected.size(); ++i) {
        ASSERT_NEAR(output_expected[i], output[i], tolerance) << "i = " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    SpreadSortedAll, SpreadSortedIntegrationTest,
    ::testing::Combine(
        ::testing::Values(1e-3, 1e-4, 1e-5),
        ::testing::Values(ConfigType::legacy, ConfigType::reference, ConfigType::avx512)));
