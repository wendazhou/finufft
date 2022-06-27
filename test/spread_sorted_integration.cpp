// Integration test comparing spread sorted to current implementation.

#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include <finufft/defs.h>
#include <finufft_spread_opts.h>

#include "spread_test_utils.h"

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
    BIGINT *sort_indices, BIGINT N1, BIGINT N2, BIGINT N3, float *data_uniform, BIGINT M,
    double *kx, double *ky, double *kz, double *data_nonuniform, finufft_spread_opts opts,
    int did_sort);

int setup_spreader(
    finufft_spread_opts &opts, double eps, double upsampfac, int kerevalmeth, int debug,
    int showwarn, int dim);

} // namespace spreadinterp

} // namespace finufft

namespace {
class SpreadSortedIntegrationTest : public ::testing::TestWithParam<std::tuple<double, double>> {};
} // namespace

TEST_P(SpreadSortedIntegrationTest, SpreadSorted1D) {
    auto params = GetParam();
    auto upsampfac = std::get<0>(params);
    auto eps = std::get<1>(params);

    auto num_points = 100;
    auto num_points_uniform = 100;

    auto input = make_random_point_collection<1, float>(num_points, 0, {-3 * M_PI, 3 * M_PI});
    auto permutation = make_random_permutation(num_points, 1);

    auto output_expected = std::make_unique<float[]>(2 * num_points_uniform);
    auto output = std::make_unique<float[]>(2 * num_points_uniform);

    finufft_spread_opts opts;
    finufft::spreadinterp::setup_spreader(opts, eps, upsampfac, 0, 0, 0, 1);
    opts.spread_direction = 1;

    finufft::spreadinterp::spreadSorted(
        permutation.get(),
        num_points_uniform,
        1,
        1,
        output.get(),
        input.num_points,
        input.coordinates[0].get(),
        nullptr,
        nullptr,
        input.strengths.get(),
        opts,
        1);

    finufft::spreadinterp::spreadSortedOriginal(
        permutation.get(),
        num_points_uniform,
        1,
        1,
        output_expected.get(),
        input.num_points,
        input.coordinates[0].get(),
        nullptr,
        nullptr,
        input.strengths.get(),
        opts,
        1);

    for (std::size_t i = 0; i < num_points_uniform; ++i) {
        ASSERT_NEAR(output_expected[i], output[i], std::abs(output_expected[i]) * 5e-3) << "i = " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    SpreadSortedAll, SpreadSortedIntegrationTest,
    ::testing::Combine(::testing::Values(1.25, 2.0), ::testing::Values(1e-4, 1e-5, 1e-6)));
