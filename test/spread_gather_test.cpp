#include "../src/spreading.h"

#include <gtest/gtest.h>
#include <random>
#include <utility>
#include <vector>

#include "spread_test_utils.h"

namespace {

template <std::size_t Dim, typename T, typename SpreaderFn>
finufft::spreading::SpreaderMemoryInput<Dim, T> run_spreader(
    std::size_t num_points, finufft::spreading::nu_point_collection<Dim, T const *> const &points,
    std::int64_t const *permutation, SpreaderFn &&spreader) {
    finufft::spreading::SpreaderMemoryInput<1, double> output(num_points);

    std::array<int64_t, Dim> sizes;
    std::fill(sizes.begin(), sizes.end(), 1);

    spreader(output, points, sizes, permutation, finufft::spreading::FoldRescaleRange::Identity);
    return std::move(output);
}

} // namespace

TEST(gather_rescale, gather_rescale_1d) {
    std::size_t num_points = 100;
    std::size_t num_gather_points = 20;

    auto points = make_random_point_collection<1, double>(num_points, 0, {-1.0, 2.0});
    auto permutation = make_random_permutation(num_points, 1);

    auto result = run_spreader<1, double>(
        num_gather_points,
        points.cast([](auto &&x) { return static_cast<double const *>(x.get()); }),
        permutation.data(),
        &finufft::spreading::gather_and_fold<1, int64_t, double>);

    auto result_expected = run_spreader<1, double>(
        num_gather_points,
        points.cast([](auto &&x) { return static_cast<double const *>(x.get()); }),
        permutation.data(),
        &finufft::spreading::gather_and_fold<1, int64_t, double>);

    EXPECT_EQ(result.num_points, result_expected.num_points);

    for (std::size_t i = 0; i < 1; ++i) {
        std::vector<double> result_vector(result.num_points);
        std::vector<double> result_expected_vector(result_expected.num_points);

        std::copy(
            result.coordinates[i].get(),
            result.coordinates[i].get() + result.num_points,
            result_vector.begin());
        std::copy(
            result_expected.coordinates[i].get(),
            result_expected.coordinates[i].get() + result_expected.num_points,
            result_expected_vector.begin());

        EXPECT_EQ(result_vector, result_expected_vector);
    }

    for (std::size_t i = 0; i < result.num_points; ++i) {
        EXPECT_EQ(result.strengths[2 * i], result_expected.strengths[2 * i]);
        EXPECT_EQ(result.strengths[2 * i + 1], result_expected.strengths[2 * i + 1]);
    }
}
