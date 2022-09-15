#include <gtest/gtest.h>

#include "../src/constants.h"
#include "../src/kernels/legacy/plan.h"

#include "spread_test_utils.h"

#include <function2/function2.h>

namespace {

template <typename T, std::size_t Dim>
using PlanFactory =
    fu2::unique_function<finufft::Type1Plan<T, Dim>(finufft::Type1TransformConfiguration<Dim>)
                             const>;

template <typename T, std::size_t Dim>
void run_type1_transform(
    std::array<std::size_t, Dim> const &num_modes, PlanFactory<T, Dim> const &plan_factory) {
    std::size_t num_modes_total = std::accumulate(
        num_modes.begin(), num_modes.end(), std::size_t{1}, std::multiplies<std::size_t>{});
    std::size_t num_points = num_modes_total - 3;

    auto points = finufft::spreading::testing::make_random_point_collection<Dim, T>(
        num_points, 0, {-3 * finufft::constants::pi_v<T>, 3 * finufft::constants::pi_v<T>});
    auto result = finufft::allocate_aligned_array<T>(2 * num_modes_total, 64);

    finufft::Type1TransformConfiguration<Dim> config = {
        .max_num_points_ = points.num_points, .modes_ = num_modes, .max_threads_ = 1};
    auto plan = plan_factory(config);

    plan(points.num_points, points.coordinates, points.strengths, result.get());
}

} // namespace

TEST(Type1IntegrationTest, Legacy1DF32) {
    run_type1_transform<float, 1>(
        {100}, &finufft::legacy::make_type1_plan<float, 1>);
}

TEST(Type1IntegrationTest, Legacy2DF32) {
    run_type1_transform<float, 2>(
        {64, 78}, &finufft::legacy::make_type1_plan<float, 2>);
}

TEST(Type1IntegrationTest, Legacy3DF32) {
    run_type1_transform<float, 3>(
        {35, 45, 31}, &finufft::legacy::make_type1_plan<float, 3>);
}


TEST(Type1IntegrationTest, Legacy1DF64) {
    run_type1_transform<double, 1>(
        {100}, &finufft::legacy::make_type1_plan<double, 1>);
}

TEST(Type1IntegrationTest, Legacy2DF64) {
    run_type1_transform<double, 2>(
        {64, 78}, &finufft::legacy::make_type1_plan<double, 2>);
}

TEST(Type1IntegrationTest, Legacy3DF64) {
    run_type1_transform<double, 3>(
        {35, 45, 31}, &finufft::legacy::make_type1_plan<double, 3>);
}

