#include <gtest/gtest.h>

#include "../src/kernels/reference/sort_bin_counting.h"
#include "spread_test_utils.h"

TEST(SortBinCountingTest, SingleThreadedDirect1D) {
    auto points = make_random_point_collection<1, float>(1654, 0, {-M_PI, M_PI});
    auto output = finufft::spreading::SpreaderMemoryInput<1, float>(points.num_points);

    finufft::spreading::IntBinInfo<float, 1> info({1024}, {16}, {4});

    finufft::spreading::reference::nu_point_counting_sort_direct_singlethreaded<float, 1>(
        points, output, info, finufft::spreading::FoldRescaleRange::Pi);

    auto point_bin_alloc = finufft::allocate_aligned_array<std::size_t>(points.num_points, 64);
    auto point_bin = tcb::span<std::size_t>(point_bin_alloc.get(), points.num_points);

    for (std::size_t i = 0; i < points.num_points; ++i) {
        point_bin[i] = finufft::spreading::compute_bin_index(info, {output.coordinates[0][i]});
    }

    ASSERT_TRUE(std::is_sorted(point_bin.begin(), point_bin.end()));
}


TEST(SortBinCountingTest, SingleThreadedDirect2D) {
    auto points = make_random_point_collection<2, float>(1654, 0, {-M_PI, M_PI});
    auto output = finufft::spreading::SpreaderMemoryInput<2, float>(points.num_points);

    finufft::spreading::IntBinInfo<float, 2> info({256, 256}, {32, 16}, {4, 4});

    finufft::spreading::reference::nu_point_counting_sort_direct_singlethreaded<float, 2>(
        points, output, info, finufft::spreading::FoldRescaleRange::Pi);

    auto point_bin_alloc = finufft::allocate_aligned_array<std::size_t>(points.num_points, 64);
    auto point_bin = tcb::span<std::size_t>(point_bin_alloc.get(), points.num_points);

    for (std::size_t i = 0; i < points.num_points; ++i) {
        point_bin[i] = finufft::spreading::compute_bin_index(info, {output.coordinates[0][i], output.coordinates[1][i]});
    }

    ASSERT_TRUE(std::is_sorted(point_bin.begin(), point_bin.end()));
}

