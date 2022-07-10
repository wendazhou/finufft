#include <gtest/gtest.h>

#include "../src/kernels/avx512/spread_bin_sort_int.h"
#include "../src/kernels/reference/spread_bin_sort_int.h"

#include "spread_test_utils.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

using finufft::spreading::reference::IntBinInfo;
using finufft::spreading::reference::PointBin;

namespace fs = finufft::spreading;

namespace {

template <typename T, std::size_t Dim, typename Fn>
void test_compute_bins_and_pack(std::size_t num_points, Fn &&fn) {
    auto points = make_random_point_collection<Dim, T>(num_points, 0, {-3 * M_PI, 3 * M_PI});
    auto packed_expected = finufft::allocate_aligned_array<PointBin<T, Dim>>(num_points, 64);
    auto packed_actual = finufft::allocate_aligned_array<PointBin<T, Dim>>(num_points, 64);

    std::array<std::size_t, Dim> size;
    size.fill(1024);

    std::array<std::size_t, Dim> bin_size;
    bin_size.fill(16);

    std::array<T, Dim> offset;
    offset.fill(4);

    IntBinInfo<T, Dim> info(size, bin_size, offset);

    fs::reference::compute_bins_and_pack<T, Dim>(
        points, fs::FoldRescaleRange::Pi, info, packed_expected.get());

    fs::avx512::compute_bins_and_pack<T, Dim>(
        points, fs::FoldRescaleRange::Pi, info, packed_actual.get());

    for (std::size_t i = 0; i < points.num_points; ++i) {
        ASSERT_EQ(packed_expected[i].bin, packed_actual[i].bin);
        ASSERT_FLOAT_EQ(packed_expected[i].coordinates[0], packed_actual[i].coordinates[0]);
        ASSERT_EQ(packed_expected[i].strength[0], packed_actual[i].strength[0]);
        ASSERT_EQ(packed_expected[i].strength[1], packed_actual[i].strength[1]);
    }
}

} // namespace

TEST(TestBinSortInt, Avx512_1D) {
    test_compute_bins_and_pack<float, 1>(200, &fs::avx512::compute_bins_and_pack<float, 1>);
}

TEST(TestBinSortInt, Avx512_2D) {
    test_compute_bins_and_pack<float, 2>(200, &fs::avx512::compute_bins_and_pack<float, 2>);
}

TEST(TestBinSortInt, Avx512_3D) {
    test_compute_bins_and_pack<float, 3>(200, &fs::avx512::compute_bins_and_pack<float, 3>);
}
