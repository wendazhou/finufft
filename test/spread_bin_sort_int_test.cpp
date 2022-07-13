#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>

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
        ASSERT_EQ(packed_expected[i].bin, packed_actual[i].bin) << "i = " << i;
        ASSERT_FLOAT_EQ(packed_expected[i].coordinates[0], packed_actual[i].coordinates[0])
            << "i = " << i;
        ASSERT_EQ(packed_expected[i].strength[0], packed_actual[i].strength[0]) << "i = " << i;
        ASSERT_EQ(packed_expected[i].strength[1], packed_actual[i].strength[1]) << "i = " << i;
    }
}

template <typename T, std::size_t Dim, typename Fn> void test_compute_bins_and_pack(Fn &&fn) {
    std::vector<std::size_t> num_points_list = {1, 2, 3};
    for (std::size_t i = 0; i < 32; ++i) {
        num_points_list.push_back(200 + i);
    }

    for (auto num_points : num_points_list) {
        SCOPED_TRACE("num_points = " + std::to_string(num_points));
        test_compute_bins_and_pack<T, Dim>(num_points, fn);
    }
}

template <typename T, std::size_t Dim> void test_unpack_bins(std::size_t num_points) {
    auto points = make_random_point_collection<Dim, T>(num_points, 0, {-3 * M_PI, 3 * M_PI});
    auto packed = finufft::allocate_aligned_array<PointBin<T, Dim>>(num_points, 64);

    std::array<std::size_t, Dim> size;
    size.fill(128);

    std::array<std::size_t, Dim> bin_size;
    bin_size.fill(16);

    std::array<T, Dim> offset;
    offset.fill(4);

    IntBinInfo<T, Dim> info(size, bin_size, offset);

    fs::reference::compute_bins_and_pack<T, Dim>(
        points, fs::FoldRescaleRange::Pi, info, packed.get());
    std::sort(packed.get(), packed.get() + num_points);

    auto bin_counts = finufft::allocate_aligned_array<std::size_t>(info.num_bins_total(), 64);
    std::memset(bin_counts.get(), 0, info.num_bins_total() * sizeof(std::size_t));

    auto bin_idx = finufft::allocate_aligned_array<uint32_t>(num_points, 64);

    finufft::spreading::SpreaderMemoryInput<Dim, T> unpacked_reference(num_points);
    finufft::spreading::SpreaderMemoryInput<Dim, T> unpacked_sorted(num_points);

    fs::reference::unpack_bins_to_points<T, Dim>(packed.get(), unpacked_reference, bin_idx.get());
    fs::reference::unpack_sorted_bins_to_points<T, Dim>(packed.get(), unpacked_sorted, bin_counts.get());

    for (std::size_t i = 0; i < num_points; ++i) {
        for (std::size_t j = 0; j < Dim; ++j) {
            ASSERT_EQ(unpacked_reference.coordinates[j][i], unpacked_sorted.coordinates[j][i])
                << "i = " << i;
        }
        ASSERT_EQ(unpacked_reference.strengths[2 * i], unpacked_sorted.strengths[2 * i])
            << "i = " << i;
        ASSERT_EQ(unpacked_reference.strengths[2 * i + 1], unpacked_sorted.strengths[2 * i + 1])
            << "i = " << i;
    }

    for (std::size_t b = 0; b < info.num_bins_total(); ++b) {
        auto r = std::equal_range(bin_idx.get(), bin_idx.get() + num_points, b);
        ASSERT_EQ(r.second - r.first, bin_counts[b]) << "b = " << b;
    }
}

template <typename T, std::size_t Dim> void test_unpack_dims() {
    for (auto i : {1, 2, 3, 200, 1024, 2048}) {
        SCOPED_TRACE("i = " + std::to_string(i));
        test_unpack_bins<T, Dim>(i);
    }
}

} // namespace

TEST(TestBinSortInt, Avx512_1D) {
    test_compute_bins_and_pack<float, 1>(&fs::avx512::compute_bins_and_pack<float, 1>);
}

TEST(TestBinSortInt, Avx512_2D) {
    test_compute_bins_and_pack<float, 2>(&fs::avx512::compute_bins_and_pack<float, 2>);
}

TEST(TestBinSortInt, Avx512_3D) {
    test_compute_bins_and_pack<float, 3>(&fs::avx512::compute_bins_and_pack<float, 3>);
}

TEST(TestBinUnpackInt, UnpackSorted2D) {
    test_unpack_dims<float, 2>();
}
