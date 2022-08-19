#include <gtest/gtest.h>

#include "../src/kernels/avx512/sort_bin_counting.h"
#include "../src/kernels/reference/sort_bin_counting.h"

#include "spread_test_utils.h"

namespace {

template <typename T, std::size_t Dim>
void test_sort_factory(
    std::size_t num_points, finufft::spreading::SortPointsFactory<T, Dim> const &fn) {
    auto points = make_random_point_collection<Dim, T>(1654, 0, {-M_PI, M_PI});
    auto output = finufft::spreading::SpreaderMemoryInput<Dim, T>(points.num_points);

    std::array<std::size_t, Dim> size;
    std::array<std::size_t, Dim> bin_size;

    if (Dim == 1) {
        size[0] = 1024;
        bin_size[0] = 16;
    } else if (Dim == 2) {
        size[0] = 256;
        size[1] = 256;

        bin_size[0] = 32;
        bin_size[1] = 16;
    }
    static_assert(Dim == 1 || Dim == 2, "Dim must be 1 or 2");

    std::array<T, Dim> offset;
    offset.fill(4);

    finufft::spreading::IntBinInfo<T, Dim> info(size, bin_size, offset);
    auto histogram = finufft::allocate_aligned_array<std::size_t>(info.num_bins_total(), 64);

    auto sort_fn = fn(finufft::spreading::FoldRescaleRange::Pi, info);
    sort_fn(points, output, histogram.get());

    auto point_bin_alloc = finufft::allocate_aligned_array<std::size_t>(points.num_points, 64);
    auto point_bin = tcb::span<std::size_t>(point_bin_alloc.get(), points.num_points);

    for (std::size_t i = 0; i < points.num_points; ++i) {
        std::array<T, Dim> coordinates;
        for (std::size_t j = 0; j < Dim; ++j) {
            coordinates[j] = output.coordinates[j][i];
        }

        point_bin[i] = finufft::spreading::compute_bin_index(info, coordinates);
    }

    ASSERT_TRUE(std::is_sorted(point_bin.begin(), point_bin.end()));
}

template <typename T, std::size_t Dim>
void test_sort(std::size_t num_points, finufft::spreading::SortPointsFunctor<T, Dim> &&fn) {
    test_sort_factory(num_points, finufft::spreading::make_factory_from_functor(std::move(fn)));
}

} // namespace

TEST(SortBinCountingTestFactory, SingleThreadedDirect1DReference) {
    test_sort_factory<float, 1>(
        1654, finufft::spreading::reference::make_sort_counting_direct_singlethreaded<float, 1>);
}

TEST(SortBinCountingTestFactory, SingleThreadedDirect2DReference) {
    test_sort_factory<float, 2>(
        1654, finufft::spreading::reference::make_sort_counting_direct_singlethreaded<float, 2>);
}

TEST(SortBinCountingTestFactory, OmpDirect1DReference) {
    test_sort_factory<float, 1>(
        1654, finufft::spreading::reference::make_sort_counting_direct_omp<float, 1>);
}

TEST(SortBinCountingTestFactory, OmpDirect2DReference) {
    test_sort_factory<float, 2>(
        1654, finufft::spreading::reference::make_sort_counting_direct_omp<float, 2>);
}

TEST(SortBinCountingTestFactory, SingleThreadedBlocked1DReference) {
    test_sort_factory<float, 1>(
        1654, finufft::spreading::reference::make_sort_counting_blocked_singlethreaded<float, 1>);
}

TEST(SortBinCountingTestFactory, SingleThreadedBlocked2DReference) {
    test_sort_factory<float, 2>(
        1654, finufft::spreading::reference::make_sort_counting_blocked_singlethreaded<float, 2>);
}

TEST(SortBinCountingTestFactory, SingleThreadedDirect1DAvx512) {
    test_sort_factory<float, 1>(
        1654, finufft::spreading::avx512::make_sort_counting_direct_singlethreaded<float, 1>);
}

TEST(SortBinCountingTestFactory, SingleThreadedDirect2DAvx512) {
    test_sort_factory<float, 2>(
        1654, finufft::spreading::avx512::make_sort_counting_direct_singlethreaded<float, 2>);
}

TEST(SortBinCountingTestFactory, SingleThreadedBlocked1DAvx512) {
    test_sort_factory<float, 1>(
        1654, finufft::spreading::avx512::make_sort_counting_blocked_singlethreaded<float, 1>);
}

TEST(SortBinCountingTestFactory, SingleThreadedBlocked2DAvx512) {
    test_sort_factory<float, 2>(
        1654, finufft::spreading::avx512::make_sort_counting_blocked_singlethreaded<float, 2>);
}

TEST(SortBinCountingTestFactory, OmpBlocked1DAvx512) {
    test_sort_factory<float, 1>(
        1654, finufft::spreading::avx512::make_sort_counting_blocked_omp<float, 1>);
}

TEST(SortBinCountingTestFactory, OmpBlocked2DAvx512) {
    test_sort_factory<float, 2>(
        1654, finufft::spreading::avx512::make_sort_counting_blocked_omp<float, 2>);
}
