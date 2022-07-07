#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "../src/kernels/hwy/spread_bin_sort_hwy.h"
#include "../src/kernels/legacy/spread_bin_sort_legacy.h"
#include "../src/kernels/reference/gather_fold_reference.h"
#include "../src/kernels/reference/spread_bin_sort_reference.h"
#include "../src/memory.h"
#include "../src/spreading.h"

#include "spread_test_utils.h"

namespace {
template <typename T, std::size_t Dim>
std::vector<int64_t> compute_bin(
    finufft::spreading::nu_point_collection<Dim, T const> const &points,
    int64_t const *sort_indices, std::array<int64_t, Dim> const &sizes,
    std::array<T, Dim> const &bin_sizes) {

    finufft::spreading::SpreaderMemoryInput<Dim, T> points_folded(points.num_points);
    finufft::spreading::nu_point_collection<Dim, const T> points_folded_view = points_folded;

    finufft::spreading::gather_and_fold(
        points_folded, points, sizes, sort_indices, finufft::spreading::FoldRescaleRange::Pi);

    std::vector<int64_t> bin_index(points.num_points);

    std::array<T, Dim> extents;
    std::copy(sizes.begin(), sizes.end(), extents.begin());

    finufft::spreading::reference::compute_bin_index(
        bin_index.data(), points.num_points, points_folded_view.coordinates, extents, bin_sizes);

    return bin_index;
}

template <typename T, std::size_t Dim>
void test_binsort_implementation(
    std::size_t num_points, finufft::spreading::BinSortFunctor<T, Dim> &functor) {
    auto points = make_random_point_collection<Dim, T>(num_points, 0, {-3 * M_PI, 3 * M_PI});
    finufft::spreading::nu_point_collection<Dim, const T> points_view = points;

    auto output = finufft::allocate_aligned_array<int64_t>(
        finufft::spreading::round_to_next_multiple(points.num_points, 64 / sizeof(int64_t)), 64);
    auto output_ptr = output.get();

    std::array<int64_t, Dim> sizes;
    sizes.fill(256);

    std::array<T, Dim> extents;
    std::copy(sizes.begin(), sizes.end(), extents.begin());

    std::array<T, Dim> bin_sizes;
    bin_sizes.fill(16);

    functor(
        output.get(),
        points.num_points,
        points_view.coordinates,
        extents,
        bin_sizes,
        finufft::spreading::FoldRescaleRange::Pi);

    // Check that we did indeed get a permutation as output
    std::vector<int64_t> indices(output.get(), output.get() + points.num_points);
    std::sort(indices.begin(), indices.end());
    std::vector<int64_t> iota(points.num_points);
    std::iota(iota.begin(), iota.end(), static_cast<int64_t>(0));
    ASSERT_EQ(indices, iota);

    // Check that the bin index is in sorted order
    auto bin_index = compute_bin(points_view, output.get(), sizes, bin_sizes);
    ASSERT_TRUE(std::is_sorted(bin_index.begin(), bin_index.end()));
}

template <typename T, std::size_t Dim>
void test_binsort_implementation(finufft::spreading::BinSortFunctor<T, Dim> &functor) {
    {
        SCOPED_TRACE("num_points = 10");
        test_binsort_implementation(10, functor);
    }
    {
        SCOPED_TRACE("num_points = 100");
        test_binsort_implementation(100, functor);
    }
}

template <typename T> class SpreadBinSortTest : public ::testing::Test {};

struct LegacyImplementation {
    template <typename T, std::size_t Dim> finufft::spreading::BinSortFunctor<T, Dim> make() const {
        return finufft::spreading::get_bin_sort_functor_legacy<T, Dim>();
    }
};

struct ReferenceImplementation {
    template <typename T, std::size_t Dim> finufft::spreading::BinSortFunctor<T, Dim> make() const {
        return finufft::spreading::reference::get_bin_sort_functor<T, Dim>();
    }
};

struct HighwayImplementation {
    template <typename T, std::size_t Dim> finufft::spreading::BinSortFunctor<T, Dim> make() const {
        return finufft::spreading::highway::get_bin_sort_functor<T, Dim>();
    }
};

using ImplementationTypes =
    ::testing::Types<LegacyImplementation, ReferenceImplementation, HighwayImplementation>;

} // namespace

TYPED_TEST_SUITE_P(SpreadBinSortTest);

TYPED_TEST_P(SpreadBinSortTest, Test1DF32) {
    auto functor = TypeParam{}.template make<float, 1>();
    test_binsort_implementation(functor);
}

TYPED_TEST_P(SpreadBinSortTest, Test2DF32) {
    auto functor = TypeParam{}.template make<float, 2>();
    test_binsort_implementation(functor);
}

TYPED_TEST_P(SpreadBinSortTest, Test3DF32) {
    auto functor = TypeParam{}.template make<float, 3>();
    test_binsort_implementation(functor);
}

TYPED_TEST_P(SpreadBinSortTest, Test1DF64) {
    auto functor = TypeParam{}.template make<double, 1>();
    test_binsort_implementation(functor);
}

TYPED_TEST_P(SpreadBinSortTest, Test2DF64) {
    auto functor = TypeParam{}.template make<double, 2>();
    test_binsort_implementation(functor);
}

TYPED_TEST_P(SpreadBinSortTest, Test3DF64) {
    auto functor = TypeParam{}.template make<double, 3>();
    test_binsort_implementation(functor);
}

REGISTER_TYPED_TEST_SUITE_P(
    SpreadBinSortTest, Test1DF32, Test1DF64, Test2DF32, Test2DF64, Test3DF32, Test3DF64);
INSTANTIATE_TYPED_TEST_SUITE_P(All, SpreadBinSortTest, ImplementationTypes);
