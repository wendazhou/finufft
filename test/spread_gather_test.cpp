#include "../src/spreading.h"

#include "../src/kernels/avx2/gather_fold_avx2.h"
#include "../src/kernels/avx512/gather_fold_avx512.h"
#include "../src/kernels/dispatch.h"
#include "../src/kernels/hwy/gather_fold_hwy.h"
#include "../src/kernels/reference/gather_fold_reference.h"

#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest-matchers.h>
#include <gtest/gtest.h>

#include "spread_test_utils.h"

namespace {

template <std::size_t Dim, typename T, typename SpreaderFn>
finufft::spreading::SpreaderMemoryInput<Dim, T> run_spreader(
    std::size_t num_points, finufft::spreading::nu_point_collection<Dim, T const> const &points,
    std::int64_t const *permutation, finufft::spreading::FoldRescaleRange range,
    SpreaderFn &&spreader) {
    finufft::spreading::SpreaderMemoryInput<Dim, T> output(num_points);

    std::array<int64_t, Dim> sizes;
    std::fill(sizes.begin(), sizes.end(), 1);

    spreader(output, points, sizes, permutation, range);
    return std::move(output);
}

} // namespace

#define MAKE_INVOKE_FUNCTOR(Name, Fn, DispatchLevel)                                               \
    struct Name {                                                                                  \
        template <std::size_t Dim, typename T>                                                     \
        void operator()(                                                                           \
            finufft::spreading::SpreaderMemoryInput<Dim, T> const &memory,                         \
            finufft::spreading::nu_point_collection<Dim, T const> const &input,                    \
            std::array<int64_t, Dim> const &sizes, std::int64_t const *sort_indices,               \
            finufft::spreading::FoldRescaleRange rescale_range) {                                  \
            Fn(memory, input, sizes, sort_indices, rescale_range);                                 \
        }                                                                                          \
                                                                                                   \
        static const finufft::DispatchCapability dispatch_level = DispatchLevel;                   \
    };

namespace {
template <typename T> class GatherRescaleFixture : public ::testing::Test {
  public:
    typedef std::tuple_element_t<0, T> InputType;
    static constexpr std::size_t Dim = std::tuple_element_t<1, T>::value;
    typedef std::tuple_element_t<2, T> FnType;
};

MAKE_INVOKE_FUNCTOR(
    GatherFoldReferenceFn, finufft::spreading::gather_and_fold, finufft::DispatchCapability::Scalar)
MAKE_INVOKE_FUNCTOR(
    GatherFoldAVX2Fn, finufft::spreading::gather_and_fold_avx2, finufft::DispatchCapability::AVX2)
MAKE_INVOKE_FUNCTOR(
    GatherFoldAVX512Fn, finufft::spreading::gather_and_fold_avx512,
    finufft::DispatchCapability::AVX512)
MAKE_INVOKE_FUNCTOR(
    GatherFoldHwyFn, finufft::spreading::highway::gather_and_fold_hwy, finufft::DispatchCapability::Scalar)

// clang-format off
typedef ::testing::Types<
    std::tuple<std::integral_constant<std::size_t, 1>, float, GatherFoldReferenceFn>,
    std::tuple<std::integral_constant<std::size_t, 2>, float, GatherFoldReferenceFn>,
    std::tuple<std::integral_constant<std::size_t, 3>, float, GatherFoldReferenceFn>,
    std::tuple<std::integral_constant<std::size_t, 1>, double, GatherFoldReferenceFn>,
    std::tuple<std::integral_constant<std::size_t, 2>, double, GatherFoldReferenceFn>,
    std::tuple<std::integral_constant<std::size_t, 3>, double, GatherFoldReferenceFn>,
    std::tuple<std::integral_constant<std::size_t, 1>, float, GatherFoldAVX512Fn>,
    std::tuple<std::integral_constant<std::size_t, 2>, float, GatherFoldAVX512Fn>,
    std::tuple<std::integral_constant<std::size_t, 3>, float, GatherFoldAVX512Fn>,
    std::tuple<std::integral_constant<std::size_t, 1>, double, GatherFoldAVX512Fn>,
    std::tuple<std::integral_constant<std::size_t, 2>, double, GatherFoldAVX512Fn>,
    std::tuple<std::integral_constant<std::size_t, 3>, double, GatherFoldAVX512Fn>,
    std::tuple<std::integral_constant<std::size_t, 1>, float, GatherFoldAVX2Fn>,
    std::tuple<std::integral_constant<std::size_t, 2>, float, GatherFoldAVX2Fn>,
    std::tuple<std::integral_constant<std::size_t, 3>, float, GatherFoldAVX2Fn>,
    std::tuple<std::integral_constant<std::size_t, 1>, float, GatherFoldHwyFn>,
    std::tuple<std::integral_constant<std::size_t, 2>, float, GatherFoldHwyFn>,
    std::tuple<std::integral_constant<std::size_t, 3>, float, GatherFoldHwyFn>
>
    GatherRescaleTestsParameters;
// clang-format on

} // namespace

TYPED_TEST_SUITE_P(GatherRescaleFixture);

TYPED_TEST_P(GatherRescaleFixture, GatherRescaleIdentity) {
    constexpr std::size_t Dim = std::tuple_element_t<0, TypeParam>::value;
    typedef std::tuple_element_t<1, TypeParam> FloatingType;
    typedef std::tuple_element_t<2, TypeParam> FnType;

    if (FnType::dispatch_level > finufft::get_current_capability()) {
        GTEST_SKIP() << "Skipping test because it requires a higher instruction set than currently "
                        "supported.";
    }

    std::size_t num_points = 100;
    std::size_t num_gather_points = 20;

    auto points = make_random_point_collection<Dim, FloatingType>(num_points, 0, {-1.0, 2.0});
    auto permutation = make_random_permutation(num_points, 1);

    auto range = finufft::spreading::FoldRescaleRange::Identity;

    auto result = run_spreader<Dim, FloatingType>(
        num_gather_points, points, permutation.get(), range, FnType{});
    auto result_expected = run_spreader<Dim, FloatingType>(
        num_gather_points,
        points,
        permutation.get(),
        range,
        &finufft::spreading::gather_and_fold<Dim, int64_t, FloatingType>);

    EXPECT_EQ(result.num_points, result_expected.num_points);

    for (std::size_t i = 0; i < 1; ++i) {
        std::vector<FloatingType> result_vector(result.num_points);
        std::vector<FloatingType> result_expected_vector(result_expected.num_points);

        std::copy(
            result.coordinates[i],
            result.coordinates[i] + result.num_points,
            result_vector.begin());
        std::copy(
            result_expected.coordinates[i],
            result_expected.coordinates[i] + result_expected.num_points,
            result_expected_vector.begin());

        ASSERT_EQ(result_vector, result_expected_vector);
    }

    for (std::size_t i = 0; i < result.num_points; ++i) {
        ASSERT_EQ(result.strengths[2 * i], result_expected.strengths[2 * i]) << "i = " << i;
        ASSERT_EQ(result.strengths[2 * i + 1], result_expected.strengths[2 * i + 1]) << "i = " << i;
    }
}

TYPED_TEST_P(GatherRescaleFixture, GatherRescalePi) {
    constexpr std::size_t Dim = std::tuple_element_t<0, TypeParam>::value;
    typedef std::tuple_element_t<1, TypeParam> FloatingType;
    typedef std::tuple_element_t<2, TypeParam> FnType;

    std::size_t num_points = 100;
    std::size_t num_gather_points = 20;

    auto points =
        make_random_point_collection<Dim, FloatingType>(num_points, 0, {-3 * M_PI, 3 * M_PI});
    auto permutation = make_random_permutation(num_points, 1);

    auto range = finufft::spreading::FoldRescaleRange::Pi;

    auto result = run_spreader<Dim, FloatingType>(
        num_gather_points, points, permutation.get(), range, FnType{});
    auto result_expected = run_spreader<Dim, FloatingType>(
        num_gather_points,
        points,
        permutation.get(),
        range,
        &finufft::spreading::gather_and_fold<Dim, int64_t, FloatingType>);

    EXPECT_EQ(result.num_points, result_expected.num_points);

    for (std::size_t i = 0; i < 1; ++i) {
        std::vector<FloatingType> result_vector(result.num_points);
        std::vector<FloatingType> result_expected_vector(result_expected.num_points);

        std::copy(
            result.coordinates[i],
            result.coordinates[i] + result.num_points,
            result_vector.begin());
        std::copy(
            result_expected.coordinates[i],
            result_expected.coordinates[i] + result_expected.num_points,
            result_expected_vector.begin());

        EXPECT_THAT(
            result_vector, ::testing::Pointwise(::testing::FloatEq(), result_expected_vector));
    }

    for (std::size_t i = 0; i < result.num_points; ++i) {
        ASSERT_EQ(result.strengths[2 * i], result_expected.strengths[2 * i]) << "i = " << i;
        ASSERT_EQ(result.strengths[2 * i + 1], result_expected.strengths[2 * i + 1]) << "i = " << i;
    }
}

REGISTER_TYPED_TEST_SUITE_P(GatherRescaleFixture, GatherRescaleIdentity, GatherRescalePi);

INSTANTIATE_TYPED_TEST_SUITE_P(GatherRescale, GatherRescaleFixture, GatherRescaleTestsParameters);

#undef MAKE_INVOKE_FUNCTION
