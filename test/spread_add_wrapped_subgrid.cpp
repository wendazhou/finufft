#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>

#include "../src/kernels/legacy/synchronized_accumulate_legacy.h"
#include "../src/kernels/reference/synchronized_accumulate_reference.h"
#include "../src/memory.h"

#include "test_utils.h"

namespace {

template <typename T, std::size_t Dim>
std::vector<T> run_add_subgrid(
    finufft::spreading::SynchronizedAccumulateFactory<T, Dim> const &factory,
    std::size_t num_subgrids) {
    auto output_base_size = 20;

    std::array<std::size_t, Dim> output_size;
    output_size.fill(output_base_size);

    auto total_size = std::accumulate(
        output_size.begin(),
        output_size.end(),
        static_cast<std::size_t>(1),
        std::multiplies<std::size_t>());

    std::vector<T> base_grid(2 * total_size);
    std::fill(base_grid.begin(), base_grid.end(), T(0));

    std::minstd_rand rng(42);
    std::uniform_int_distribution<int> offset_dist(-output_base_size + 1, output_base_size - 1);
    std::uniform_int_distribution<int> size_dist(1, 2 * output_base_size - 1);

    auto accumulate = factory(base_grid.data(), output_size);

    for (std::size_t i = 0; i < num_subgrids; ++i) {
        finufft::spreading::grid_specification<Dim> subgrid;
        std::generate(
            subgrid.offsets.begin(), subgrid.offsets.end(), [&]() { return offset_dist(rng); });
        std::generate(
            subgrid.extents.begin(), subgrid.extents.end(), [&]() { return size_dist(rng); });
        for (std::size_t j = 0; j < Dim; ++j) {
            if (subgrid.offsets[j] + subgrid.extents[j] > 2 * output_size[j]) {
                subgrid.extents[j] = 2 * output_size[j] - subgrid.offsets[j];
            }
        }

        auto data = finufft::allocate_aligned_array<T>(2 * subgrid.num_elements(), 64);
        finufft::testing::fill_random_uniform(data.get(), 2 * subgrid.num_elements(), i, -1, 1);

        accumulate(data.get(), subgrid);
    }

    return base_grid;
}

template <typename T, std::size_t Dim>
void test_add_subgrid_implementation(
    finufft::spreading::SynchronizedAccumulateFactory<T, Dim> const &factory) {

    auto num_additions = 10;

    auto result_reference = run_add_subgrid<T, Dim>(
        finufft::spreading::get_legacy_singlethreaded_accumulator<T, Dim>(), num_additions);
    auto result = run_add_subgrid<T, Dim>(factory, num_additions);

    // linear error tolerance in sum
    auto error = num_additions * 10 * std::numeric_limits<T>::epsilon();

    EXPECT_THAT(
        result,
        ::testing::Pointwise(::testing::DoubleNear(error), result_reference));
}

template <typename T> class AddSubgridTest : public ::testing::Test {};

struct LegacyImplementation {
    template <typename T, std::size_t Dim>
    finufft::spreading::SynchronizedAccumulateFactory<T, Dim> make() const {
        return finufft::spreading::get_legacy_singlethreaded_accumulator<T, Dim>();
    }
};

struct ReferenceImplementation {
    template <typename T, std::size_t Dim>
    finufft::spreading::SynchronizedAccumulateFactory<T, Dim> make() const {
        return finufft::spreading::get_reference_locking_accumulator<T, Dim>();
    }
};

struct ReferenceBlockLockingImplementation {
    template <typename T, std::size_t Dim>
    finufft::spreading::SynchronizedAccumulateFactory<T, Dim> make() const {
        return finufft::spreading::get_reference_block_locking_accumulator<T, Dim>();
    }
};

using ImplementationTypes = ::testing::Types<LegacyImplementation, ReferenceImplementation, ReferenceBlockLockingImplementation>;

} // namespace

TYPED_TEST_SUITE_P(AddSubgridTest);

TYPED_TEST_P(AddSubgridTest, Test1DF32) {
    TypeParam implementation;
    test_add_subgrid_implementation(implementation.template make<float, 1>());
}

TYPED_TEST_P(AddSubgridTest, Test2DF32) {
    TypeParam implementation;
    test_add_subgrid_implementation(implementation.template make<float, 2>());
}

TYPED_TEST_P(AddSubgridTest, Test3DF32) {
    TypeParam implementation;
    test_add_subgrid_implementation(implementation.template make<float, 3>());
}

TYPED_TEST_P(AddSubgridTest, Test1DF64) {
    TypeParam implementation;
    test_add_subgrid_implementation(implementation.template make<double, 1>());
}

TYPED_TEST_P(AddSubgridTest, Test2DF64) {
    TypeParam implementation;
    test_add_subgrid_implementation(implementation.template make<double, 2>());
}

TYPED_TEST_P(AddSubgridTest, Test3DF64) {
    TypeParam implementation;
    test_add_subgrid_implementation(implementation.template make<double, 3>());
}

REGISTER_TYPED_TEST_SUITE_P(AddSubgridTest, Test1DF32, Test1DF64, Test2DF32, Test2DF64, Test3DF32, Test3DF64);
INSTANTIATE_TYPED_TEST_SUITE_P(All, AddSubgridTest, ImplementationTypes);
