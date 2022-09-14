/** @file
 *
 * Tests for `spread_kernel.h` functionality.
 * Direct evaluation of kernel related quantities.
 *
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../src/kernels/legacy/spread_kernel.h"
#include "../src/kernels/reference/spread_kernel.h"
#include "../src/memory.h"

TEST(OneDimFseriesKernel, TestF32) {
    std::size_t num_frequencies = 256;
    std::size_t output_size = num_frequencies / 2 + 1;
    finufft::spreading::kernel_specification kernel_spec{2.0, 7};

    auto output_legacy_holder = finufft::allocate_aligned_array<float>(output_size, 64);
    auto output_reference_holder = finufft::allocate_aligned_array<float>(output_size, 64);

    tcb::span<float> output_legacy = {output_legacy_holder.get(), output_size};
    tcb::span<float> output_reference = {output_reference_holder.get(), output_size};

    finufft::spreading::reference::onedim_fseries_kernel(
        num_frequencies, output_reference.data(), kernel_spec);
    finufft::spreading::legacy::onedim_fseries_kernel(
        num_frequencies, output_legacy.data(), kernel_spec);

    ASSERT_THAT(output_legacy, testing::Pointwise(testing::FloatEq(), output_reference));
}

TEST(OneDimFseriesKernel, TestF64) {
    std::size_t num_frequencies = 256;
    std::size_t output_size = num_frequencies / 2 + 1;
    finufft::spreading::kernel_specification kernel_spec{2.0, 7};

    auto output_legacy_holder = finufft::allocate_aligned_array<double>(output_size, 64);
    auto output_reference_holder = finufft::allocate_aligned_array<double>(output_size, 64);

    tcb::span<double> output_legacy = {output_legacy_holder.get(), output_size};
    tcb::span<double> output_reference = {output_reference_holder.get(), output_size};

    finufft::spreading::reference::onedim_fseries_kernel(
        num_frequencies, output_reference.data(), kernel_spec);
    finufft::spreading::legacy::onedim_fseries_kernel(
        num_frequencies, output_legacy.data(), kernel_spec);

    ASSERT_THAT(output_legacy, testing::Pointwise(testing::DoubleEq(), output_reference));
}
