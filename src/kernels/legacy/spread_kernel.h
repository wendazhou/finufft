#pragma once

/** @file
 *
 * Various implementations and utilities to deal with the main
 * spreading kernel used by the FINUFFT algorithm.
 *
 */

#include "../spreading.h"

namespace finufft {
namespace spreading {
namespace legacy {

/** Adapters for the legacy implementations of onedim_fseries_kernel.
 *
 */
void onedim_fseries_kernel(
    std::size_t num_frequencies, float *coeffs, kernel_specification const &kernel_spec);
void onedim_fseries_kernel(
    std::size_t num_frequencies, double *coeffs, kernel_specification const &kernel_spec);

} // namespace legacy
} // namespace spreading
} // namespace finufft
