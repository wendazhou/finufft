#pragma once

/** @file
 *
 * Various implementations and utilities to deal with the main
 * spreading kernel used by the FINUFFT algorithm.
 *
 */

#include "../interpolation.h"
#include "../spreading.h"

namespace finufft {
namespace spreading {
namespace reference {

/** Computes Fourier series coefficient of the real symmetric kernel,
 * directly via q-node quadrature on Euler-Fourier formula.
 *
 * @param num_frequencies The number of frequencies to compute the Fourier series for, must be even
 * @param coeffs The output array of size num_frequencies / 2 + 1.
 * @param kernel_spec The parameters of the kernel
 *
 */
void onedim_fseries_kernel(
    std::size_t num_frequencies, float *coeffs, kernel_specification const &kernel_spec);
void onedim_fseries_kernel(
    std::size_t num_frequencies, double *coeffs, kernel_specification const &kernel_spec);


/** Create a factory method for filling the interpolation kernel.
 * 
 * This encapsulates calls to `onedim_fseries_kernel` using the given `kernel_spec`
 * and `num_frequencies`.
 * 
 */
template <typename T, std::size_t Dim>
interpolation::InterpolationKernelFactory<T> make_interpolation_kernel_factory(
    tcb::span<const std::size_t, Dim> num_frequencies, kernel_specification const &kernel_spec);

/** Default choice of spreading kernel for the given tolerance and upsampling factor.
 *
 */
kernel_specification get_default_kernel_specification(double tolerance, double upsampling_factor);

} // namespace reference
} // namespace spreading
} // namespace finufft
