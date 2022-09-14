#pragma once

#include "../interpolation.h"
#include <tcb/span.hpp>

namespace finufft {
namespace interpolation {
namespace legacy {

/** Create an interpolation functor for the given problem using the
 * current legacy implementation.
 *
 * Note that all sizes and strides are expressed for complex interleaved
 * elements in `output` and `input`.
 *
 * @param output_size Size of the output array
 * @param output_stride Stride of the output array. Note that the legacy functor
 *    only supports column major contiguous order.
 * @param input_size Size of the input array
 * @param input_stride Stride of the input array. Note that the legacy functor
 *    only supports column major contiguous order.
 * @param kernel_factory Callable which fills given array with kernel values.
 * @param mode_ordering Ordering of the modes in the input array.
 *
 */
template <typename T, std::size_t Dim>
InterpolationFunctor<T, Dim> make_legacy_interpolation_functor(
    tcb::span<const std::size_t, Dim> output_size, tcb::span<const std::size_t, Dim> output_stride,
    tcb::span<const std::size_t, Dim> input_size, tcb::span<const std::size_t, Dim> input_stride,
    InterpolationKernelFactory<T> const &kernel_factory, ModeOrdering mode_ordering);
} // namespace legacy
} // namespace interpolation
} // namespace finufft
