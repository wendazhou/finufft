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
 * @param output Output array to which the result will be written
 * @param output_size Size of the output array
 * @param output_stride Stride of the output array. Note that the legacy functor
 *    only supports column major contiguous order.
 * @param input Input array from which the data will be read
 * @param input_size Size of the input array
 * @param input_stride Stride of the input array. Note that the legacy functor
 *    only supports column major contiguous order.
 * @param kernel Array of kernel values in each dimension. Each array must be
 *    of size output_size[i] / 2 + 1.
 * @param mode_ordering Ordering of the modes in the input array.
 *
 */
template <typename T, std::size_t Dim>
InterpolationFunctor<T, Dim> make_legacy_interpolation_functor(
    T *output, tcb::span<const std::size_t, Dim> output_size,
    tcb::span<const std::size_t, Dim> output_stride, T const *input,
    tcb::span<const std::size_t, Dim> input_size, tcb::span<const std::size_t, Dim> input_stride,
    tcb::span<const T *const, Dim> kernel, ModeOrdering mode_ordering);
} // namespace legacy
} // namespace interpolation
} // namespace finufft
