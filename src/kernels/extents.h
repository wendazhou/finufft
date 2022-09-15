#pragma once

#include <array>
#include <cstddef>
#include <tcb/span.hpp>

/** @file Utilities for working with strided arrays
 *
 */

namespace finufft {

/** Computes strides corresponding to a contiguous column-major layout.
 *
 */
template <std::size_t Dim>
std::array<std::size_t, Dim> strides_from_sizes(tcb::span<const std::size_t, Dim> sizes) {
    std::array<std::size_t, Dim> strides;
    // Contiguous column-major strides
    std::size_t stride = 1;

    for (std::size_t i = 0; i < Dim; ++i) {
        strides[i] = stride;
        stride *= sizes[i];
    }
    return strides;
}

/** Checks whether a given array is contiguous in column-major layout.
 *
 */
template <std::size_t Dim>
bool is_fortran_contiguous(
    tcb::span<const std::size_t, Dim> sizes, tcb::span<const std::size_t, Dim> strides) {
    std::size_t stride = 1;
    for (std::size_t i = 0; i < Dim; ++i) {
        if (strides[i] != stride) {
            return false;
        }
        stride *= sizes[i];
    }
    return true;
}

} // namespace finufft
