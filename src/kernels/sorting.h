#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>

#include <function2/function2.h>
#include <tcb/span.hpp>

#include "../spreading.h"

/** Definitions for common interfaces to the sorting problem.
 *
 */

namespace finufft {
namespace spreading {

/** Basic information about a binned grid.
 *
 * This structure is used to hold commonly needed information
 * about a basic integer binned grid.
 *
 * An integer binned is defined as a rectangular grid, divided
 * into equally sized rectangular bins of integer length.
 *
 * For a point at (real) coordinates `(x, y, z)`, the bin containing
 * that point is defined as the bin containing the integer coordinate
 * `(ceil(x - o_x) + g_x, ceil(y - o_y) + g_y, ceil(z - o_z) + g_z)`,
 * where o_x etc. denote the offset to apply, and g_x = ceil(-o_x).
 *
 */
template <typename T, std::size_t Dim> struct IntBinInfo {
    std::array<std::size_t, Dim> size;             ///< Size of the underlying target grid
    std::array<std::size_t, Dim> bin_size;         ///< Size of each bin
    std::array<std::size_t, Dim> bin_index_stride; ///< Stride used to compute global bin index
    std::array<std::size_t, Dim> num_bins;         ///< Number of bins in each dimension
    std::array<T, Dim> offset;                     ///< Offset to use when computing bin index
    std::array<int64_t, Dim> global_offset;        ///< Offset to use when computing bin index

    IntBinInfo(
        tcb::span<const std::size_t, Dim> size, tcb::span<const std::size_t, Dim> bin_size,
        tcb::span<const T, Dim> offset) {
        std::copy(size.begin(), size.end(), this->size.begin());
        std::copy(bin_size.begin(), bin_size.end(), this->bin_size.begin());
        std::copy(offset.begin(), offset.end(), this->offset.begin());

        for (std::size_t d = 0; d < Dim; ++d) {
            num_bins[d] = (size[d] + bin_size[d] - 1) / bin_size[d];
            global_offset[d] = std::ceil(-offset[d]);
        }

        bin_index_stride[0] = 1;
        for (std::size_t d = 1; d < Dim; ++d) {
            bin_index_stride[d] = bin_index_stride[d - 1] * num_bins[d - 1];
        }
    }

    std::size_t num_bins_total() const {
        return std::accumulate(num_bins.begin(), num_bins.end(), 1, std::multiplies<std::size_t>());
    }
};

/** Structure representing a packed bin, coordinate and strength triple.
 *
 * When sorting, it is more efficient to move all related information concerning
 * a given point jointly, rather than computing a permutation and applying it,
 * due to the fact that advanced parallel sorters such as IPS4o attempt to
 * break up the sorting in contiguous chunks.
 *
 */
template <typename T, std::size_t Dim> struct PointBin {
    uint32_t bin;                   ///< Index of the bin containing the point
    std::array<T, Dim> coordinates; ///< Coordinates of the point
    std::array<T, 2> strength;      ///< Complex strength of the point
};

template <typename T, std::size_t Dim>
bool operator<(const PointBin<T, Dim> &lhs, const PointBin<T, Dim> &rhs) {
    return lhs.bin < rhs.bin;
}

/** Functor type for the compute-bin-pack operation
 *
 * This operation fold-rescales each point, computes the corresponding bin, and writes the packed
 * output.
 *
 * Some implementations may require arrays to be aligned.
 *
 * @param input The input points, with coordinates and strengths
 * @param range The range of the input point, will determine the type of rescaling applied
 * @param info Bin / grid configuration
 * @param[out] output An array of size input.num_points to which the packed point / bin info will be
 * written.
 *
 */
template <typename T, std::size_t Dim>
using ComputeAndPackBinsFunctor = fu2::unique_function<void(
    nu_point_collection<Dim, const T> const &, FoldRescaleRange, IntBinInfo<T, Dim> const &,
    PointBin<T, Dim> *) const>;

/** Functor type for the bin-unpack operation.
 *
 * This operation unpacks the packed bin / point info, writing the point information
 * to the given point collection, and the bin index of each corresponding point
 * to the given array.
 *
 * @param input An array of packed point data, with length `output.num_points`
 * @param output The point collection to which the points will be written
 * @param[out] bin_index An array of length `output.num_points` to which the bin index of each point
 * will be written.
 *
 */
template <typename T, std::size_t Dim>
using UnpackBinsFunctor = fu2::unique_function<void(
    PointBin<T, Dim> const *, nu_point_collection<Dim, T> const &, uint32_t *) const>;

} // namespace spreading
} // namespace finufft
