#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>

#include <function2/function2.h>
#include <tcb/span.hpp>

#include "../tracing.h"
#include "spreading.h"

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
            global_offset[d] = std::ceil(-offset[d]);
            num_bins[d] = (size[d] - global_offset[d] + bin_size[d] - 1) / bin_size[d];
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

template <typename T, std::size_t Dim>
std::array<std::size_t, Dim> compute_bin_size_from_grid_and_padding(
    tcb::span<const std::size_t, Dim> grid_size, tcb::span<const KernelWriteSpec<T>, Dim> padding) {
    std::array<std::size_t, Dim> bin_size;

    for (std::size_t i = 0; i < Dim; ++i) {
        if (grid_size[i] < padding[i].grid_left + padding[i].grid_right) {
            throw std::runtime_error("Grid size is too small for padding");
        }

        bin_size[i] = grid_size[i] - padding[i].grid_left - padding[i].grid_right;
    }

    return bin_size;
}

template <typename T, std::size_t Dim>
std::array<T, Dim> get_offsets_from_padding(tcb::span<const KernelWriteSpec<T>, Dim> padding) {
    std::array<T, Dim> offset;
    for (std::size_t i = 0; i < Dim; ++i) {
        offset[i] = padding[i].offset;
    }
    return offset;
}

/** Structure representing bin information for integer-sized bins
 * based on target-sized grids.
 *
 * For performance reasons, it is crucial to solve the spread subproblem
 * on a subgrid of size bounded by the size of the L1 cache. We derive
 * the size of the bins from the size of the grid, minus any padding
 * required by the subproblem functor.
 *
 */
template <typename T, std::size_t Dim> struct IntGridBinInfo : IntBinInfo<T, Dim> {

    /** Initialize the bin information from the given specification.
     *
     * @param extents The size of the uniform buffer
     * @param grid_sizes The base size of the subproblem grid
     * @param padding The padding required by the subproblem functor
     *
     */
    IntGridBinInfo(
        tcb::span<const std::size_t, Dim> extents, tcb::span<const std::size_t, Dim> grid_size,
        tcb::span<const finufft::spreading::KernelWriteSpec<T>, Dim> padding)
        : IntBinInfo<T, Dim>(
              extents, compute_bin_size_from_grid_and_padding(grid_size, padding),
              get_offsets_from_padding(padding)) {

        std::copy(grid_size.begin(), grid_size.end(), this->grid_size.begin());
        std::copy(padding.begin(), padding.end(), this->padding.begin());
    }

    std::array<std::size_t, Dim> grid_size;      ///< Desired size for subproblem grid
    std::array<KernelWriteSpec<T>, Dim> padding; ///< The padding required by the subproblem functor
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

/** Set of timers used to measure performance of the pack-sort-unpack
 * operation.
 *
 */
struct SortPackedTimers {
    Timer pack;   ///< Time spent packing the data
    Timer sort;   ///< Time spent sorting the data
    Timer unpack; ///< Time spent unpacking the data

    SortPackedTimers() = default;
    SortPackedTimers(finufft::Timer const &timer)
        : pack(timer.make_timer("pack")), sort(timer.make_timer("sort")),
          unpack(timer.make_timer("unpack")) {}
    SortPackedTimers(SortPackedTimers const &) = default;
    SortPackedTimers(SortPackedTimers &&) = default;
};

// Macro to define a new functor with the given signature
// This creates a custom type derivig from fu2::unique_function with the given signature,
// enabling type erasure while giving a semantically meaningful name to the type-erasure class.
// In particular, this makes it significantly easier to diagnose which templates have not been
// instantiated in case of linker errors.
#define F(NAME, ...)                                                                               \
    template <typename T, std::size_t Dim> struct NAME : fu2::unique_function<__VA_ARGS__> {       \
        using fu2::unique_function<__VA_ARGS__>::unique_function;                                  \
        using fu2::unique_function<__VA_ARGS__>::operator=;                                        \
    };

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
F(ComputeAndPackBinsFunctor, void(
                                 nu_point_collection<Dim, const T> const &, FoldRescaleRange,
                                 IntBinInfo<T, Dim> const &, PointBin<T, Dim> *) const)

/** Functor type for the bin-unpack operation.
 *
 * This operation unpacks the packed bin / point info, writing the point information
 * to the given point collection, and the bin index of each corresponding point
 * to the given array.
 *
 * @param input An array of packed point data, with length `output.num_points`
 * @param output The point collection to which the points will be written
 * @param[out] bin_count An array of length `num_bins` to which the number of points in each bin
 *  will be written. Note that bins with no points will not have a value written to them.
 *
 */
F(UnpackBinsFunctor,
  void(PointBin<T, Dim> const *, nu_point_collection<Dim, T> const &, std::size_t *) const)

/** Functor for point sorting operation.
 *
 * Sorts the given points according to their bin index, and writes out fold-rescaled coordinates.
 * Additionally computes number of points falling into each bin.
 *
 * @param input A collection of points to be sorted.
 * @param range The range of the input data - determines how folding will be performed
 * @param output A collection of points to which the sorted points will be written. May be the same
 *            as `input`.
 * @param num_points_per_bin An array of length `info.num_bins_total()` to which the number of
 * points falling into each bin will be written. If there are no points in a bin, the value will not
 * be written.
 * @param info Bin / grid configuration
 *
 */
F(SortPointsFunctor,
  void(
      nu_point_collection<Dim, const T> const &, FoldRescaleRange,
      nu_point_collection<Dim, T> const &, std::size_t *, IntBinInfo<T, Dim> const &) const);

#undef F

} // namespace spreading
} // namespace finufft
