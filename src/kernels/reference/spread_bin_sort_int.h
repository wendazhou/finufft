#pragma once

/** @file
 *
 * Experimental implementation towards bin-sort
 * using integer-based bins with cache-aligned subgrids.
 *
 * These implementations may eventually replace the current
 * sort - gather - spread - accumulate pipeline.
 *
 */

#include "../../bit.h"
#include "../../spreading.h"

#include "gather_fold_reference.h"

#include <tcb/span.hpp>

namespace finufft {
namespace spreading {
namespace reference {

/** Structure representing bin information for integer-sized bins
 * based on target-sized grids.
 *
 * For performance reasons, it is crucial to solve the spread subproblem
 * on a subgrid of size bounded by the size of the L1 cache. We derive
 * the size of the bins from the size of the grid, minus any padding
 * required by the subproblem functor.
 *
 */
template <typename T, std::size_t Dim> struct IntBinInfo {

    /** Initialize the bin information from the given specification.
     *
     * @param num_points The total number of points to be spread
     * @param extents The size of the uniform buffer
     * @param grid_sizes The base size of the subproblem grid
     * @param padding The padding required by the subproblem functor
     *
     */
    IntBinInfo(
        std::size_t num_points, tcb::span<const std::size_t, Dim> extents,
        tcb::span<const std::size_t, Dim> grid_sizes,
        tcb::span<const finufft::spreading::KernelWriteSpec<T>, Dim> padding)
        : grid_sizes(grid_sizes), extents(extents), padding(padding),
          bin_key_shift(finufft::bit_width(num_points)) {

        for (std::size_t i = 0; i < Dim; ++i) {
            if (grid_sizes[i] < padding[i].grid_left + padding[i].grid_right) {
                throw std::runtime_error("Grid size is too small for padding");
            }

            bin_sizes[i] = grid_sizes[i] - padding[i].grid_left - padding[i].grid_right;
            global_offset[i] = int64_t(std::ceil(-padding[i].offset));
            num_bins[i] = (extents[i] + bin_sizes[i] - 1) / bin_sizes[i];
        }

        bin_stride[0] = 1;
        for (std::size_t i = 1; i < Dim; ++i) {
            bin_stride[i] = bin_stride[i - 1] * num_bins[i - 1];
        }

        std::copy(extents.begin(), extents.end(), extents_f.begin());
    }

    tcb::span<const std::size_t, Dim>
        grid_sizes; ///< The size of the subproblem grid in each dimension
    tcb::span<const std::size_t, Dim> extents; ///< The size of the overall grid in each dimension
    tcb::span<const finufft::spreading::KernelWriteSpec<T>, Dim>
        padding; ///< The padding required by the subproblem functor
    std::array<int64_t, Dim>
        global_offset; ///< Computed global offset introduced by the subfunctor padding
    std::array<std::size_t, Dim> bin_sizes; ///< The computed size of the bins in each dimension
    std::array<std::size_t, Dim>
        bin_stride;            ///< The computed stride of the bin index in each dimension
    std::size_t bin_key_shift; ///< The computed shift for the bin key.
    std::array<std::size_t, Dim> num_bins; ///< The number of bins in each dimension
    std::array<T, Dim> extents_f; ///< Floating point version of the `extents` array.

    std::size_t num_bins_total() const {
        std::size_t total = 1;
        for (std::size_t i = 0; i < Dim; ++i) {
            total *= num_bins[i];
        }
        return total;
    }
};

/** Computation of bin grid index based on integer grid indices.
 * 
 * The computation of the bin index is based on the exact computation
 * that is performed for subproblem spreading, and hence we ensure
 * that the point is fully within the bin despite potential rounding
 * issues.
 *
 */
template <typename T, std::size_t Dim, typename FoldRescale>
std::size_t compute_bin_index_single(
    std::array<T, Dim> const &coords, IntBinInfo<T, Dim> const &info, FoldRescale &&fold_rescale) {
    std::size_t bin_index = 0;

    for (std::size_t j = 0; j < Dim; ++j) {
        auto coord = fold_rescale(coords[j], info.extents_f[j]);
        auto coord_grid_left = std::ceil(coord - info.padding[j].offset);
        std::size_t coord_grid = std::size_t(coord_grid_left - info.global_offset[j]);

        std::size_t bin_index_j = coord_grid / info.bin_sizes[j];

        bin_index += bin_index_j * info.bin_stride[j];
    }

    return bin_index;
}

} // namespace reference
} // namespace spreading
} // namespace finufft
