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
    uint32_t bin;
    std::array<T, Dim> coordinates;
    std::array<T, 2> strength;
};

template <typename T, std::size_t Dim>
bool operator<(const PointBin<T, Dim> &lhs, const PointBin<T, Dim> &rhs) {
    return lhs.bin < rhs.bin;
}

/** Fold-rescale each point, compute corresponding bin, and write packed output.
 *
 * @param input The input points, with coordinates and strengths
 * @param range The range of the input point, will determine the type of rescaling applied
 * @param info Bin / grid configuration
 * @param[out] output An array of size input.num_points to which the packed point / bin info will be
 * written.
 *
 */
template <typename T, std::size_t Dim>
void compute_bins_and_pack(
    nu_point_collection<Dim, const T> input, FoldRescaleRange range, IntBinInfo<T, Dim> const &info,
    PointBin<T, Dim> *output);

/** Functor for computing bin index.
 *
 */
template <typename T, std::size_t Dim, typename FoldRescale> struct ComputeBinIndexSingle {
    IntBinInfo<T, Dim> const &info;
    FoldRescale fold_rescale;
    std::array<T, Dim> extents;

    ComputeBinIndexSingle(IntBinInfo<T, Dim> const &info, FoldRescale fold_rescale)
        : info(info), fold_rescale(fold_rescale) {
        std::copy(info.size.begin(), info.size.end(), extents.begin());
    }

    std::size_t operator()(tcb::span<const T, Dim> coords) const {
        std::size_t bin_index = 0;

        for (std::size_t j = 0; j < Dim; ++j) {
            auto coord = fold_rescale(coords[j], extents[j]);
            auto coord_grid_left = std::ceil(coord - info.offset[j]);
            std::size_t coord_grid = std::size_t(coord_grid_left - info.global_offset[j]);
            std::size_t bin_index_j = coord_grid / info.bin_size[j];

            bin_index += bin_index_j * info.bin_index_stride[j];
        }

        return bin_index;
    }
};

} // namespace reference
} // namespace spreading
} // namespace finufft
