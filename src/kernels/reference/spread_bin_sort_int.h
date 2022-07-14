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
#include "../spreading.h"

#include "../sorting.h"

#include "gather_fold_reference.h"

#include <tcb/span.hpp>

namespace finufft {
namespace spreading {
namespace reference {

using finufft::spreading::IntBinInfo;
using finufft::spreading::PointBin;

/** Fold-rescale each point, compute corresponding bin, and write packed output.
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
void compute_bins_and_pack(
    nu_point_collection<Dim, const T> input, FoldRescaleRange range, IntBinInfo<T, Dim> const &info,
    PointBin<T, Dim> *output);

/** Unpacks packed point data into the given point collection.
 *
 * Some implementations may require arrays to be aligned.
 *
 * This function is used to unpack points after sorting.
 * The bin index of each point is additionally recorded to a separate
 * array for subsequent processing if needed.
 *
 * @param input An array of packed point data, with length `output.num_points`
 * @param output The point collection to which the points will be written
 * @param[out] bin_index An array of length `output.num_points` to which the bin index of each point
 * will be written.
 *
 */
template <typename T, std::size_t Dim>
void unpack_bins_to_points(
    PointBin<T, Dim> const *input, nu_point_collection<Dim, T> const &output, uint32_t *bin_index);

/** Unpacks sorted packed ponit data into the given point collection.
 *
 * This function is used to unpack points after sorting.
 * Unlike `unpack_bins_to_points`, this function requires the `input`
 * to be sorted according to the bin value, or it may not give correct results.
 * It additionally counts the number of points in each bin.
 *
 * @param input An array of packed point data, with length `output.num_points`
 * @param output The point collection to which the points will be written
 * @param[out] bin_count An array of length the number of bins to which the number of points in each
 * bin will be written. Note that if a bin is empty, no entry will be written to the array for that
 * bin.
 *
 */
template <typename T, std::size_t Dim>
void unpack_sorted_bins_to_points(
    PointBin<T, Dim> const *input, nu_point_collection<Dim, T> const &output,
    std::size_t *bin_count);

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

/** Create a point sorting functor backed by a ips4o parallel packed sort.
 * 
 * @param pack The packing functor to use to pack the points
 * @param unpack The unpacking functor to use to unpack the points
 * @param timers If not null, the record timining information using the given timer
 * 
 */
template <typename T, std::size_t Dim>
SortPointsFunctor<T, Dim> make_ips4o_sort_functor(
    ComputeAndPackBinsFunctor<T, Dim>&& pack, UnpackBinsFunctor<T, Dim>&& unpack,
    SortPackedTimers const *timers = nullptr);

/** Get the reference sort functor.
 * 
 */
template <typename T, std::size_t Dim>
SortPointsFunctor<T, Dim> get_sort_functor(SortPackedTimers const *timers = nullptr);

} // namespace reference
} // namespace spreading
} // namespace finufft
