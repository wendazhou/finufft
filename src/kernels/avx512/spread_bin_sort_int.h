#pragma once

/** @file
 *
 * Optimized integer bin-sorting routines.
 *
 */

#include "../reference/spread_bin_sort_int.h"

#include <tcb/span.hpp>

namespace finufft {
namespace spreading {
namespace avx512 {

using finufft::spreading::reference::IntBinInfo;
using finufft::spreading::reference::PointBin;

/** Fold and rescale the input coordinates, compute corresponding bin indices and pack into output.
 *
 */
template <typename T, std::size_t Dim>
void compute_bins_and_pack(
    nu_point_collection<Dim, const T> const &input, FoldRescaleRange range,
    IntBinInfo<T, Dim> const &info, PointBin<T, Dim> *output);

template <typename T, std::size_t Dim>
SortPointsFunctor<T, Dim> get_sort_functor(SortPackedTimers const& timers = {});

} // namespace avx512
} // namespace spreading
} // namespace finufft
