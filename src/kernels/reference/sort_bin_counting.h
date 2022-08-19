#pragma once

/** @file
 *
 * Declaration of reference implementation bits for bin-sorting of non-uniform points
 * based on a counting sort approach.
 *
 */

#include "../sorting.h"
#include "../spreading.h"

namespace finufft {
namespace spreading {
namespace reference {

/** Bin-sort the given collection of non-uniform points.
 *
 * This implementation directly moves the points across memory and is
 * single-threaded. This method is only performant for small point collections.
 *
 */
template <typename T, std::size_t Dim>
SortPointsPlannedFunctor<T, Dim> make_sort_counting_direct_singlethreaded(
    FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info);

template <typename T, std::size_t Dim>
SortPointsPlannedFunctor<T, Dim>
make_sort_counting_direct_omp(FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info);

/** Bin-sort the given collection of non-uniform points.
 *
 * This implementation is single threaded but makes use
 * of local buffer when moving memory. It is more efficient
 * for collections of points which are larger than the local
 * cache of the processor.
 *
 */
template <typename T, std::size_t Dim>
SortPointsPlannedFunctor<T, Dim> make_sort_counting_blocked_singlethreaded(
    FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info);

} // namespace reference
} // namespace spreading
} // namespace finufft
