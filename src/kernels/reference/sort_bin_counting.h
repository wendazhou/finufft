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
void nu_point_counting_sort_direct_singlethreaded(
    nu_point_collection<Dim, const T> const &input, nu_point_collection<Dim, T> const &output,
    IntBinInfo<T, Dim> const &info, FoldRescaleRange input_range);

} // namespace reference
} // namespace spreading
} // namespace finufft
