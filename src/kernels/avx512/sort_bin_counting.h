#pragma once

#include "../reference/sort_bin_counting.h"

namespace finufft {
namespace spreading {
namespace avx512 {

/** Bin-sort the given collection of non-uniform points.
 *
 * This implementation directly moves the points across memory and is
 * single-threaded. This method is only performant for small point collections.
 * This method uses AVX512 instructions to accelerate bin index computation.
 *
 */
template <typename T, std::size_t Dim>
void nu_point_counting_sort_direct_singlethreaded(
    nu_point_collection<Dim, const T> const &input, FoldRescaleRange input_range,
    nu_point_collection<Dim, T> const &output, std::size_t *num_points_per_bin,
    IntBinInfo<T, Dim> const &info);

template <typename T, std::size_t Dim>
void nu_point_counting_sort_blocked_singlethreaded(
    nu_point_collection<Dim, const T> const &input, FoldRescaleRange input_range,
    nu_point_collection<Dim, T> const &output, std::size_t *num_points_per_bin,
    IntBinInfo<T, Dim> const &info);

} // namespace avx512
} // namespace spreading
} // namespace finufft
