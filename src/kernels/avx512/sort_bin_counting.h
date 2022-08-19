#pragma once

#include "../reference/sort_bin_counting.h"

namespace finufft {
namespace spreading {
namespace avx512 {

template <typename T, std::size_t Dim>
SortPointsPlannedFunctor<T, Dim> make_sort_counting_direct_singlethreaded(
    FoldRescaleRange input_range, IntBinInfo<T, Dim> const &info);

template <typename T, std::size_t Dim>
SortPointsPlannedFunctor<T, Dim> make_sort_counting_blocked_singlethreaded(
    FoldRescaleRange input_range, IntBinInfo<T, Dim> const &info);

} // namespace avx512
} // namespace spreading
} // namespace finufft
