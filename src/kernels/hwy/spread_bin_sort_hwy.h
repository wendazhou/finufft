#pragma once

#include "../spreading.h"
#include "../../tracing.h"

namespace finufft {
namespace spreading {
namespace highway {

/** Sub-timers for highway bin sorting.
 * 
 * Note that depending on the number of bins, the bin sort
 * may choose to use a quick sort or a counting sort strategy,
 * hence only one set of quicksort* or countsort* timers is used.
 *
 */
struct BinSortTimers {
    Timer total;
    Timer bin_index_computation; ///< Time spent calculating the bin indices
    Timer quicksort_sort;        ///< Time spent sorting the bin indices
    Timer quicksort_fixup;       ///< Time spent fixing up the bin indices
    Timer countsort_allocate;    ///< Time spent allocating / zeroing the bin counts
    Timer countsort_count;       ///< Time spent counting the number of elements in each bin
    Timer countsort_cumsum;      ///< Time spent computing the cumulative sum of the bin counts
    Timer countsort_spread;      ///< Time spent spreading the elements in each bin

    BinSortTimers() = default;
    BinSortTimers(Timer &base)
        : total(base.make_timer("binsort_highway")),
          bin_index_computation(total.make_timer("comp_index")),
          quicksort_sort(total.make_timer("qs/sort")),
          quicksort_fixup(total.make_timer("qs/fixup")),
          countsort_count(total.make_timer("cs/count")),
          countsort_cumsum(total.make_timer("cs/cumsum")),
          countsort_spread(total.make_timer("cs/spread")) {}
    BinSortTimers(BinSortTimers const &) = default;
    BinSortTimers(BinSortTimers &&) = default;
};

template <typename T, std::size_t Dim>
void bin_sort(
    int64_t *index, std::size_t num_points, std::array<T const *, Dim> const &coordinates,
    std::array<T, Dim> const &extents, std::array<T, Dim> const &bin_sizes,
    FoldRescaleRange rescale_range, BinSortTimers &);

template <typename T, std::size_t Dim>
void bin_sort(
    int64_t *index, std::size_t num_points, std::array<T const *, Dim> const &coordinates,
    std::array<T, Dim> const &extents, std::array<T, Dim> const &bin_sizes,
    FoldRescaleRange rescale_range);

/** Gets a functor for bin-sorting using the highway implementation.
 *
 * If timer is provided, subtiming information will be additionally accumulated
 * into the given timer and subtimers.
 *
 */
template <typename T, std::size_t Dim>
BinSortFunctor<T, Dim> get_bin_sort_functor(Timer *timer = nullptr);

} // namespace highway
} // namespace spreading
} // namespace finufft
