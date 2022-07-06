#pragma once

#include "../../spreading.h"
#include "../../tracing.h"

namespace finufft {
namespace spreading {
namespace highway {

/** Sub-timers for highway bin sorting.
 *
 */
struct BinSortTimers {
    Timer total;
    Timer index_zero;            ///< Time spent setting the input array to zero
    Timer bin_index_computation; ///< Time spent calculating the bin indices
    Timer index_sort;            ///< Time spent sorting the bin indices
    Timer index_fixup;           ///< Time spend fixing up the bin indices

    BinSortTimers() = default;
    BinSortTimers(Timer &base)
        : total(base.make_timer("binsort_highway")), index_zero(total.make_timer("zero")),
          bin_index_computation(total.make_timer("comp_index")),
          index_sort(total.make_timer("sort")), index_fixup(total.make_timer("fixup")) {
    }

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
