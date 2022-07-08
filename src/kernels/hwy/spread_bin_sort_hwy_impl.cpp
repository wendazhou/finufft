/** @file
 *
 * Vectorized implementation of bin-sorting.
 *
 * This file provides a vectorized implementation of bin-sorting.
 * It makes use of google/highway to enable portable intrinsics.
 *
 * The implementation is separated into two parts (+ fixup):
 * - a bin index computation, which performs a first part to compute
 *   aggregated bin and point index packed into a 64-bit integer.
 * - a sort, which sorts all points according to the bin index.
 * - a fixup, where the high bits of the index are masked to recover
 *   the point index.
 *
 */

#include "spread_bin_sort_hwy.h"

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "../../bit.h"
#include "../reference/spread_bin_sort_reference.h"

#include <hwy/contrib/sort/vqsort.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "spread_bin_sort_hwy_impl.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "fold_rescale.inl.h"

HWY_BEFORE_NAMESPACE();
namespace finufft {
namespace spreading {
namespace highway {
namespace HWY_NAMESPACE {

namespace hn = ::hwy::HWY_NAMESPACE;

using reference::BinInfo;

/** Computes packed bin-key point-index pairs and puts them into the index array.
 *
 * This function computes the bin key for each point, packs it with the original
 * point index, and puts it into the index array. The output of the index array
 * corresponds to elements such that the high bits contain the bin key, and the
 * low bits contain the point index.
 *
 * This is done to ensure that we can recover an indirect sort of the input points
 * by directly sorting the index array.
 *
 * @param[out] index Array of size num_points to store the bin-key point-index packed pairs.
 * @param num_points Number of points to sort.
 * @param coordinates Array of pointers to the coordinates of the points.
 * @param info Information about the bins.
 * @param fold_rescale Function to rescale the coordinates.
 *
 */
template <std::size_t Dim, typename FoldRescale>
void compute_bin_index_impl(
    int64_t *index, std::size_t num_points, std::array<float const *, Dim> const &coordinates,
    BinInfo<float, Dim> const &info, FoldRescale &&fold_rescale) {

    // Set up for main loop
    std::size_t i = 0;
    {
        hn::ScalableTag<float> d;
        hn::ScalableTag<uint32_t> di;
        hn::ScalableTag<uint64_t> di64;
        hn::ScalableTag<int32_t> di_s;

        auto index_v_both = hn::Iota(di, 0);
        auto index_even = hn::MulEven(index_v_both, hn::Set(di, 1));
        auto index_odd = hn::MulEven(hn::Reverse2(di, index_v_both), hn::Set(di, 1));

        const auto N = hn::Lanes(d);

        for (; i + N < num_points; i += N) {
            auto bin_index_even = hn::Zero(di64);
            auto bin_index_odd = hn::Zero(di64);

            for (std::size_t dim = 0; dim < Dim; ++dim) {
                auto folded =
                    fold_rescale(hn::LoadU(d, coordinates[dim] + i), info.extents[dim], d);
                auto bin_floating = hn::Mul(folded, hn::Set(d, info.bin_scaling[dim]));
                auto bin_int = hn::BitCast(di, hn::ConvertTo(di_s, bin_floating));

                auto bin_index_even_dim = hn::MulEven(bin_int, hn::Set(di, info.bin_stride[dim]));
                auto bin_index_odd_dim =
                    hn::MulEven(hn::Reverse2(di, bin_int), hn::Set(di, info.bin_stride[dim]));

                bin_index_even = hn::Add(bin_index_even, bin_index_even_dim);
                bin_index_odd = hn::Add(bin_index_odd, bin_index_odd_dim);
            }

            bin_index_even =
                hn::ShiftLeftSame(bin_index_even, static_cast<int>(info.bin_key_shift));
            bin_index_odd = hn::ShiftLeftSame(bin_index_odd, static_cast<int>(info.bin_key_shift));

            bin_index_even = hn::Add(bin_index_even, index_even);
            bin_index_odd = hn::Add(bin_index_odd, index_odd);

            index_even = hn::Add(index_even, hn::Set(di64, N));
            index_odd = hn::Add(index_odd, hn::Set(di64, N));

            // Note: OK to store not in order, since we're going to sort in any case.
            // Our store order here unpacks even and odd elements by group.
            hn::Store(bin_index_even, di64, reinterpret_cast<uint64_t *>(index) + i);
            hn::Store(
                bin_index_odd, di64, reinterpret_cast<uint64_t *>(index) + i + hn::Lanes(di64));
        }
    }
    {
        // process tail loop
        for (; i < num_points; ++i) {
            std::size_t bin_index = 0;
            for (std::size_t j = 0; j < Dim; ++j) {
                auto bin_index_j = static_cast<std::size_t>(
                    fold_rescale(coordinates[j][i], info.extents[j]) * info.bin_scaling[j]);
                bin_index += bin_index_j * info.bin_stride[j];
            }

            index[i] = (bin_index << info.bin_key_shift) + i;
        }
    }
}

template <std::size_t Dim, typename FoldRescale>
void compute_bin_index_impl(
    int64_t *index, std::size_t num_points, std::array<double const *, Dim> const &coordinates,
    BinInfo<double, Dim> const &info, FoldRescale &&fold_rescale) {

    // Set up for main loop
    std::size_t i = 0;
    {
        hn::ScalableTag<double> d;
        hn::ScalableTag<uint32_t> di;
        hn::ScalableTag<uint64_t> di64;
        hn::ScalableTag<int64_t> di64_s;

        auto i_v = hn::Iota(di64, 0);
        const auto N = hn::Lanes(d);

        for (; i + N < num_points; i += N) {
            auto bin_index = hn::Zero(di64);

            for (std::size_t dim = 0; dim < Dim; ++dim) {
                auto folded =
                    fold_rescale(hn::LoadU(d, coordinates[dim] + i), info.extents[dim], d);
                auto bin_floating = hn::Mul(folded, hn::Set(d, info.bin_scaling[dim]));

                auto bin_int = hn::BitCast(di, hn::ConvertTo(di64_s, bin_floating));
                bin_index =
                    hn::Add(bin_index, hn::MulEven(bin_int, hn::Set(di, info.bin_stride[dim])));
            }

            bin_index = hn::ShiftLeftSame(bin_index, static_cast<int>(info.bin_key_shift));
            bin_index = hn::Add(bin_index, i_v);

            hn::Store(bin_index, di64, reinterpret_cast<uint64_t *>(index) + i);

            i_v = hn::Add(i_v, hn::Set(di64, N));
        }
    }
    {
        // process tail loop
        for (; i < num_points; ++i) {
            std::size_t bin_index = 0;
            for (std::size_t j = 0; j < Dim; ++j) {
                auto bin_index_j = static_cast<std::size_t>(
                    fold_rescale(coordinates[j][i], info.extents[j]) * info.bin_scaling[j]);
                bin_index += bin_index_j * info.bin_stride[j];
            }

            index[i] = (bin_index << info.bin_key_shift) + i;
        }
    }
}

/** Implementation of bin index computation.
 *
 * This function packs a bin index and the original index
 * into a 64-bit integer, with the bin index being placed
 * in the high bits and the original index in the low bits.
 *
 */
template <typename T, std::size_t Dim>
void compute_bin_index(
    int64_t *index, std::size_t num_points, std::array<T const *, Dim> const &coordinates,
    BinInfo<T, Dim> const &info, FoldRescaleRange rescale_range) {
    if (rescale_range == FoldRescaleRange::Pi) {
        compute_bin_index_impl(index, num_points, coordinates, info, FoldRescalePi<T>{});
    } else {
        compute_bin_index_impl(index, num_points, coordinates, info, FoldRescaleIdentity<T>{});
    }
}

/** Uses a counting sort to sort the bin indices.
 *
 * The counting sort may be faster (compared to a quicksort) when
 * the number of bins is small, and when the data fits in cache.
 * It may potentially be easier to fully parallelize?
 *
 */
template <typename T, std::size_t Dim>
inline void counting_sort_with_fixup(
    std::size_t num, uint64_t *__restrict values, BinInfo<T, Dim> const &info,
    BinSortTimers &timers) {

    auto bin_counts_holder = allocate_aligned_array<uint64_t>(info.num_bins_total(), 64);
    auto bin_counts = bin_counts_holder.get();

    {
        ScopedTimerGuard guard(timers.countsort_allocate);
        std::memset(bin_counts, 0, info.num_bins_total() * sizeof(uint64_t));
    }

    {
        ScopedTimerGuard guard(timers.countsort_count);
        for (std::size_t i = 0; i < num; ++i) {
            auto k = values[i] >> info.bin_key_shift;
            bin_counts[k]++;
        }
    }

    {
        ScopedTimerGuard guard(timers.countsort_cumsum);
        std::partial_sum(bin_counts, bin_counts + info.num_bins_total(), bin_counts);
    }

    {
        ScopedTimerGuard guard(timers.countsort_spread);
        auto values_target = allocate_aligned_array<uint64_t>(num, 64);

        for (std::size_t i = num; i >= 1; --i) {
            auto k = values[i - 1] >> info.bin_key_shift;
            values_target[--bin_counts[k]] = values[i - 1] << info.bin_key_shift;
        }

        std::memcpy(values, values_target.get(), num * sizeof(uint64_t));
    }
}

/** Uses a standard (quicksort) strategy to sort the bins, then fixes up the
 * array to only contain indices.
 *
 * TODO: integrate a parallel sorter such as ips4o in addition to the vectorized
 * scalar sorter currently being used.
 *
 */
template <typename T, std::size_t Dim>
inline void quicksort_and_fixup(
    std::size_t num, uint64_t *__restrict index, BinInfo<T, Dim> const &info,
    BinSortTimers &timers) {
    // Step 2: Sort the bins
    {
        ScopedTimerGuard guard(timers.quicksort_sort);
        hwy::Sorter{}(index, num, hwy::SortAscending{});
    }

    // Step 3: fixup the index by masking out the index.
    {
        ScopedTimerGuard guard(timers.quicksort_fixup);
        auto mask = (size_t(1) << info.bin_key_shift) - 1;

        {
            hn::ScalableTag<uint64_t> di;

            const auto N = hn::Lanes(di);
            auto mask_v = hn::Set(di, mask);

            // Note: rely on alignment and padding of underlying array here.
            for (std::size_t i = 0; i < num; i += N) {
                auto v = hn::Load(di, index + i);
                v = hn::And(v, mask_v);
                hn::Store(v, di, index + i);
            }
        }
    }
}

template <typename T, std::size_t Dim>
void bin_sort_generic(
    int64_t *index, std::size_t num_points, std::array<T const *, Dim> const &coordinates,
    std::array<T, Dim> const &extents, std::array<T, Dim> const &bin_sizes,
    FoldRescaleRange input_range, BinSortTimers &timers) {

    ScopedTimerGuard guard(timers.total);

    BinInfo<T, Dim> info(num_points, extents, bin_sizes);

    // Step 1: compute bin-original index pairs in index.
    {
        ScopedTimerGuard guard(timers.bin_index_computation);
        compute_bin_index(index, num_points, coordinates, info, input_range);
    }

    // Step 2: directly sort the bin-original index pairs.
    quicksort_and_fixup(num_points, reinterpret_cast<uint64_t *>(index), info, timers);
    // counting_sort_with_fixup(num_points, reinterpret_cast<uint64_t *>(index), info, timers);
}

#define BIN_SORT_NAME(type, dim) bin_sort_##type##_##dim

#define INSTANTIATE_BIN_SORT(type, dim)                                                            \
    void BIN_SORT_NAME(type, dim)(                                                                 \
        int64_t * index,                                                                           \
        std::size_t num_points,                                                                    \
        std::array<type const *, dim> const &coordinates,                                          \
        std::array<type, dim> const &extents,                                                      \
        std::array<type, dim> const &bin_sizes,                                                    \
        FoldRescaleRange rescale_range,                                                            \
        BinSortTimers &timers) {                                                                   \
        bin_sort_generic<type, dim>(                                                               \
            index, num_points, coordinates, extents, bin_sizes, rescale_range, timers);            \
    }

INSTANTIATE_BIN_SORT(float, 1)
INSTANTIATE_BIN_SORT(float, 2)
INSTANTIATE_BIN_SORT(float, 3)

INSTANTIATE_BIN_SORT(double, 1)
INSTANTIATE_BIN_SORT(double, 2)
INSTANTIATE_BIN_SORT(double, 3)

#undef INSTANTIATE_BIN_SORT

} // namespace HWY_NAMESPACE
} // namespace highway
} // namespace spreading
} // namespace finufft
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace finufft {
namespace spreading {
namespace highway {

#define EXPORT_AND_INSTANTIATE(type, dim)                                                          \
    HWY_EXPORT(BIN_SORT_NAME(type, dim));                                                          \
    template <>                                                                                    \
    void bin_sort(                                                                                 \
        int64_t *index,                                                                            \
        std::size_t num_points,                                                                    \
        std::array<type const *, dim> const &coordinates,                                          \
        std::array<type, dim> const &extents,                                                      \
        std::array<type, dim> const &bin_sizes,                                                    \
        FoldRescaleRange rescale_range,                                                            \
        BinSortTimers &timers) {                                                                   \
        HWY_DYNAMIC_DISPATCH(BIN_SORT_NAME(type, dim))                                             \
        (index, num_points, coordinates, extents, bin_sizes, rescale_range, timers);               \
    }

EXPORT_AND_INSTANTIATE(float, 1)
EXPORT_AND_INSTANTIATE(float, 2)
EXPORT_AND_INSTANTIATE(float, 3)

EXPORT_AND_INSTANTIATE(double, 1)
EXPORT_AND_INSTANTIATE(double, 2)
EXPORT_AND_INSTANTIATE(double, 3)

#undef EXPORT_AND_INSTANTIATE

template <typename T, std::size_t Dim> BinSortFunctor<T, Dim> get_bin_sort_functor(Timer *timer) {
    return [timers = timer ? BinSortTimers(*timer) : BinSortTimers()](
               int64_t *index,
               std::size_t num_points,
               std::array<T const *, Dim> const &coordinates,
               std::array<T, Dim> const &extents,
               std::array<T, Dim> const &bin_sizes,
               FoldRescaleRange rescale_range) {
        BinSortTimers timers_copy(timers);
        bin_sort<T, Dim>(index, num_points, coordinates, extents, bin_sizes, rescale_range, timers_copy);
    };
}

template <typename T, std::size_t Dim>
void bin_sort(
    int64_t *index, std::size_t num_points, std::array<T const *, Dim> const &coordinates,
    std::array<T, Dim> const &extents, std::array<T, Dim> const &bin_sizes,
    FoldRescaleRange rescale_range) {
    BinSortTimers timers;
    bin_sort(index, num_points, coordinates, extents, bin_sizes, rescale_range, timers);
}

#define INSTANTIATE_TEMPLATES(T, Dim)                                                              \
    template void bin_sort<T, Dim>(                                                                \
        int64_t * index,                                                                           \
        std::size_t num_points,                                                                    \
        std::array<T const *, Dim> const &coordinates,                                             \
        std::array<T, Dim> const &extents,                                                         \
        std::array<T, Dim> const &bin_sizes,                                                       \
        FoldRescaleRange rescale_range);                                                           \
    template BinSortFunctor<T, Dim> get_bin_sort_functor<T, Dim>(Timer * timer = nullptr);

INSTANTIATE_TEMPLATES(float, 1)
INSTANTIATE_TEMPLATES(float, 2)
INSTANTIATE_TEMPLATES(float, 3)

INSTANTIATE_TEMPLATES(double, 1)
INSTANTIATE_TEMPLATES(double, 2)
INSTANTIATE_TEMPLATES(double, 3)

#undef INSTANTIATE_TEMPLATES

} // namespace highway
} // namespace spreading
} // namespace finufft

#endif

#undef BIN_SORT_NAME
