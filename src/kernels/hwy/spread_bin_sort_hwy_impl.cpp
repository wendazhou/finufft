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

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "../../bit.h"

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

template <std::size_t Dim, typename FoldRescale>
void compute_bin_index_impl(
    int64_t *index, std::size_t num_points, std::array<float const *, Dim> const &coordinates,
    std::array<float, Dim> const &extents, std::array<float, Dim> const &bin_sizes,
    FoldRescale &&fold_rescale) {

    // Pre-compute information
    std::array<std::size_t, Dim> num_bins;
    std::array<float, Dim> bin_scaling;

    for (std::size_t i = 0; i < Dim; ++i) {
        num_bins[i] = static_cast<std::size_t>(extents[i] / bin_sizes[i]) + 1;
        bin_scaling[i] = static_cast<float>(1. / bin_sizes[i]);
    }

    std::array<std::size_t, Dim> stride;
    stride[0] = 1;
    for (std::size_t i = 1; i < Dim; ++i) {
        stride[i] = stride[i - 1] * num_bins[i - 1];
    }

    std::size_t bins_total = stride[Dim - 1] * num_bins[Dim - 1];

    if (bit_width(bins_total) + bit_width(num_points) > 64) {
        throw std::runtime_error("Too many bins to sort");
    }

    std::size_t points_bits = bit_width(num_points);

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
                auto folded = fold_rescale(hn::LoadU(d, coordinates[dim] + i), extents[dim], d);
                auto bin_floating = hn::Mul(folded, hn::Set(d, bin_scaling[dim]));
                auto bin_int = hn::BitCast(di, hn::ConvertTo(di_s, bin_floating));

                auto bin_index_even_dim = hn::MulEven(bin_int, hn::Set(di, stride[dim]));
                auto bin_index_odd_dim =
                    hn::MulEven(hn::Reverse2(di, bin_int), hn::Set(di, stride[dim]));

                bin_index_even = hn::Add(bin_index_even, bin_index_even_dim);
                bin_index_odd = hn::Add(bin_index_odd, bin_index_odd_dim);
            }

            bin_index_even = hn::ShiftLeftSame(bin_index_even, static_cast<int>(points_bits));
            bin_index_odd = hn::ShiftLeftSame(bin_index_odd, static_cast<int>(points_bits));

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
                    fold_rescale(coordinates[j][i], extents[j]) * bin_scaling[j]);
                bin_index += bin_index_j * stride[j];
            }

            index[i] = (bin_index << points_bits) + i;
        }
    }
}

template <std::size_t Dim, typename FoldRescale>
void compute_bin_index_impl(
    int64_t *index, std::size_t num_points, std::array<double const *, Dim> const &coordinates,
    std::array<double, Dim> const &extents, std::array<double, Dim> const &bin_sizes,
    FoldRescale &&fold_rescale) {

    // Pre-compute information
    std::array<std::size_t, Dim> num_bins;
    std::array<double, Dim> bin_scaling;

    for (std::size_t i = 0; i < Dim; ++i) {
        num_bins[i] = static_cast<std::size_t>(extents[i] / bin_sizes[i]) + 1;
        bin_scaling[i] = static_cast<double>(1. / bin_sizes[i]);
    }

    std::array<std::size_t, Dim> stride;
    stride[0] = 1;
    for (std::size_t i = 1; i < Dim; ++i) {
        stride[i] = stride[i - 1] * num_bins[i - 1];
    }

    std::size_t bins_total = stride[Dim - 1] * num_bins[Dim - 1];

    if (bit_width(bins_total) + bit_width(num_points) > 64) {
        throw std::runtime_error("Too many bins to sort");
    }

    std::size_t points_bits = bit_width(num_points);

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
                auto folded = fold_rescale(hn::LoadU(d, coordinates[dim] + i), extents[dim], d);
                auto bin_floating = hn::Mul(folded, hn::Set(d, bin_scaling[dim]));

                auto bin_int = hn::BitCast(di, hn::ConvertTo(di64_s, bin_floating));
                bin_index = hn::Add(bin_index, hn::MulEven(bin_int, hn::Set(di, stride[dim])));
            }

            bin_index = hn::ShiftLeftSame(bin_index, static_cast<int>(points_bits));
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
                    fold_rescale(coordinates[j][i], extents[j]) * bin_scaling[j]);
                bin_index += bin_index_j * stride[j];
            }

            index[i] = (bin_index << points_bits) + i;
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
    std::array<T, Dim> const &extents, std::array<T, Dim> const &bin_sizes,
    FoldRescaleRange rescale_range) {
    if (rescale_range == FoldRescaleRange::Pi) {
        compute_bin_index_impl(
            index, num_points, coordinates, extents, bin_sizes, FoldRescalePi<T>{});
    } else {
        compute_bin_index_impl(
            index, num_points, coordinates, extents, bin_sizes, FoldRescaleIdentity<T>{});
    }
}

template <typename T, std::size_t Dim>
void bin_sort(
    int64_t *index, std::size_t num_points, std::array<T const *, Dim> const &coordinates,
    std::array<T, Dim> const &extents, std::array<T, Dim> const &bin_sizes,
    FoldRescaleRange input_range) {

    // Zero memory
    std::memset(index, 0, sizeof(int64_t) * num_points);

    // Step 1: compute bin-original index pairs in index.
    compute_bin_index(index, num_points, coordinates, extents, bin_sizes, input_range);

    // Step 2: directly sort the bin-original index pairs.
    hwy::Sorter{}(index, num_points, hwy::SortAscending{});

    // Step 3: fixup the index by masking out the index.
    std::size_t point_bits = bit_width(num_points);
    auto mask = (size_t(1) << point_bits) - 1;

    {
        hn::ScalableTag<int64_t> di;

        const auto N = hn::Lanes(di);
        auto mask_v = hn::Set(di, mask);

        // Note: rely on alignment and padding of underlying array here.
        for (std::size_t i = 0; i < num_points; i += N) {
            auto v = hn::Load(di, index + i);
            v = hn::And(v, mask_v);
            hn::Store(v, di, index + i);
        }
    }
}

#define BIN_SORT_NAME(type, dim) bin_sort_##type##_##dim

#define INSTANTIATE_BIN_SORT(type, dim)                                                            \
    void BIN_SORT_NAME(type, dim)(                                                                 \
        int64_t * index,                                                                           \
        std::size_t num_points,                                                                    \
        std::array<type const *, dim> const &coordinates,                                          \
        std::array<type, dim> const &extents,                                                      \
        std::array<type, dim> const &bin_sizes,                                                    \
        FoldRescaleRange rescale_range) {                                                          \
        bin_sort<type, dim>(index, num_points, coordinates, extents, bin_sizes, rescale_range);    \
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

template <typename T, std::size_t Dim>
void bin_sort(
    int64_t *index, std::size_t num_points, std::array<T const *, Dim> const &coordinates,
    std::array<T, Dim> const &extents, std::array<T, Dim> const &bin_sizes,
    FoldRescaleRange rescale_range);

#define EXPORT_AND_INSTANTIATE(type, dim)                                                          \
    HWY_EXPORT(BIN_SORT_NAME(type, dim));                                                          \
    template <>                                                                                    \
    void bin_sort(                                                                                 \
        int64_t *index,                                                                            \
        std::size_t num_points,                                                                    \
        std::array<type const *, dim> const &coordinates,                                          \
        std::array<type, dim> const &extents,                                                      \
        std::array<type, dim> const &bin_sizes,                                                    \
        FoldRescaleRange rescale_range) {                                                          \
        HWY_DYNAMIC_DISPATCH(BIN_SORT_NAME(type, dim))                                             \
        (index, num_points, coordinates, extents, bin_sizes, rescale_range);                       \
    }

EXPORT_AND_INSTANTIATE(float, 1)
EXPORT_AND_INSTANTIATE(float, 2)
EXPORT_AND_INSTANTIATE(float, 3)

EXPORT_AND_INSTANTIATE(double, 1)
EXPORT_AND_INSTANTIATE(double, 2)
EXPORT_AND_INSTANTIATE(double, 3)

#undef EXPORT_AND_INSTANTIATE

} // namespace highway
} // namespace spreading
} // namespace finufft

#endif

#undef BIN_SORT_NAME
