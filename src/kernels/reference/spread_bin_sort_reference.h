#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "../../bit.h"
#include "../../spreading.h"

#include "gather_fold_reference.h"

namespace finufft {
namespace spreading {
namespace reference {

/** Basic structure which keeps track of useful
 * information about the bin structure for sorting.
 *
 */
template <typename T, std::size_t Dim> struct BinInfo {
    BinInfo(
        std::size_t num_points, std::array<T, Dim> const &extents,
        std::array<T, Dim> const &bin_sizes)
        : extents(extents) {
        for (std::size_t i = 0; i < Dim; ++i) {
            num_bins[i] = static_cast<std::size_t>(extents[i] / bin_sizes[i]) + 1;
            bin_scaling[i] = static_cast<T>(1.0 / bin_sizes[i]);
        }

        bin_stride[0] = 1;
        for (std::size_t i = 1; i < Dim; ++i) {
            bin_stride[i] = bin_stride[i - 1] * num_bins[i - 1];
        }

        bin_key_shift = bit_width(num_points);

        if (bit_width(num_bins_total()) + bit_width(num_points) > 64) {
            throw std::runtime_error("Too many bins to sort");
        }
    }

    std::array<std::size_t, Dim> num_bins;
    std::array<T, Dim> bin_scaling;
    std::array<std::size_t, Dim> bin_stride;
    std::array<T, Dim> const &extents;
    std::size_t bin_key_shift;

    std::size_t num_bins_total() const {
        std::size_t total = 1;
        for (std::size_t i = 0; i < Dim; ++i) {
            total *= num_bins[i];
        }
        return total;
    }
};


template <typename T, std::size_t Dim, typename FoldRescale>
void compute_bin_index_impl(
    int64_t *index, std::size_t num_points, std::array<T const *, Dim> const &coordinates,
    BinInfo<T, Dim> const& info,
    FoldRescale&& fold_rescale) {

    // Set-up the sorted values.
    for (std::size_t i = 0; i < num_points; ++i) {
        std::size_t bin_index = 0;
        for (std::size_t j = 0; j < Dim; ++j) {
            auto bin_index_j = static_cast<std::size_t>(
                fold_rescale(coordinates[j][i], info.extents[j]) * info.bin_scaling[j]);
            bin_index += bin_index_j * info.bin_stride[j];
        }

        index[i] = (bin_index << info.bin_key_shift) + i;
    }
}

/** Computes the bin each point belongs to.
 *
 */
template <typename T, std::size_t Dim>
void compute_bin_index(
    int64_t *index, std::size_t num_points, std::array<T const *, Dim> const &coordinates,
    std::array<T, Dim> const &extents, std::array<T, Dim> const &bin_sizes) {

    BinInfo<T, Dim> info(num_points, extents, bin_sizes);

    // Compute bin key + index
    compute_bin_index_impl(index, num_points, coordinates, info, FoldRescaleIdentity<T>{});

    // Shift to only keep the bin key.
    std::transform(index, index + num_points, index, [&info](int64_t i) {
        return i >> info.bin_key_shift;
    });
}

/** Reference implementation of bin sorting.
 *
 * This sorting strategy is based on a three-step process.
 *
 * 1) The bin index is computed for each point, and packed into the
 *    high bits of index array along with the original index (in the low bits).
 * 2) The index array is sorted
 * 3) The sorted index array is masked to only retain the point index and not
 *    the bin index.
 *
 * Note that this function is parametrized by the fold-rescale functor.
 *
 */
template <typename T, std::size_t Dim, typename FoldRescale>
void bin_sort_reference_impl(
    int64_t *index, std::size_t num_points, std::array<T const *, Dim> const &coordinates,
    std::array<T, Dim> const &extents, std::array<T, Dim> const &bin_sizes,
    FoldRescale &&fold_rescale) {

    BinInfo<T, Dim> info(num_points, extents, bin_sizes);

    // Set up key-index pairs
    compute_bin_index_impl(index, num_points, coordinates, info, fold_rescale);

    // Sort values
    std::sort(index, index + num_points);

    // Mask indices to only retain index and not bin
    // Note: in principle would be more efficient to move this responsibility to the reader
    //   (e.g. gather-rescale), but this is significantly easier to coordinate.
    auto mask = (std::size_t(1) << info.bin_key_shift) - 1;
    for (std::size_t i = 0; i < num_points; ++i) {
        index[i] &= mask;
    }
}

template <typename T, std::size_t Dim>
void bin_sort_reference(
    int64_t *index, std::size_t num_points, std::array<T const *, Dim> const &coordinates,
    std::array<T, Dim> const &extents, std::array<T, Dim> const &bin_sizes,
    FoldRescaleRange input_range) {
    if (input_range == FoldRescaleRange::Pi) {
        bin_sort_reference_impl(
            index, num_points, coordinates, extents, bin_sizes, FoldRescalePi<T>{});
    } else {
        bin_sort_reference_impl(
            index, num_points, coordinates, extents, bin_sizes, FoldRescaleIdentity<T>{});
    }
}

#define FINUFFT_BIN_DECLARE_BIN_SORT(type, dim)                                                    \
    extern template void bin_sort_reference<type, dim>(                                            \
        int64_t * index,                                                                           \
        std::size_t num_points,                                                                    \
        std::array<type const *, dim> const &coordinates,                                          \
        std::array<type, dim> const &extents,                                                      \
        std::array<type, dim> const &bin_sizes,                                                    \
        FoldRescaleRange input_range);

FINUFFT_BIN_DECLARE_BIN_SORT(float, 1)
FINUFFT_BIN_DECLARE_BIN_SORT(float, 2)
FINUFFT_BIN_DECLARE_BIN_SORT(float, 3)

FINUFFT_BIN_DECLARE_BIN_SORT(double, 1)
FINUFFT_BIN_DECLARE_BIN_SORT(double, 2)
FINUFFT_BIN_DECLARE_BIN_SORT(double, 3)

#undef FINUFFT_BIN_DECLARE_BIN_SORT

template <typename T, std::size_t Dim> BinSortFunctor<T, Dim> get_bin_sort_functor() {
    return &bin_sort_reference<T, Dim>;
}

extern template BinSortFunctor<float, 1> get_bin_sort_functor<float, 1>();
extern template BinSortFunctor<float, 2> get_bin_sort_functor<float, 2>();
extern template BinSortFunctor<float, 3> get_bin_sort_functor<float, 3>();

extern template BinSortFunctor<double, 1> get_bin_sort_functor<double, 1>();
extern template BinSortFunctor<double, 2> get_bin_sort_functor<double, 2>();
extern template BinSortFunctor<double, 3> get_bin_sort_functor<double, 3>();

}
} // namespace spreading
} // namespace finufft
