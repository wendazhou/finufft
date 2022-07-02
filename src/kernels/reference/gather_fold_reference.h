#pragma once

#include <algorithm>
#include <array>

#define _USE_MATH_DEFINES
#include <cmath>
#undef _USE_MATH_DEFINES

namespace finufft {
namespace spreading {

template <typename T> struct FoldRescalePi {
    T operator()(T x, T extent) const {
        if (x < -M_PI) {
            x += 2 * M_PI;
        } else if (x >= M_PI) {
            x -= 2 * M_PI;
        }

        return (x + M_PI) * extent * 0.5 * M_1_PI;
    }
};

template <typename T> struct FoldRescaleIdentity {
    T operator()(T x, T extent) const {
        if (x < 0) {
            x += extent;
        } else if (x >= extent) {
            x -= extent;
        }

        return x;
    }
};

/** Generic implementation of spread-rescale with parametrizable rescaling.
 *
 * See gather_and_fold for general API details.
 *
 */
template <std::size_t Dim, typename T, typename IdxT, typename RescaleFn>
void gather_and_fold_impl(
    nu_point_collection<Dim, T> const &memory, nu_point_collection<Dim, T const> const &input,
    std::array<T, Dim> const &extent, IdxT const *sort_indices, RescaleFn &&fold_rescale) {

    for (std::size_t i = 0; i < memory.num_points; ++i) {
        auto idx = sort_indices[i];

        for (int j = 0; j < Dim; ++j) {
            memory.coordinates[j][i] = fold_rescale(input.coordinates[j][idx], extent[j]);
        }

        memory.strengths[2 * i] = input.strengths[2 * idx];
        memory.strengths[2 * i + 1] = input.strengths[2 * idx + 1];
    }
}

/** Collect non-uniform points by index into contiguous array, and rescales coordinates.
 *
 * To prepare for the spreading operation, non-uniform points are collected according
 * to the indirect sorting array. Additionally, the coordinates are processed to a normalized
 * format.
 *
 * @param memory Output memory which will be filled with the gathered points.
 *               This parameter additionally implicitly specifies the number of points to gather.
 * @param input Input points to be gathered.
 * @param sizes Extent of the spreading target. Used to rescale the coordinates.
 * @param sort_indices Indirect sorting array.
 * @param rescale_range Whether the input is expected to be 2pi-periodic or extent-periodic.
 *
 */
template <std::size_t Dim, typename IdxT, typename T>
void gather_and_fold(
    nu_point_collection<Dim, T> const &memory, nu_point_collection<Dim, T const> const &input,
    std::array<int64_t, Dim> const &sizes, IdxT const *sort_indices,
    FoldRescaleRange rescale_range) {

    std::array<T, Dim> sizes_floating;
    std::copy(sizes.begin(), sizes.end(), sizes_floating.begin());

    if (rescale_range == FoldRescaleRange::Pi) {
        gather_and_fold_impl(memory, input, sizes_floating, sort_indices, FoldRescalePi<T>{});
    } else {
        gather_and_fold_impl(memory, input, sizes_floating, sort_indices, FoldRescaleIdentity<T>{});
    }
}

} // namespace spreading
} // namespace finufft
