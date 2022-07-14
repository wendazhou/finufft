#pragma once

#include <algorithm>
#include <array>

#define _USE_MATH_DEFINES
#include <cmath>
#undef _USE_MATH_DEFINES

#include "../spreading.h"

namespace finufft {
namespace spreading {

template <typename T> struct FoldRescalePi {
    T operator()(T x, T extent) const {
        // Create temporaries in correct precision
        // to have exact same behavior as vectorized versions.
        T pi = M_PI;
        T two_pi = 2 * pi;
        T one_over_2_pi = 0.5 * M_1_PI;

        if (x < -pi) {
            x += two_pi;
        } else if (x >= pi) {
            x -= two_pi;
        }

        return (x + pi) * extent * one_over_2_pi;
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

/** gather-rescale functor with reference implementation.
 *
 */
struct GatherAndFoldReferenceFunctor {
    FoldRescaleRange rescale_range_;

    template <typename T, std::size_t Dim>
    void operator()(
        nu_point_collection<Dim, T> const &memory, nu_point_collection<Dim, T const> const &input,
        std::array<int64_t, Dim> const &sizes, int64_t const *sort_indices) const {
        gather_and_fold(memory, input, sizes, sort_indices, rescale_range_);
    }
};

} // namespace spreading
} // namespace finufft
