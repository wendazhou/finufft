#pragma once

#include "../../spreading.h"

namespace finufft {

namespace spreading {

namespace avx512 {

template <typename T, std::size_t Dim>
void gather_fold_avx512_impl(
    nu_point_collection<Dim, T> const &memory, nu_point_collection<Dim, T const> const &input,
    std::array<int64_t, Dim> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) noexcept;

extern template void gather_fold_avx512_impl<float, 1>(
    nu_point_collection<1, float> const &memory, nu_point_collection<1, float const> const &input,
    std::array<int64_t, 1> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) noexcept;
extern template void gather_fold_avx512_impl<float, 2>(
    nu_point_collection<2, float> const &memory, nu_point_collection<2, float const> const &input,
    std::array<int64_t, 2> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) noexcept;
extern template void gather_fold_avx512_impl<float, 3>(
    nu_point_collection<3, float> const &memory, nu_point_collection<3, float const> const &input,
    std::array<int64_t, 3> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) noexcept;

extern template void gather_fold_avx512_impl<double, 1>(
    nu_point_collection<1, double> const &memory, nu_point_collection<1, double const> const &input,
    std::array<int64_t, 1> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) noexcept;
extern template void gather_fold_avx512_impl<double, 2>(
    nu_point_collection<2, double> const &memory, nu_point_collection<2, double const> const &input,
    std::array<int64_t, 2> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) noexcept;
extern template void gather_fold_avx512_impl<double, 3>(
    nu_point_collection<3, double> const &memory, nu_point_collection<3, double const> const &input,
    std::array<int64_t, 3> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) noexcept;

} // namespace avx512

/** Main functor implementing AVX-512 gather and fold.
 *
 */
struct GatherFoldAvx512 {
    template <typename T, std::size_t Dim>
    void operator()(
        nu_point_collection<Dim, T> const &memory, nu_point_collection<Dim, typename identity<T>::type const> const &input,
        std::array<int64_t, Dim> const &sizes, std::int64_t const *sort_indices,
        FoldRescaleRange rescale_range) const {
        avx512::gather_fold_avx512_impl<T, Dim>(memory, input, sizes, sort_indices, rescale_range);
    }
};

/** Function object which encapsulates the AVX-512 gather and fold implementation.
 *
 * This function encapsulates the implementation of the AVX-512 gather and fold for
 * dimension 1, 2, 3 in single and double precision.
 *
 */
extern const GatherFoldAvx512 gather_and_fold_avx512;

struct GatherFoldAvx512Functor {
    FoldRescaleRange rescale_range_;

    template <typename T, std::size_t Dim>
    void operator()(
        nu_point_collection<Dim, T> const &memory, nu_point_collection<Dim, T const> const &input,
        std::array<int64_t, Dim> const &sizes, int64_t const *sort_indices) const {
        GatherFoldAvx512{}(memory, input, sizes, sort_indices, rescale_range_);
    }
};

} // namespace spreading

} // namespace finufft
