#pragma once

#include "../../spreading.h"

namespace finufft {
namespace spreading {
namespace highway {

template <typename T, std::size_t Dim>
void gather_and_fold_hwy(
    nu_point_collection<Dim, T> const &memory, nu_point_collection<Dim, T const> const &input,
    std::array<int64_t, Dim> const &sizes, int64_t const *sort_indices,
    FoldRescaleRange rescale_range);

extern template void gather_and_fold_hwy<float, 1>(
    nu_point_collection<1, float> const &memory, nu_point_collection<1, float const> const &input,
    std::array<int64_t, 1> const &sizes, int64_t const *sort_indices,
    FoldRescaleRange rescale_range);
extern template void gather_and_fold_hwy<float, 2>(
    nu_point_collection<2, float> const &memory, nu_point_collection<2, float const> const &input,
    std::array<int64_t, 2> const &sizes, int64_t const *sort_indices,
    FoldRescaleRange rescale_range);
extern template void gather_and_fold_hwy<float, 3>(
    nu_point_collection<3, float> const &memory, nu_point_collection<3, float const> const &input,
    std::array<int64_t, 3> const &sizes, int64_t const *sort_indices,
    FoldRescaleRange rescale_range);

template <typename T, std::size_t Dim> GatherRescaleFunctor<T, Dim> get_gather_rescale_hwy() {
    return &gather_and_fold_hwy<T, Dim>;
}

} // namespace hwy
} // namespace spreading
} // namespace finufft
