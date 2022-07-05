#pragma once

#include "../../spreading.h"

namespace finufft {
namespace spreading {
namespace highway {
template <typename T, std::size_t Dim>
void bin_sort(
    int64_t *index, std::size_t num_points, std::array<T const *, Dim> const &coordinates,
    std::array<T, Dim> const &extents, std::array<T, Dim> const &bin_sizes,
    FoldRescaleRange rescale_range);

#define DECLARE_TEMPLATE(T, Dim)                                                                   \
    extern template void bin_sort(                                                                 \
        int64_t *index,                                                                            \
        std::size_t num_points,                                                                    \
        std::array<T const *, Dim> const &coordinates,                                             \
        std::array<T, Dim> const &extents,                                                         \
        std::array<T, Dim> const &bin_sizes,                                                       \
        FoldRescaleRange rescale_range);

DECLARE_TEMPLATE(float, 1)
DECLARE_TEMPLATE(float, 2)
DECLARE_TEMPLATE(float, 3)

DECLARE_TEMPLATE(double, 1)
DECLARE_TEMPLATE(double, 2)
DECLARE_TEMPLATE(double, 3)

#undef DECLARE_TEMPLATE

template <typename T, std::size_t Dim> BinSortFunctor<T, Dim> get_bin_sort_functor() {
    return &bin_sort<T, Dim>;
}

} // namespace highway
} // namespace spreading
} // namespace finufft
