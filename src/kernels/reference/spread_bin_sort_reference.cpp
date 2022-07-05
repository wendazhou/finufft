#include "spread_bin_sort_reference.h"

namespace finufft {
    namespace spreading {

#define FINUFFT_BIN_DEFINE_BIN_SORT(type, dim)                                                    \
    template void bin_sort_reference<type, dim>(                                            \
        int64_t * index,                                                                           \
        std::size_t num_points,                                                                    \
        std::array<type const *, dim> const &coordinates,                                          \
        std::array<type, dim> const &extents,                                                      \
        std::array<type, dim> const &bin_sizes,                                                    \
        FoldRescaleRange input_range);

FINUFFT_BIN_DEFINE_BIN_SORT(float, 1)
FINUFFT_BIN_DEFINE_BIN_SORT(float, 2)
FINUFFT_BIN_DEFINE_BIN_SORT(float, 3)

FINUFFT_BIN_DEFINE_BIN_SORT(double, 1)
FINUFFT_BIN_DEFINE_BIN_SORT(double, 2)
FINUFFT_BIN_DEFINE_BIN_SORT(double, 3)

#undef FINUFFT_BIN_DECLARE_BIN_SORT

template BinSortFunctor<float, 1> get_bin_sort_functor_reference<float, 1>();
template BinSortFunctor<float, 2> get_bin_sort_functor_reference<float, 2>();
template BinSortFunctor<float, 3> get_bin_sort_functor_reference<float, 3>();

template BinSortFunctor<double, 1> get_bin_sort_functor_reference<double, 1>();
template BinSortFunctor<double, 2> get_bin_sort_functor_reference<double, 2>();
template BinSortFunctor<double, 3> get_bin_sort_functor_reference<double, 3>();

    }
}
