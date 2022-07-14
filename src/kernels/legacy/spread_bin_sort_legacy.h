#pragma once

#include "../spreading.h"
#include <cstdint>

namespace finufft {
namespace spreadinterp {
void bin_sort_singlethread(
    int64_t *ret, int64_t M, float *kx, float *ky, float *kz, int64_t N1, int64_t N2, int64_t N3,
    int pirange, double bin_size_x, double bin_size_y, double bin_size_z, int debug);
void bin_sort_singlethread(
    int64_t *ret, int64_t M, double *kx, double *ky, double *kz, int64_t N1, int64_t N2, int64_t N3,
    int pirange, double bin_size_x, double bin_size_y, double bin_size_z, int debug);
} // namespace spreadinterp
} // namespace finufft

namespace finufft {
namespace spreading {

template <typename T, std::size_t Dim>
void bin_sort_singlethread_legacy(
    int64_t *ret, int64_t num_points, std::array<const T *, Dim> const &coordinates,
    std::array<T, Dim> const &extents, std::array<T, Dim> const &bin_sizes,
    FoldRescaleRange input_range) {

    static_assert(Dim <= 3, "Only 1D, 2D, and 3D are supported");
    static_assert(Dim != 0, "Dimension must be greater than 0");

    finufft::spreadinterp::bin_sort_singlethread(
        ret,
        num_points,
        const_cast<T *>(coordinates[0]),
        Dim > 1 ? const_cast<T *>(coordinates[1]) : nullptr,
        Dim > 2 ? const_cast<T *>(coordinates[2]) : nullptr,
        extents[0],
        Dim > 1 ? extents[1] : 1,
        Dim > 2 ? extents[2] : 1,
        input_range == FoldRescaleRange::Pi,
        bin_sizes[0],
        Dim > 1 ? bin_sizes[1] : 1,
        Dim > 2 ? bin_sizes[2] : 1,
        0);
}

template <typename T, std::size_t Dim> BinSortFunctor<T, Dim> get_bin_sort_functor_legacy() {
    return &bin_sort_singlethread_legacy<T, Dim>;
}

} // namespace spreading
} // namespace finufft
