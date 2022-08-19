#include "sort_bin_counting.h"

#include <libdivide.h>

#include <array>
#include <cmath>
#include <cstring>
#include <tcb/span.hpp>

#include "../../memory.h"
#include "../sorting.h"
#include "../spreading.h"
#include "gather_fold_reference.h"
#include "sort_bin_counting_impl.h"

namespace finufft {
namespace spreading {
namespace reference {

namespace {
template <std::size_t Unroll, typename T, std::size_t Dim, typename FoldRescale>
struct ComputeBinIndex {
    IntBinInfo<T, Dim> info;
    FoldRescale fold_rescale;
    std::array<T, Dim> size_f;
    std::array<libdivide::divider<uint32_t>, Dim> dividers;

    static constexpr std::size_t unroll = Unroll;
    typedef std::uint32_t index_type;

    explicit ComputeBinIndex(IntBinInfo<T, Dim> const &info, FoldRescale const &fold_rescale)
        : info(info), fold_rescale(fold_rescale) {
        std::copy(info.size.begin(), info.size.end(), size_f.begin());
        for (std::size_t j = 0; j < Dim; ++j) {
            dividers[j] = libdivide::divider<uint32_t>(info.bin_size[j]);
        }

        // Basic error checking for valid 32-bit processing of input.
        if (info.num_bins_total() > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Too many bins");
        }

        for (std::size_t dim = 0; dim < Dim; ++dim) {
            if (info.size[dim] > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("Grid too large");
            }
        }
    }

    template <bool Partial, typename WriteTransformedCoordinate>
    void operator()(
        nu_point_collection<Dim, const T> const &input, std::size_t i, std::size_t limit,
        tcb::span<std::uint32_t, Unroll> bins, std::integral_constant<bool, Partial>,
        WriteTransformedCoordinate &&write_transformed_coordinate) const {
        std::fill(bins.begin(), bins.end(), 0);

        for (std::size_t j = 0; j < Dim; ++j) {
            for (std::size_t offset = 0; offset < (Partial ? limit : Unroll); ++offset) {
                auto x = input.coordinates[j][i + offset];
                x = fold_rescale(x, size_f[j]);

                auto x_c = std::ceil(x - info.offset[j]);
                std::uint32_t x_b =
                    static_cast<uint32_t>(x_c) - static_cast<uint32_t>(info.global_offset[j]);
                x_b /= dividers[j];

                bins[offset] += x_b * info.bin_index_stride[j];
                write_transformed_coordinate(j, offset, x);
            }
        }
    }

    template <bool Partial>
    void operator()(
        nu_point_collection<Dim, const T> const &input, std::size_t i, std::size_t limit,
        tcb::span<std::uint32_t, Unroll> bins,
        std::integral_constant<bool, Partial> partial) const {
        return (*this)(
            input, i, limit, bins, partial, [](std::size_t j, std::size_t offset, T x) {});
    }
};

template <typename T, FoldRescaleRange input_range> struct FoldRescaleScalar;
template <typename T>
struct FoldRescaleScalar<T, FoldRescaleRange::Identity> : FoldRescaleIdentity<T> {};
template <typename T> struct FoldRescaleScalar<T, FoldRescaleRange::Pi> : FoldRescalePi<T> {};

template <typename T, std::size_t Dim> struct ComputeBinIndexScalar {
    template <FoldRescaleRange input_range>
    struct Impl : ComputeBinIndex<1, T, Dim, FoldRescaleScalar<T, input_range>> {
        Impl(IntBinInfo<T, Dim> const &info)
            : ComputeBinIndex<1, T, Dim, FoldRescaleScalar<T, input_range>>(
                  info, FoldRescaleScalar<T, input_range>{}) {}
    };
};

template <typename T, std::size_t Dim, std::size_t Unroll> struct WriteTransformedCoordinateScalar {
    typedef std::array<std::array<T, Unroll>, Dim> value_type;

    void operator()(value_type &v, std::size_t d, std::size_t j, T const &t) const { v[d][j] = t; }
};

} // namespace

template <typename T, std::size_t Dim>
SortPointsPlannedFunctor<T, Dim> make_sort_counting_direct_singlethreaded(
    FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info) {
    typedef detail::NuSortImpl<
        T,
        Dim,
        ComputeBinIndexScalar<T, Dim>::template Impl,
        WriteTransformedCoordinateScalar<T, Dim, 1>,
        detail::MovePointsDirect<T, Dim>>
        impl_type;

    return detail::make_sort_functor<T, Dim, detail::SortPointsSingleThreadedImpl, impl_type::template Impl>(
        input_range, info);
}

template <typename T, std::size_t Dim>
SortPointsPlannedFunctor<T, Dim> make_sort_counting_blocked_singlethreaded(
    FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info) {
    typedef detail::NuSortImpl<
        T,
        Dim,
        ComputeBinIndexScalar<T, Dim>::template Impl,
        WriteTransformedCoordinateScalar<T, Dim, 1>,
        detail::MovePointsBlocked<T, Dim, 128>>
        impl_type;

    return detail::make_sort_functor<T, Dim, detail::SortPointsSingleThreadedImpl, impl_type::template Impl>(
        input_range, info);
}

template <typename T, std::size_t Dim>
SortPointsPlannedFunctor<T, Dim>
make_sort_counting_direct_omp(FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info) {
    typedef detail::NuSortImpl<
        T,
        Dim,
        ComputeBinIndexScalar<T, Dim>::template Impl,
        WriteTransformedCoordinateScalar<T, Dim, 1>,
        detail::MovePointsDirect<T, Dim>>
        impl_type;

    return detail::make_sort_functor<T, Dim, detail::SortPointsOmpImpl, impl_type::template Impl>(
        input_range, info);
}

#define INSTANTIATE(T, Dim)                                                                        \
    template SortPointsPlannedFunctor<T, Dim> make_sort_counting_direct_singlethreaded(            \
        FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info);                      \
    template SortPointsPlannedFunctor<T, Dim> make_sort_counting_blocked_singlethreaded(           \
        FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info);                      \
    template SortPointsPlannedFunctor<T, Dim> make_sort_counting_direct_omp(                       \
        FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info);

INSTANTIATE(float, 1);
INSTANTIATE(float, 2);
INSTANTIATE(float, 3);

INSTANTIATE(double, 1);
INSTANTIATE(double, 2);
INSTANTIATE(double, 3);

#undef INSTANTIATE

} // namespace reference
} // namespace spreading
} // namespace finufft
