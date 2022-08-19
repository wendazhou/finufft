#include "sort_bin_counting.h"
#include "../reference/sort_bin_counting_impl.h"
#include "gather_fold_impl.h"

#include <immintrin.h>
#include <libdivide.h>

namespace finufft {
namespace spreading {
namespace avx512 {

namespace {

template <typename T, FoldRescaleRange input_range> struct FoldRescaleAvx512;
template <typename T>
struct FoldRescaleAvx512<T, FoldRescaleRange::Identity> : FoldRescaleIdentityAvx512<T> {};
template <typename T> struct FoldRescaleAvx512<T, FoldRescaleRange::Pi> : FoldRescalePiAvx512<T> {};

template <typename T, std::size_t Dim, typename FoldRescale> struct ComputeBinIndex;

template <std::size_t Dim, typename FoldRescale> struct ComputeBinIndex<float, Dim, FoldRescale> {
    IntBinInfo<float, Dim> info;
    FoldRescale fold_rescale;
    std::array<float, Dim> size_f;
    std::array<libdivide::divider<uint32_t>, Dim> dividers;

    typedef std::uint32_t index_type;
    static constexpr std::size_t unroll = 16;

    explicit ComputeBinIndex(IntBinInfo<float, Dim> const &info, FoldRescale const &fold_rescale)
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
        nu_point_collection<Dim, const float> const &input, std::size_t i, std::size_t limit,
        tcb::span<std::uint32_t, 16> bins_out, std::integral_constant<bool, Partial>,
        WriteTransformedCoordinate &&write_transformed_coordinate) const {

        __m512i bin = _mm512_setzero_si512();

        for (std::size_t j = 0; j < Dim; ++j) {
            __m512 x;

            if (Partial) {
                __mmask16 load_mask = (1 << limit) - 1;
                x = _mm512_maskz_load_ps(load_mask, input.coordinates[j] + i);
            } else {
                x = _mm512_loadu_ps(input.coordinates[j] + i);
            }

            fold_rescale(x, size_f[j]);

            __m512 x_c = _mm512_ceil_ps(_mm512_sub_ps(x, _mm512_set1_ps(info.offset[j])));
            __m512i x_b =
                _mm512_sub_epi32(_mm512_cvtps_epi32(x_c), _mm512_set1_epi32(info.global_offset[j]));
            x_b /= dividers[j];

            bin = _mm512_add_epi32(
                bin, _mm512_mullo_epi32(x_b, _mm512_set1_epi32(info.bin_index_stride[j])));

            write_transformed_coordinate(j, 0, x);
        }

        _mm512_storeu_si512(bins_out.data(), bin);
    }
};

template <typename T, std::size_t Dim> struct ComputeBinIndexBound {
    template <FoldRescaleRange input_range>
    struct Impl : ComputeBinIndex<T, Dim, FoldRescaleAvx512<T, input_range>> {
        explicit Impl(IntBinInfo<T, Dim> const &info)
            : ComputeBinIndex<T, Dim, FoldRescaleAvx512<T, input_range>>(
                  info, FoldRescaleAvx512<T, input_range>{}) {}
    };
};

template <typename T, std::size_t Dim> struct WriteTransformedCoordinate;

template <std::size_t Dim> struct WriteTransformedCoordinate<float, Dim> {
    struct AlignedArray {
        alignas(64) float data[Dim * 16];
        float *operator[](std::size_t i) { return data + i * 16; }
        const float *operator[](std::size_t i) const { return data + i * 16; }
    };

    typedef AlignedArray value_type;

    void operator()(AlignedArray &arr, std::size_t j, std::size_t, __m512 x) const {
        _mm512_store_ps(arr[j], x);
    }
};

} // namespace

template <typename T, std::size_t Dim>
SortPointsPlannedFunctor<T, Dim> make_sort_counting_direct_singlethreaded(
    FoldRescaleRange input_range, IntBinInfo<T, Dim> const &info) {
    typedef reference::detail::NuSortImpl<
        T,
        Dim,
        ComputeBinIndexBound<T, Dim>::template Impl,
        WriteTransformedCoordinate<T, Dim>,
        reference::detail::MovePointsDirect<T, Dim>>
        impl_type;

    return reference::detail::make_sort_functor<
        T,
        Dim,
        reference::detail::SortPointsSingleThreadedImpl,
        impl_type::template Impl>(input_range, info);
}

template <typename T, std::size_t Dim>
SortPointsPlannedFunctor<T, Dim> make_sort_counting_blocked_singlethreaded(
    FoldRescaleRange input_range, IntBinInfo<T, Dim> const &info) {
    typedef reference::detail::NuSortImpl<
        T,
        Dim,
        ComputeBinIndexBound<T, Dim>::template Impl,
        WriteTransformedCoordinate<T, Dim>,
        reference::detail::MovePointsBlocked<T, Dim, 128>>
        impl_type;

    return reference::detail::make_sort_functor<
        T,
        Dim,
        reference::detail::SortPointsSingleThreadedImpl,
        impl_type::template Impl>(input_range, info);
}


#define INSTANTIATE(T, Dim)                                                                        \
    template SortPointsPlannedFunctor<T, Dim> make_sort_counting_direct_singlethreaded(            \
        FoldRescaleRange input_range, IntBinInfo<T, Dim> const &info);                             \
    template SortPointsPlannedFunctor<T, Dim> make_sort_counting_blocked_singlethreaded(           \
        FoldRescaleRange input_range, IntBinInfo<T, Dim> const &info);

INSTANTIATE(float, 1);
INSTANTIATE(float, 2);
INSTANTIATE(float, 3);

#undef INSTANTIATE

} // namespace avx512
} // namespace spreading
} // namespace finufft
