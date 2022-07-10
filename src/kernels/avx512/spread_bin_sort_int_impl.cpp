#include "spread_bin_sort_int.h"

#include <type_traits>

#include "gather_fold_impl.h"

#include <immintrin.h>
#include <libdivide.h>

namespace finufft {
namespace spreading {
namespace avx512 {

template <typename T, std::size_t Dim, typename FoldRescale> struct ComputeBinAndPackSingle;

/** Process inner loop of bin index and packing.
 *
 * This implementation processes the inner loop of the bin index computation.
 * It assumes that the integer coordinate of the point, as well as the bin index,
 * will fit in 32 bit integers.
 *
 * Note that although the implementation of the fold / rescale and bin computation
 * is batched using vector instructions, the packing of the data is not due its
 * irregular nature.
 *
 */
template <std::size_t Dim, typename FoldRescale>
struct ComputeBinAndPackSingle<float, Dim, FoldRescale> {
    alignas(64) float coordinates[Dim * 16];
    alignas(64) int32_t bins[16];

    std::array<float, Dim> size_f;
    std::array<libdivide::divider<uint32_t>, Dim> dividers;
    IntBinInfo<float, Dim> const &info;
    FoldRescale fold_rescale;

    explicit ComputeBinAndPackSingle(IntBinInfo<float, Dim> const &info, FoldRescale fold_rescale)
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

    template <bool Partial>
    void operator()(
        std::size_t i, std::size_t limit, nu_point_collection<Dim, const float> const &input,
        PointBin<float, Dim> *__restrict output, std::integral_constant<bool, Partial>) {

        __m512i bin = _mm512_setzero_si512();

        for (std::size_t j = 0; j < Dim; ++j) {
            __m512 x;

            if (Partial) {
                __mmask16 load_mask = _bzhi_u32(-1, limit);
                x = _mm512_maskz_load_ps(load_mask, input.coordinates[j] + i);
            } else {
                x = _mm512_load_ps(input.coordinates[j] + i);
            }

            fold_rescale(x, size_f[j]);

            // Compute bin indices
            __m512 x_c = _mm512_ceil_ps(_mm512_sub_ps(x, _mm512_set1_ps(info.offset[j])));
            __m512i x_b =
                _mm512_sub_epi32(_mm512_cvtps_epi32(x_c), _mm512_set1_epi32(info.global_offset[j]));
            x_b /= dividers[j];

            bin = _mm512_add_epi32(
                bin, _mm512_mullo_epi32(x_b, _mm512_set1_epi32(info.bin_index_stride[j])));

            _mm512_store_ps(coordinates + j * 16, x);
        }

        _mm512_store_epi32(bins, bin);

        // Write out individual points
        for (std::size_t j = 0; j < Partial ? limit : 16; ++j) {
            auto &p = output[i * 16 + j];

            p.bin = bins[j];
            for (std::size_t dim = 0; dim < Dim; ++dim) {
                p.coordinates[dim] = coordinates[dim * 16 + j];
            }
            p.strength[0] = input.strengths[2 * (i + j)];
            p.strength[1] = input.strengths[2 * (i + j) + 1];
        }
    }
};

/** Fast-path algorithm for computing bins and packing.
 *
 */
template <typename T, std::size_t Dim, typename FoldRescale>
void compute_bins_and_pack_impl(
    nu_point_collection<Dim, const T> const &input, IntBinInfo<T, Dim> const &info,
    PointBin<T, Dim> *output, FoldRescale &&fold_rescale) {

    ComputeBinAndPackSingle<T, Dim, FoldRescale> loop(
        info, std::forward<FoldRescale>(fold_rescale));

    std::size_t i = 0;
    for (; i + 16 < input.num_points; i += 16) {
        loop(i, 16, input, output, std::false_type{});
    }
    loop(i, input.num_points - i, input, output, std::true_type{});
}

template <typename T, std::size_t Dim>
void compute_bins_and_pack(
    nu_point_collection<Dim, const T> const &input, FoldRescaleRange range,
    IntBinInfo<T, Dim> const &info, PointBin<T, Dim> *output) {

    if (range == FoldRescaleRange::Pi) {
        compute_bins_and_pack_impl(input, info, output, FoldRescalePiAvx512<T>{});
    } else {
        compute_bins_and_pack_impl(input, info, output, FoldRescaleIdentityAvx512<T>{});
    }
}

#define INSTANTIATE(T, Dim)                                                                        \
    template void compute_bins_and_pack<T, Dim>(                                                   \
        nu_point_collection<Dim, const T> const &input,                                            \
        FoldRescaleRange range,                                                                    \
        IntBinInfo<T, Dim> const &info,                                                            \
        PointBin<T, Dim> *output);

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)
INSTANTIATE(float, 3)

#undef INSTANTIATE

} // namespace avx512
} // namespace spreading
} // namespace finufft
