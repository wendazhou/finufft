#include "spread_bin_sort_int.h"

#include <cstring>
#include <type_traits>

#include "gather_fold_impl.h"

#include <immintrin.h>
#include <libdivide.h>

namespace finufft {
namespace spreading {
namespace avx512 {

template <typename T, std::size_t Dim> struct WriteSoAFromAoS;

/** Generic implementation of packed write-out.
 *
 * After bin computation, we need to transpose from an Structure-of-Arrays format to an
 * Array-of-Structure format. The generic implementation writes out to an intermediate
 * buffer to transpose. However, this is quite inefficient, and better performance can
 * be achieved by transposing in registers.
 *
 */
template <std::size_t Dim> struct WriteSoAFromAoS<float, Dim> {
    alignas(64) float coordinates[Dim * 16];
    alignas(64) float strengths[32];
    alignas(64) int32_t bins[16];
    alignas(64) PointBin<float, Dim> packed[16];

    template <bool Partial>
    void operator()(
        PointBin<float, Dim> *output, __m512i const &b, __m512 *coords, __m512 const &s1,
        __m512 const &s2, std::size_t limit, std::integral_constant<bool, Partial>) noexcept {
        _mm512_store_epi32(bins, b);
        for (std::size_t i = 0; i < Dim; i++) {
            _mm512_store_ps(coordinates + i * 16, coords[i]);
        }
        _mm512_store_ps(strengths, s1);
        _mm512_store_ps(strengths + 16, s2);

        // Write out individual points
        for (std::size_t j = 0; j < 16; ++j) {
            auto &p = packed[j];

            p.bin = bins[j];
            for (std::size_t dim = 0; dim < Dim; ++dim) {
                p.coordinates[dim] = coordinates[dim * 16 + j];
            }
            p.strength[0] = strengths[2 * j];
            p.strength[1] = strengths[2 * j + 1];
        }

        if (Partial) {
            std::memcpy(output, packed, sizeof(PointBin<float, Dim>) * limit);
        } else {
            auto output_si512 = reinterpret_cast<__m512i *>(output);
            auto packed_si512 = reinterpret_cast<__m512i *>(packed);

            for (std::size_t j = 0; j < sizeof(packed) / 64; ++j) {
                _mm512_stream_si512(output_si512 + j, _mm512_load_si512(packed_si512 + j));
            }
        }
    }
};

/** Specialized transpose for 2d float case.
 *
 */
template <> struct WriteSoAFromAoS<float, 2> {
    static_assert(
        sizeof(PointBin<float, 2>) == (sizeof(uint32_t) + sizeof(float) * 4),
        "PointBin<float, 2> has additional padding.");

    template <bool Partial>
    void operator()(
        PointBin<float, 2> *output, __m512i const &b, __m512 *coords, __m512 const &s1,
        __m512 const &s2, int64_t limit, std::integral_constant<bool, Partial>) noexcept {

        __m512 x = coords[0];
        __m512 y = coords[1];

        // Step 1: 16x4 transpose of x, y, s1, s2
        // We start with 16-long vectors [x0, x1, ..], [y0, y1, ..], [u0, v0, u1 ..], [u8, v8, ...]
        // and transpose into 4x4 vectors [x0 y0 u0 v0 x1 ..], [x3 y3 u3 v3 ..], ...
        //
        // This is done through a combination of shuffles and unpack.
        __m512i p = _mm512_setr_epi64(0, 4, 1, 5, 2, 6, 3, 7);
        const int from_x = 0;
        const int from_y = 0b10000;

        // First shuffle: interleave x and y, and shuffle 128-bit lanes
        // for prepartion to interleave with s1 and s2.

        // clang-format off
        __m512 xy_lo_p = _mm512_permutex2var_ps(
            x,
            _mm512_setr_epi32(
                0 | from_x, 0 | from_y, 4 | from_x, 4 | from_y,
                1 | from_x, 1 | from_y, 5 | from_x, 5 | from_y,
                2 | from_x, 2 | from_y, 6 | from_x, 6 | from_y,
                3 | from_x, 3 | from_y, 7 | from_x, 7 | from_y),
            y);
        __m512 xy_hi_p = _mm512_permutex2var_ps(
            x,
            _mm512_setr_epi32(
                8 | from_x, 8 | from_y, 12 | from_x, 12 | from_y,
                9 | from_x, 9 | from_y, 13 | from_x, 13 | from_y,
                10 | from_x, 10 | from_y, 14 | from_x, 14 | from_y,
                11 | from_x, 11 | from_y, 15 | from_x, 15 | from_y),
            y);
        // clang-format on

        // Shuffle s1, s2 to prepare for interleaving with xy_lo_p and xy_hi_p.
        __m512 s1_p = _mm512_castpd_ps(_mm512_permutexvar_pd(p, _mm512_castps_pd(s1)));
        __m512 s2_p = _mm512_castpd_ps(_mm512_permutexvar_pd(p, _mm512_castps_pd(s2)));

        // Interleave xy_lo_p and xy_hi_p with s1_p and s2_p.
        // This produces the x, y, s1, s2 interleaved in the correct position.
        __m512 xyuv_1 =
            _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(xy_lo_p), _mm512_castps_pd(s1_p)));
        __m512 xyuv_2 =
            _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(xy_lo_p), _mm512_castps_pd(s1_p)));
        __m512 xyuv_3 =
            _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(xy_hi_p), _mm512_castps_pd(s2_p)));
        __m512 xyuv_4 =
            _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(xy_hi_p), _mm512_castps_pd(s2_p)));

        // Permute bin indices to final position in preparation for final merging
        __m512i b_final_i = _mm512_permutexvar_epi32(
            _mm512_setr_epi32(0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3), b);
        __m512 b_final = _mm512_castsi512_ps(b_final_i);

        // clang-format off

        // The first row is given as:
        // [b0, x0, y0, u0, v0, b1, x1, y1, u1, v1, b2, x2, y2, u2, v2, b3]
        // and can be assembled entirely from xyuv_1 and the bin indices.
        __m512 r1 = _mm512_permutex2var_ps(
            xyuv_1,
            _mm512_setr_epi32(
                0 | from_y,
                0, 1, 2, 3,
                5 | from_y,
                4, 5, 6, 7,
                10 | from_y,
                8, 9, 10, 11,
                15 | from_y),
            b_final);

        // The second row is given as:
        // [x3, y3, u3, v3, b4, x4, y4, u4, v4, b5, x5, y5, u5, v5, b6, x6]
        // we first assemble the x,y,u,v values into a single vector,
        // then blend with the bin indices.
        __m512 r2 = _mm512_permutex2var_ps(
            xyuv_2,
            _mm512_setr_epi32(
                12 | from_y, 13 | from_y, 14 | from_y, 15 | from_y,
                0, // bin index blended away later
                0, 1, 2, 3,
                0, // bin index blended away later
                4, 5, 6, 7,
                0, // bin index blended away later
                8),
            xyuv_1);
        r2 = _mm512_mask_blend_ps(
            (1 << 4) | (1 << 9) | (1 << 14),
            r2,
            b_final);

        // The third row is given as
        // [y6, u6, v6, b7, x7, y7, u7, v7, b8, x8, y8, u8, v8, b9, x9, y9]
        // We first assemble the x,y,u,v values into a single vector,
        // then blend with the bin indices.
        __m512 r3 = _mm512_permutex2var_ps(
            xyuv_2,
            _mm512_setr_epi32(
                9, 10, 11,
                0, // bin index blended away later
                12, 13, 14, 15,
                0, // bin index blended away later
                0 | from_y, 1 | from_y, 2 | from_y, 3 | from_y,
                0, // bin index blended away later
                4 | from_y, 5 | from_y),
            xyuv_3);
        r3 = _mm512_mask_blend_ps(
            (1 << 3) | (1 << 8) | (1 << 13),
            r3,
            b_final);

        // The fourth row is given as
        // [u9, v9, b10, x10, y10, u10, v10, b11, x11, y11, u11, v11, b12, x12, y12, u12]
        __m512 r4 = _mm512_permutex2var_ps(
            xyuv_3,
            _mm512_setr_epi32(
                6, 7,
                0, // bin index blended away later
                8, 9, 10, 11,
                0, // bin index blended away later
                12, 13, 14, 15,
                0, // bin index blended away later
                0 | from_y, 1 | from_y, 2 | from_y),
            xyuv_4);
        r4 = _mm512_mask_blend_ps(
            (1 << 2) | (1 << 7) | (1 << 12),
            r4,
            b_final);

        // The fifth row is given as:
        // [v12, b13, x13, y13, u13, v13, b14, x14, y14, u14, v14, b15, x15, y15, u15, v15]
        // Note that like the first row, this row only depends on xyuv_4, and hence we
        // directly permute with the bin indices.
        __m512 r5 = _mm512_permutex2var_ps(
            xyuv_4,
            _mm512_setr_epi32(
                3,
                1 | from_y,
                4, 5, 6, 7,
                6 | from_y,
                8, 9, 10, 11,
                11 | from_y,
                12, 13, 14, 15),
            b_final);

        // clang-format on

        float *out = reinterpret_cast<float *>(output);
        if (!Partial) {
            _mm512_stream_ps(out, r1);
            _mm512_stream_ps(out + 16, r2);
            _mm512_stream_ps(out + 32, r3);
            _mm512_stream_ps(out + 48, r4);
            _mm512_stream_ps(out + 64, r5);
        } else {
            if (limit < 4) {
                auto num_elements = 5 * limit;
                _mm512_mask_store_ps(out, (1 << (num_elements)) - 1, r1);
                return;
            }

            _mm512_stream_ps(out, r1);
            limit -= 4;
            out += 16;

            // Limit now excludes the first partial element in the row
            if (limit <= 2) {
                auto num_elements = 4 + limit * 5;
                _mm512_mask_store_ps(out, (1 << (num_elements)) - 1, r2);
                return;
            }

            _mm512_stream_ps(out, r2);
            limit -= 3;
            out += 16;

            if (limit <= 2) {
                auto num_elements = 3 + limit * 5;
                _mm512_mask_store_ps(out, (1 << (num_elements)) - 1, r3);
                return;
            }

            _mm512_stream_ps(out, r3);
            limit -= 3;
            out += 16;

            if (limit <= 2) {
                auto num_elements = 2 + limit * 5;
                _mm512_mask_store_ps(out, (1 << (num_elements)) - 1, r4);
                return;
            }

            _mm512_stream_ps(out, r4);
            limit -= 3;
            out += 16;

            auto num_elements = 1 + limit * 5;
            _mm512_mask_store_ps(out, (1 << (num_elements)) - 1, r5);
        }
    }
};

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
    WriteSoAFromAoS<float, Dim> write_points;

    static_assert(
        sizeof(PointBin<float, Dim>) % 4 == 0, "PointBin must have a size that is a multiple of 4");

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
        PointBin<float, Dim> *__restrict output, std::integral_constant<bool, Partial> partial) {

        __m512i bin = _mm512_setzero_si512();
        __m512 coordinates[Dim];

        for (std::size_t j = 0; j < Dim; ++j) {
            __m512 x;

            if (Partial) {
                __mmask16 load_mask = (1 << limit) - 1;
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

            coordinates[j] = x;
        }

        __m512 s1 = _mm512_load_ps(input.strengths + 2 * i);
        __m512 s2 = _mm512_load_ps(input.strengths + 2 * i + 16);

        write_points(output, bin, coordinates, s1, s2, limit, partial);
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

    // Main loop, vectorized by 16x
#pragma omp parallel firstprivate(loop)
    {
#pragma omp for
        for (std::size_t i = 0; i < input.num_points / 16; ++i) {
            loop(i * 16, 16, input, output + i * 16, std::false_type{});
        }

        // Use a fence to force serialization of non-temporal stores which we may use in the loop.
        _mm_sfence();
    }

    // Masked tail elements
    std::size_t next_index = (input.num_points / 16) * 16;
    if (next_index < input.num_points) {
        loop(
            next_index,
            input.num_points - next_index,
            input,
            output + next_index,
            std::true_type{});
    }
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

template <typename T, std::size_t Dim>
SortPointsFunctor<T, Dim> get_sort_functor(SortPackedTimers const& timers) {
    return reference::make_ips4o_sort_functor<T, Dim>(
        &compute_bins_and_pack<T, Dim>, &reference::unpack_sorted_bins_to_points<T, Dim>, timers);
}

#define INSTANTIATE(T, Dim)                                                                        \
    template void compute_bins_and_pack<T, Dim>(                                                   \
        nu_point_collection<Dim, const T> const &input,                                            \
        FoldRescaleRange range,                                                                    \
        IntBinInfo<T, Dim> const &info,                                                            \
        PointBin<T, Dim> *output);                                                                 \
    template SortPointsFunctor<T, Dim> get_sort_functor(SortPackedTimers const &timers);

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)
INSTANTIATE(float, 3)

#undef INSTANTIATE

} // namespace avx512
} // namespace spreading
} // namespace finufft
