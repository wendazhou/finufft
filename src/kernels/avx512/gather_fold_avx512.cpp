#include "gather_fold_avx512.h"

#include <cmath>
#include <immintrin.h>

#include "../../kernels/reference/gather_fold_reference.h"
#include "gather_fold_impl.h"

namespace finufft {
namespace spreading {

namespace {

/** Generic vectorized implementation for gathering and folding.
 *
 * This version operates on a set of points corresponding to the vector width at a time.
 *
 */
template <std::size_t Dim, typename FoldRescale>
void gather_and_fold_avx512_rescale(
    nu_point_collection<Dim, float> const &memory,
    nu_point_collection<Dim, float const> const &input, std::array<int64_t, Dim> const &sizes,
    std::int64_t const *sort_indices, FoldRescale &&fold_rescale) {

    std::array<float, Dim> sizes_floating;
    std::copy(sizes.begin(), sizes.end(), sizes_floating.begin());

    std::size_t i = 0;

    for (; i + 15 < memory.num_points; i += 16) {
        // We are using 64-bit indices for 32-bit data.
        // Need to load two registers of indices to cover one register of data.
        // Note: we don't control this input memory yet, so use unaligned here.
        auto addr1 = _mm512_loadu_epi64(sort_indices + i);
        auto addr2 = _mm512_loadu_epi64(sort_indices + i + 8);

        for (std::size_t dim = 0; dim < Dim; ++dim) {
            // Load values from memory, and combine into single 16-wide register.
            __m256 v1 = _mm512_i64gather_ps(addr1, input.coordinates[dim], sizeof(float));
            __m256 v2 = _mm512_i64gather_ps(addr2, input.coordinates[dim], sizeof(float));
            __m512 v = _mm512_insertf32x8(_mm512_castps256_ps512(v1), v2, 1);

            // Perform folding and store result
            fold_rescale(v, sizes_floating[dim]);
            _mm512_store_ps(memory.coordinates[dim] + i, v);
        }

        // Strengths are stored as interleaved complex values.
        // Hence they are "virtually" 64-bit values.
        // We load both the real and imaginary parts together, and store them together
        // by using 64-bit wide operations.
        auto strengths1 = _mm512_i64gather_epi64(addr1, input.strengths, sizeof(double));
        auto strengths2 = _mm512_i64gather_epi64(addr2, input.strengths, sizeof(double));
        _mm512_store_epi64(memory.strengths + 2 * i, strengths1);
        _mm512_store_epi64(memory.strengths + 2 * i + 16, strengths2);
    }

    for (; i < memory.num_points; ++i) {
        auto idx = sort_indices[i];

        for (int j = 0; j < Dim; ++j) {
            memory.coordinates[j][i] = fold_rescale(input.coordinates[j][idx], sizes_floating[j]);
        }

        memory.strengths[2 * i] = input.strengths[2 * idx];
        memory.strengths[2 * i + 1] = input.strengths[2 * idx + 1];
    }
}

/** Generic vectorized implementation for gathering and folding.
 *
 * This version operates on a set of points corresponding to the vector width at a time.
 *
 */
template <std::size_t Dim, typename FoldRescale>
void gather_and_fold_avx512_rescale(
    nu_point_collection<Dim, double> const &memory,
    nu_point_collection<Dim, double const> const &input, std::array<int64_t, Dim> const &sizes,
    std::int64_t const *sort_indices, FoldRescale &&fold_rescale) {

    std::array<double, Dim> sizes_floating;
    std::copy(sizes.begin(), sizes.end(), sizes_floating.begin());

    std::size_t i = 0;

    for (; i + 7 < memory.num_points; i += 8) {
        // We are using 64-bit indices for 32-bit data.
        // Need to load two registers of indices to cover one register of data.
        auto addr = _mm512_loadu_epi64(sort_indices + i);

        for (std::size_t dim = 0; dim < Dim; ++dim) {
            // Load values from memory, and combine into single 16-wide register.
            __m512d v = _mm512_i64gather_pd(addr, input.coordinates[dim], sizeof(double));

            // Perform folding and store result
            fold_rescale(v, sizes_floating[dim]);
            _mm512_store_pd(memory.coordinates[dim] + i, v);
        }

        // Strengths are stored as interleaved complex values.
        // Need to process two double words at a time.
        const int re = 0b0000;
        const int im = 0b1000;

        auto addr_x2_re = _mm512_add_epi64(addr, addr);
        auto addr_x2_im = _mm512_add_epi64(addr_x2_re, _mm512_set1_epi64(1));

        auto addr_strengths_1 = _mm512_permutex2var_epi64(
            addr_x2_re,
            _mm512_setr_epi64(re | 0, im | 0, re | 1, im | 1, re | 2, im | 2, re | 3, im | 3),
            addr_x2_im);
        auto addr_strengths_2 = _mm512_permutex2var_epi64(
            addr_x2_re,
            _mm512_setr_epi64(re | 4, im | 4, re | 5, im | 5, re | 6, im | 6, re | 7, im | 7),
            addr_x2_im);

        auto strengths1 = _mm512_i64gather_pd(addr_strengths_1, input.strengths, sizeof(double));
        auto strengths2 = _mm512_i64gather_pd(addr_strengths_2, input.strengths, sizeof(double));
        _mm512_store_pd(memory.strengths + 2 * i, strengths1);
        _mm512_store_pd(memory.strengths + 2 * i + 8, strengths2);
    }

    for (; i < memory.num_points; ++i) {
        auto idx = sort_indices[i];

        for (int j = 0; j < Dim; ++j) {
            memory.coordinates[j][i] = fold_rescale(input.coordinates[j][idx], sizes_floating[j]);
        }

        memory.strengths[2 * i] = input.strengths[2 * idx];
        memory.strengths[2 * i + 1] = input.strengths[2 * idx + 1];
    }
}

} // namespace

namespace avx512 {

template <typename T, std::size_t Dim>
void gather_fold_avx512_impl(
    nu_point_collection<Dim, T> const &memory, nu_point_collection<Dim, T const> const &input,
    std::array<int64_t, Dim> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) noexcept {
    if (rescale_range == FoldRescaleRange::Identity) {
        gather_and_fold_avx512_rescale(
            memory, input, sizes, sort_indices, avx512::FoldRescaleIdentityAvx512<T>{});
    } else {
        gather_and_fold_avx512_rescale(
            memory, input, sizes, sort_indices, avx512::FoldRescalePiAvx512<T>{});
    }
}

template void gather_fold_avx512_impl<float, 1>(
    nu_point_collection<1, float> const &memory, nu_point_collection<1, float const> const &input,
    std::array<int64_t, 1> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) noexcept;
template void gather_fold_avx512_impl<float, 2>(
    nu_point_collection<2, float> const &memory, nu_point_collection<2, float const> const &input,
    std::array<int64_t, 2> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) noexcept;
template void gather_fold_avx512_impl<float, 3>(
    nu_point_collection<3, float> const &memory, nu_point_collection<3, float const> const &input,
    std::array<int64_t, 3> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) noexcept;

template void gather_fold_avx512_impl<double, 1>(
    nu_point_collection<1, double> const &memory, nu_point_collection<1, double const> const &input,
    std::array<int64_t, 1> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) noexcept;
template void gather_fold_avx512_impl<double, 2>(
    nu_point_collection<2, double> const &memory, nu_point_collection<2, double const> const &input,
    std::array<int64_t, 2> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) noexcept;
template void gather_fold_avx512_impl<double, 3>(
    nu_point_collection<3, double> const &memory, nu_point_collection<3, double const> const &input,
    std::array<int64_t, 3> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) noexcept;

} // namespace avx512


const GatherFoldAvx512 gather_and_fold_avx512;

} // namespace spreading

} // namespace finufft
