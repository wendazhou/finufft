#pragma once

#include <array>
#include <cstdint>

#include <immintrin.h>

/** @file
 *
 * This file contains AVX512 subroutines which align and split an unaligned
 * vector (or set of vectors) according to the offset.
 *
 * Conceptually, all methods in the current file do the following:
 * Given a vector [x1, x2, x3, x4], and an offset which is not a multiple
 * of the vector length (e.g. 2), it produces two vectors:
 *   [0, 0, x1, x2], [x3, x4, 0, 0]
 * This is implemented generally for a number of vector widths,
 * and for logical vectors spanning multiple hardware registers.
 *
 */

namespace finufft {
namespace spreading {
namespace avx512 {

/** Lookup table for 2-vector rotate shuffles.
 *
 * The ith row of the table contains a shuffle vector, such that the result of
 * the instruction
 *
 * _mm512_permutex2var_pd(x, table + i * 8, y)
 *
 * is a vector such that the lower i elements of the vector correspond
 * to the upper i elements of the input vector x, and the upper (8 - i)
 * elements of the vector correspond to the lower (8 - i) elements of the
 * of the input vector y.
 *
 * Pictorially, we have that (illustrating using width 4, the actual table
 * being for width 8):
 *  i = 0: [x1, x2, x3, x4], [y1, y2, y3, y4] -> [y1, y2, y3, y4]
 *  i = 1: [x1, x2, x3, x4], [y1, y2, y3, y4] -> [x4, y1, y2, y3]
 *  i = 2: [x1, x2, x3, x4], [y1, y2, y3, y4] -> [x3, x4, y1, y2]
 *  i = 3: [x1, x2, x3, x4], [y1, y2, y3, y4] -> [x2, x3, x4, y1]
 *
 */
alignas(64) static constexpr std::array<int64_t, 8 * 8> rotate_2vec_shuffle_lut = ([] {
    std::array<int64_t, 8 * 8> result = {};

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < i; ++j) {
            result[i * 8 + j] = (8 - i) + j;
        }

        for (int j = i; j < 8; ++j) {
            result[i * 8 + j] = 0b1000 + (j - i);
        }
    }

    return result;
})();

/** Split unaligned offset 8-wide double vector into two aligned 8-wide double vectors.
 *
 * @param v The vector to split
 * @param offset The amount to offset the vector by, must be in the range [0, 8)
 * @param v1 The low half of the aligned vector
 * @param v2 The high half of the aligned vector
 *
 * This function uses a strategy based on a look-up table and two multi-vector
 * permutes in order to assemble v1 and v2. The look-up tables are constructed
 * so as to contain permutations which shift the vector elements by the correct
 * amount, and then fetches from the second input vector of the two-element permute
 * (typically set to a zero vector).
 *
 * Attempts at using a zero-masked permute lead to slightly lower performance,
 * with the two-vector shuffle being approximately 10% faster on Intel Skylake-X
 * for the implentation of spread_subproblem in 1d.
 *
 */
inline void split_unaligned_vector(const __m512d &v, const int offset, __m512d &o1, __m512d &o2) {
    __m512i shuf = _mm512_load_epi64(rotate_2vec_shuffle_lut.data() + offset * 8);

    o1 = _mm512_permutex2var_pd(_mm512_setzero_pd(), shuf, v);
    o2 = _mm512_permutex2var_pd(v, shuf, _mm512_setzero_pd());
}

/** Splits unaligned offset 4-wide double vector into two aligned 4-wide double vectors.
 *
 * @param offset The amount to offset the vector by, must be in the range [0, 4)
 *
 * This function uses a strategy based on a look-up table and a single 512-bit wide
 * permutation. This requires the use of AVX-512 instructions.
 *
 */
inline void split_unaligned_vector(const __m256d &v, int offset, __m256d &o1, __m256d &o2) {
    // Ok to use 2-vector shuffle LUT here.
    // For the 1-vector use case, the provenance of information is simply ignored,
    // and the shuffle vector corresponds to a circular shift of the vector elements.
    __m512d v_offset = _mm512_permutexvar_pd(
        _mm512_load_epi64(rotate_2vec_shuffle_lut.data() + offset * 8),
        _mm512_castpd256_pd512(v));

    o1 = _mm512_castpd512_pd256(v_offset);
    o2 = _mm512_extractf64x4_pd(v_offset, 1);
}

/** Splits unaligned pair of 8-wide double vector into three aligned 8-wide double vectors.
 *
 * This function splits a pair of 8-wide vector viewed as a contiguous 16-wide vector
 * after shifting by the given offset into three aligned 8-wide vectors.
 *
 * @param offset The amount to offset the vector by, must be in the range [0, 8)
 *
 */
inline void split_unaligned_vector(
    const __m512d &v1, const __m512d &v2, int offset, __m512d &o1, __m512d &o2, __m512d &o3) {
    auto shuf = _mm512_load_epi64(rotate_2vec_shuffle_lut.data() + offset * 8);

    o1 = _mm512_permutex2var_pd(_mm512_setzero_pd(), shuf, v1);
    o2 = _mm512_permutex2var_pd(v1, shuf, v2);
    o3 = _mm512_permutex2var_pd(v2, shuf, _mm512_setzero_pd());
}

} // namespace avx512
} // namespace spreading
} // namespace finufft
