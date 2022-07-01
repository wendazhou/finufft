#pragma once

/** @file
 *
 * Routines for evaluating polynomials in a vectorized fashion.
 *
 */

#include <immintrin.h>

namespace finufft {
namespace spreading {
namespace avx512 {

/** Utility structure which implements a fully unrolled Horner polynomial evaluation strategy.
 *
 * @see horner_polynomial_evaluation
 *
 */
template <std::size_t Degree> struct Avx512HornerPolynomialEvaluation {
    __m512 operator()(__m512 z, float const *coeffs) const {
        return _mm512_fmadd_ps(
            Avx512HornerPolynomialEvaluation<Degree - 1>()(z, coeffs),
            z,
            _mm512_load_ps(coeffs + 16 * Degree));
    }

    __m512d operator()(__m512d z, double const *coeffs) const {
        return _mm512_fmadd_pd(
            Avx512HornerPolynomialEvaluation<Degree - 1>()(z, coeffs),
            z,
            _mm512_load_pd(coeffs + 8 * Degree));
    }
};

template <> struct Avx512HornerPolynomialEvaluation<0> {
    __m512 operator()(__m512 z, float const *coeffs) const { return _mm512_load_ps(coeffs); }
    __m512d operator()(__m512d z, double const *coeffs) const { return _mm512_load_pd(coeffs); }
};

/** Utility to evaluate a polynomial through Horner's method.
 *
 * @param z The value at which to evaluate the polynomial.
 * @param coeffs The coefficients of the polynomials, in reverse order.
 *               Must be aligned to 64 bytes.
 * @param degree The degree of the polynomial to evaluate.
 *
 * Note that we make use of an auxiliary structure to implement a fully
 * statically unrolled Horner polynomial evaluation strategy. The naive
 * loop implementation generates a loop on GCC 11.2, which leads to a
 * small performance loss compared to the statically unrolled version.
 *
 */
template <std::size_t Degree>
inline __m512 horner_polynomial_evaluation(__m512 z, float const *coeffs) {
    return Avx512HornerPolynomialEvaluation<Degree>{}(z, coeffs);
}

template <std::size_t Degree>
inline __m512d horner_polynomial_evaluation(__m512d z, double const *coeffs) {
    return Avx512HornerPolynomialEvaluation<Degree>{}(z, coeffs);
}

/** Evaluate a polynomial kernel and multiply by strengths, width 8 dual issue.
 * 
 * This function is used to implement the evaluation of two polynomial kernels
 * of width 8, multiply by complex strengths, interleave and separate the results.
 * 
 * @param z The value at which to evaluate the polynomial. In general, this is expected
 *        to be a vector of length 16 corresponding of two halves [z1 x 8, z2 x 8].
 * @param coeffs A column-major array of size 16 x (Degree + 1) representing the coefficients
 *        of the polynomials at each point, in descending order of degree.
 * @param strengths An array of length 4 containing the complex strengths for the first and
 *        second point.
 * @param[out] v1 The complex interleaved result for the first point
 * @param[out] v2 The complex interleaved result for the second point
 * 
 */
template <std::size_t Degree>
void poly_eval_multiply_strengths_2x8(
    __m512 z, float const *coeffs, float const *strengths, __m512 &v1, __m512 &v2) {
    __m512 k = horner_polynomial_evaluation<Degree>(z, coeffs);

    // Load real and imaginary coefficients, split by 256-bit lane.
    __m512 w_re = _mm512_insertf32x8(_mm512_set1_ps(strengths[0]), _mm256_set1_ps(strengths[2]), 1);
    __m512 w_im = _mm512_insertf32x8(_mm512_set1_ps(strengths[1]), _mm256_set1_ps(strengths[3]), 1);

    // Multiply by coefficients in lane.
    __m512 k_re = _mm512_mul_ps(k, w_re);
    __m512 k_im = _mm512_mul_ps(k, w_im);

    // To finish, we need to write out the results in interleaved format
    // and separate the results for x1 and x2.
    // We achieve this by using two two-vector shuffles.
    // The shuffles below fully interleave the lower (v1) or upper (v2)
    // 256-bit lanes of k_re and k_im.
    const int re = 0b00000;
    const int im = 0b10000;

    // clang-format off
    v1 = _mm512_permutex2var_ps(
        k_re,
        _mm512_setr_epi32(
            re | 0, im | 0, re | 1, im | 1, re | 2, im | 2, re | 3, im | 3,
            re | 4, im | 4, re | 5, im | 5, re | 6, im | 6, re | 7, im | 7),
        k_im);
    v2 = _mm512_permutex2var_ps(
        k_re,
        _mm512_setr_epi32(
            re | 8,  im | 8,  re | 9,  im | 9,  re | 10, im | 10, re | 11, im | 11,
            re | 12, im | 12, re | 13, im | 13, re | 14, im | 14, re | 15, im | 15),
        k_im);
    // clang-format on
}

} // namespace avx512
} // namespace spreading
} // namespace finufft
