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

} // namespace avx512
} // namespace spreading
} // namespace finufft
