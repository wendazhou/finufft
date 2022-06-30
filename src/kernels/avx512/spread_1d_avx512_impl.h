#pragma once

/** @file
 *
 * Implementation for 1-dimensional spreading subproblem with avx-512 intrinsics.
 * The public interface for obtaining functors leveraging these implementations
 * may be found in `spread_axv512.h`.
 *
 */

#include "align_split_routines.h"
#include "poly_eval_routines.h"

#include <cstddef>
#include <immintrin.h>

#include "../../spreading.h"
#include "../reference/spreading_reference.h"

namespace finufft {
namespace spreading {
namespace avx512 {

/** Spreading functor for width-8 polynomial.
 *
 * This functor implements a AVX-512 vectorized strategy for spreading based on
 * a width-8 polynomial approximation to the kernel function.
 *
 * The current implementation packs the evaluation of two 8-wide polynomials
 * into a single 16-wide vector. Additionally, the computation of the base
 * index for accumulation is vectorized, so that in total 8 elements are
 * processed at a time in the inner loop, in 4 groups of 2.
 *
 * The functor class separates the functionality into three main functions,
 * in addition to the main loop implemented in `operator()`.
 *
 * - compute_kernel: computes the kernel value for a set of points
 *       through a polynomial approximation, merges the complex strengths
 *       and produces interleaved values to be written out
 * - accumulate_strengths: processes the computed pre-multiplied complex
 *       kernels, and accumulates them onto the main output array.
 * - process_8: main unrolled loop which computes accumulation indices from
 *       input coordinates, then calls `compute_kernel` and `accumulate_strengths`
 *       to process the points.
 *
 */
template <std::size_t Degree> struct SpreadSubproblemPolyW8 {
    finufft::spreading::aligned_unique_array<float> coefficients;
    float kernel_width;

    template <typename U>
    SpreadSubproblemPolyW8(U const *coefficients, std::size_t width)
        : coefficients(allocate_aligned_array<float>((Degree + 1) * 16, 64)),
          kernel_width(static_cast<float>(width)) {

        // Duplicate the polynomial coefficients as we will be evaluating the same
        // polynomial twice, one in each 256-bit lane.
        fill_polynomial_coefficients(Degree, coefficients, width, this->coefficients.get(), 8, 16);
        fill_polynomial_coefficients(
            Degree, coefficients, width, this->coefficients.get() + 8, 8, 16);
    }

    /** Computes the kernel for two points, multiplies and interleaves to produce complex
     * interleaved output.
     *
     * This function computes the polynomial kernel for two points with width 8, for 16 values
     * total. It then evaluates then multiplies by the real and imaginary strengths, and interleaves
     * the computed parts to produce a result in complex interleaved format.
     *
     * @param z The points to evaluate the kernel for. The vector is expected to be of the form:
     *   [z_1, z_1, ..., z_2, z_2, ...], where z_1 and z_2 denote the two values for which
     *   the kernel is to be evaluated.
     * @param strengths Pointer to the interleaved strengths. The first two elements correspond to
     * the real and imaginary strengths of the the point at z_1, and the second two elements
     * correspond to the those of the point at z_2.
     * @param v1 The first output, corresponds to the complex interleaved kernel for z_1.
     *   [r^1_1, i^1_1, r^2_2, i^2_2, ...]
     * @param v2 The second output, corresponds to the complex interleaved kernel for z_2.
     *
     */
    void compute_kernel(__m512 z, float const *strengths, __m512 &v1, __m512 &v2) const {
        __m512 k = horner_polynomial_evaluation<Degree>(z, coefficients.get());

        // Load real and imaginary coefficients, split by 256-bit lane.
        __m512 w_re =
            _mm512_insertf32x8(_mm512_set1_ps(strengths[0]), _mm256_set1_ps(strengths[2]), 1);
        __m512 w_im =
            _mm512_insertf32x8(_mm512_set1_ps(strengths[1]), _mm256_set1_ps(strengths[3]), 1);

        // Multiply by coefficients in lane.
        __m512 k_re = _mm512_mul_ps(k, w_re);
        __m512 k_im = _mm512_mul_ps(k, w_im);

        // To finish, we need to write out the results in interleaved format
        // and separate the results for x1 and x2.
        // We achieve this by using two two-vector shuffles.
        // The shuffles below fully interleave the lower (v1) or upper (v2)
        // 256-bit lanes of k_re and k_im.
        const int from_b = (1 << 4);
        const int from_a = (0 << 4);

        // clang-format off
        v1 = _mm512_permutex2var_ps(
            k_re,
            _mm512_setr_epi32(
                from_a | 0, from_b | 0, from_a | 1, from_b | 1, from_a | 2, from_b | 2, from_a | 3, from_b | 3,
                from_a | 4, from_b | 4, from_a | 5, from_b | 5, from_a | 6, from_b | 6, from_a | 7, from_b | 7),
            k_im);
        v2 = _mm512_permutex2var_ps(
            k_re,
            _mm512_setr_epi32(
                from_a | 8, from_b | 8, from_a | 9, from_b | 9, from_a | 10, from_b | 10, from_a | 11, from_b | 11,
                from_a | 12, from_b | 12, from_a | 13, from_b | 13, from_a | 14, from_b | 14, from_a | 15, from_b | 15),
            k_im);
        // clang-format on
    }

    /** Accumulates and stores the given vector into du at the given index.
     * This function splits the stores to ensure that they are aligned,
     * and ensuring that store-to-load forwarding may be successful (as all stores
     * operate on aligned addresses, they either coincide exactly or do not alias).
     *
     * Operationally, given the vector of elements v, we conceptually consider its
     * representation offset by 2 * i elements. This representation is then split
     * into the lower 16 elements and the upper 16 elements using a shuffle.
     * To avoid branches, the shuffle is produced by a lookup table.
     *
     * We illustrate the operation below for vector width of 4 (complex), accumulating into a zero
     * vector.
     *
     * Given the following:
     *   v = [r0, i0, r1, i1, r2, i2, r3, i3]
     *   idx = 2
     * we wish to store v into result starting at index 2 * idx, hence w would be
     *   w = [0, 0, 0, 0, r0, i0, r1, i1, r2, i2, r3, i3, 0, 0, 0, 0]
     *
     * To do so, we compute:
     *   v_lo = [0, 0, 0, 0, r0, i0, r1, i1]
     *   v_hi = [r2, i2, r3, i3, 0, 0, 0, 0]
     * and store each one separately at w, and w + 8.
     * This ensures that our load / stores are aligned, and
     * may benefit from store-to-load forwarding.
     *
     * @param du The vector to store the data into. Must be aligned to 64 bytes.
     * @param i The index at which to store the data at. Note that this is interpreted
     *    as an index into the vector of complex interleaved elements.
     * @param v The vector of elements to store. Interpreted as 8 complex elements
     *    in interleaved format.
     *
     */
    void accumulate_strengths(float *output, int i, __m512 v) const {
        // Compute index as base index to aligned location
        // and offset from aligned location to actual index.
        int i_aligned = i & ~7;
        int i_remainder = i - i_aligned;

        float *out = output + 2 * i_aligned;

        // Split using double operation in order to shuffle pairs
        // of fp32 values (representing a complex number).
        __m512d v_lo, v_hi;
        split_unaligned_vector(_mm512_castps_pd(v), i_remainder, v_lo, v_hi);

        __m512 out_lo = _mm512_load_ps(out);
        __m512 out_hi = _mm512_load_ps(out + 16);

        out_lo = _mm512_add_ps(out_lo, _mm512_castpd_ps(v_lo));
        out_hi = _mm512_add_ps(out_hi, _mm512_castpd_ps(v_hi));

        _mm512_store_ps(out, out_lo);
        _mm512_store_ps(out + 16, out_hi);
    }

    /** Function for a single unrolled iteration of the subproblem.
     *
     * This is the core loop of the kernel, and operates on 8 points at a time
     * in order to leverage vectorization in the computation of the index into the grid.
     *
     */
    void process_8(
        float *__restrict output, float const *coord_x, float const *strengths, int64_t offset,
        std::size_t i) const {
        // Load position of 8 non-uniform points, compute grid and subgrid offsets (vectorized)
        __m256 x = _mm256_load_ps(coord_x + i);
        __m256 x_ceil = _mm256_ceil_ps(_mm256_sub_ps(x, _mm256_set1_ps(0.5f * kernel_width)));
        __m256i x_ceili = _mm256_cvtps_epi32(x_ceil);
        __m256 xi = _mm256_sub_ps(x_ceil, x);

        // Normalized subgrid position for each point
        // [z_0, z_1, ...]
        __m256 z = _mm256_add_ps(_mm256_add_ps(xi, xi), _mm256_set1_ps(kernel_width - 1.0f));

        // Prepare zd register so that we can obtain pairs using vpermilps
        // This vector now contains the subgrid offsets for the 8 points,
        // duplicated once in order to facilitate future shuffling.
        // [z_0, z_2, ..., z_0, z_2, ..., z_1, z_3, ..., z_1, z_3, ...]
        __m512 zd = _mm512_permutexvar_ps(
            _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 1, 3, 5, 7, 1, 3, 5, 7),
            _mm512_castps256_ps512(z));

        __m512 v1;
        __m512 v2;

        // Store integer coordinates for accumulation later (adjust indices to account for offset)
        alignas(16) int indices[8];
        _mm256_store_epi32(indices, _mm256_sub_epi32(x_ceili, _mm256_set1_epi32(offset)));

        // Unrolled loop to compute 8 values, two at once.
        // At each stage, we permute from xid into a register
        // which contains z_0 in the lower 256-bit, and z_1 in the upper 256-bit.
        compute_kernel(_mm512_permute_ps(zd, 0), strengths + 2 * i, v1, v2);
        accumulate_strengths(output, indices[0], v1);
        accumulate_strengths(output, indices[1], v2);

        // Permute to obtain z_2 in the lower 256-bit, and z_3 in the upper 256-bit.
        compute_kernel(_mm512_permute_ps(zd, 0b01010101), strengths + 2 * i + 4, v1, v2);
        accumulate_strengths(output, indices[2], v1);
        accumulate_strengths(output, indices[3], v2);

        // Permute to obtain z_4 in the lower 256-bit, and z_5 in the upper 256-bit.
        compute_kernel(_mm512_permute_ps(zd, 0b10101010), strengths + 2 * i + 8, v1, v2);
        accumulate_strengths(output, indices[4], v1);
        accumulate_strengths(output, indices[5], v2);

        // Permute to obtain z_6 in the lower 256-bit, and z_7 in the upper 256-bit.
        compute_kernel(_mm512_permute_ps(zd, 0b11111111), strengths + 2 * i + 12, v1, v2);
        accumulate_strengths(output, indices[6], v1);
        accumulate_strengths(output, indices[7], v2);
    }

    void operator()(
        // Main loop of the spreading subproblem.
        // This loop is unrolled to process 8 points at a time.
        nu_point_collection<1, float const *> const &input, grid_specification<1> const &grid,
        float *__restrict output) const {

        std::fill_n(output, grid.num_elements(), 0.0f);

        float const *coord_x = input.coordinates[0];
        float const *strengths = input.strengths;

        auto offset = grid.offsets[0];

        for (std::size_t i = 0; i < input.num_points; i += 8) {
            process_8(output, coord_x, strengths, offset, i);
        }
    }

    std::size_t num_points_multiple() const {
        // We process 8 points at a time.
        return 8;
    }
    std::size_t extent_multiple() const { return 1; }
    std::pair<double, double> target_padding() const {
        // We exceed the standard padding on the right by at most 8
        // Due to the split writing of the kernel.
        return {0.5 * kernel_width, 0.5 * kernel_width + 8};
    }
};

// Declare commonly used polynomial degrees for width-8 kernels.
extern template struct SpreadSubproblemPolyW8<7>;
extern template struct SpreadSubproblemPolyW8<8>;
extern template struct SpreadSubproblemPolyW8<9>;
extern template struct SpreadSubproblemPolyW8<10>;
extern template struct SpreadSubproblemPolyW8<11>;

/** Spreading functor for width-4 polynomial
 *
 * This functor implements a AVX-512 vectorized strategy for spreading based on
 * a width-8 polynomial approximation to the kernel function.
 *
 * The current implementation packs the evaluation of 4 4-wide polynomials
 * into a single 16-wide vector. Additionally, the computation of the base
 * index for accumulation is vectorized, so that in total 8 elements are
 * processed at a time in the inner loop, in 2 groups of 4.
 *
 */
template <std::size_t Degree> struct SpreadSubproblemPolyW4 {
    finufft::spreading::aligned_unique_array<float> coefficients;
    float kernel_width;

    template <typename U>
    SpreadSubproblemPolyW4(U const *coefficients, std::size_t width)
        : coefficients(allocate_aligned_array<float>((Degree + 1) * 16, 64)),
          kernel_width(static_cast<float>(width)) {

        // Replicate the polynomial coefficients 4 times as we will be evaluating the same
        // polynomial 4 times, one in each 128-bit lane.
        for (std::size_t i = 0; i < 4; ++i) {
            fill_polynomial_coefficients(
                Degree, coefficients, width, this->coefficients.get() + 4 * i, 4, 16);
        }
    }

    void compute_kernel(__m512 z, float const *dd, __m512 &v1, __m512 &v2) const {
        __m512 k = horner_polynomial_evaluation<Degree>(z, coefficients.get());

        // Load weights for the 4 points
        __m256 w = _mm256_load_ps(dd);
        __m512 w_re = _mm512_permutexvar_ps(
            _mm512_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6),
            _mm512_castps256_ps512(w));
        __m512 w_im = _mm512_permutexvar_ps(
            _mm512_setr_epi32(1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7),
            _mm512_castps256_ps512(w));

        // Compute real and imaginary parts of kernel for each point
        __m512 k_re = _mm512_mul_ps(k, w_re);
        __m512 k_im = _mm512_mul_ps(k, w_im);

        // Interleave real and imaginary parts for each point by pair
        const int from_b = (1 << 4);
        const int from_a = (0 << 4);

        // clang-format off
        v1 = _mm512_permutex2var_ps(
            k_re,
            _mm512_setr_epi32(
                from_a | 0, from_b | 0, from_a | 1, from_b | 1, from_a | 2, from_b | 2, from_a | 3, from_b | 3,
                from_a | 4, from_b | 4, from_a | 5, from_b | 5, from_a | 6, from_b | 6, from_a | 7, from_b | 7),
            k_im);
        v2 = _mm512_permutex2var_ps(
            k_re,
            _mm512_setr_epi32(
                from_a | 8, from_b | 8, from_a | 9, from_b | 9, from_a | 10, from_b | 10, from_a | 11, from_b | 11,
                from_a | 12, from_b | 12, from_a | 13, from_b | 13, from_a | 14, from_b | 14, from_a | 15, from_b | 15),
            k_im);
        // clang-format on
    }

    inline void accumulate_strengths(float *du, int i, __m256 v) const {
        int i_aligned = i & ~3;
        int i_remainder = i - i_aligned;

        float *out = du + 2 * i_aligned;

        // Split using double operation in order to shuffle pairs
        // of fp32 values (representing a complex number).
        __m256d v_lo, v_hi;
        split_unaligned_vector(_mm256_castps_pd(v), i_remainder, v_lo, v_hi);

        // Accumulate and store
        __m256 out_lo = _mm256_load_ps(out);
        __m256 out_hi = _mm256_load_ps(out + 8);
        out_lo = _mm256_add_ps(out_lo, _mm256_castpd_ps(v_lo));
        out_hi = _mm256_add_ps(out_hi, _mm256_castpd_ps(v_hi));
        _mm256_store_ps(out, out_lo);
        _mm256_store_ps(out + 8, out_hi);
    }

    void process_8(
        float *__restrict output, float const *coord_x, float const *strengths, int64_t offset,
        std::size_t i) const {
        // Load position of 8 non-uniform points, compute grid and subgrid offsets (vectorized)
        __m256 x = _mm256_load_ps(coord_x + i);
        __m256 x_ceil = _mm256_ceil_ps(_mm256_sub_ps(x, _mm256_set1_ps(0.5f * kernel_width)));
        __m256i x_ceili = _mm256_cvtps_epi32(x_ceil);
        __m256 xi = _mm256_sub_ps(x_ceil, x);

        // Normalized subgrid position for each point
        // [z_0, z_1, ...]
        __m256 z = _mm256_add_ps(_mm256_add_ps(xi, xi), _mm256_set1_ps(kernel_width - 1.0f));

        // Prepare zd register so that we can obtain pairs using vpermilps
        // This vector now contains the subgrid offsets for the 8 points,
        // duplicated once in order to facilitate future shuffling.
        // [z_0, z_2, ..., z_0, z_2, ..., z_1, z_3, ..., z_1, z_3, ...]
        __m512 zd = _mm512_permutexvar_ps(
            _mm512_setr_epi32(0, 4, 0, 4, 1, 5, 1, 5, 2, 6, 2, 6, 3, 7, 3, 7),
            _mm512_castps256_ps512(z));

        __m512 v1;
        __m512 v2;

        // Store integer coordinates for accumulation later (adjust indices to account for offset)
        alignas(16) int indices[8];
        _mm256_store_epi32(indices, _mm256_sub_epi32(x_ceili, _mm256_set1_epi32(offset)));

        // Unrolled loop to compute 8 values, four at once.
        // At each stage, we permute from zd into a register
        // which contains [z_0 x4, z_1 x4, z_2 x4, z_3 x4]
        compute_kernel(_mm512_permute_ps(zd, 0), strengths + 2 * i, v1, v2);
        accumulate_strengths(output, indices[0], _mm512_extractf32x8_ps(v1, 0));
        accumulate_strengths(output, indices[1], _mm512_extractf32x8_ps(v1, 1));
        accumulate_strengths(output, indices[2], _mm512_extractf32x8_ps(v2, 0));
        accumulate_strengths(output, indices[3], _mm512_extractf32x8_ps(v2, 1));

        compute_kernel(_mm512_permute_ps(zd, 0b11111111), strengths + 2 * i + 8, v1, v2);
        accumulate_strengths(output, indices[4], _mm512_extractf32x8_ps(v1, 0));
        accumulate_strengths(output, indices[5], _mm512_extractf32x8_ps(v1, 1));
        accumulate_strengths(output, indices[6], _mm512_extractf32x8_ps(v2, 0));
        accumulate_strengths(output, indices[7], _mm512_extractf32x8_ps(v2, 1));
    }

    void operator()(
        nu_point_collection<1, float const *> const &input, grid_specification<1> const &grid,
        float *__restrict output) const {

        std::fill_n(output, grid.num_elements(), 0.0f);

        float const *coord_x = input.coordinates[0];
        float const *strengths = input.strengths;

        auto offset = grid.offsets[0];

        for (std::size_t i = 0; i < input.num_points; i += 8) {
            process_8(output, coord_x, strengths, offset, i);
        }
    }

    std::size_t num_points_multiple() const {
        // We process 8 points at a time.
        return 8;
    }
    std::size_t extent_multiple() const { return 1; }
    std::pair<double, double> target_padding() const {
        // We exceed the standard padding on the right by at most 4
        // Due to the split writing of the kernel.
        return {0.5 * kernel_width, 0.5 * kernel_width + 4};
    }
};

extern template struct SpreadSubproblemPolyW4<4>;
extern template struct SpreadSubproblemPolyW4<5>;
extern template struct SpreadSubproblemPolyW4<6>;
extern template struct SpreadSubproblemPolyW4<7>;

template <std::size_t Degree> struct SpreadSubproblemPolyW8F64 {
    aligned_unique_array<double> coefficients;
    double kernel_width;

    template <typename U>
    SpreadSubproblemPolyW8F64(U const *coefficients, std::size_t width)
        : coefficients(allocate_aligned_array<double>(8 * (Degree + 1), 64)), kernel_width(width) {
        fill_polynomial_coefficients(Degree, coefficients, width, this->coefficients.get(), 8);

        // We need to adjust the coefficients at each degree to reorder in the width dimension.
        // The array is given to use in the standard order:
        //   a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8
        // and we wish to use it in the following order:
        //   a_1 a_4 a_2 a_6 a_3 a_7 a_4 a_8
        // To apply the permutation, we use the following factorization into cycles
        //   (1) (2 3 5) (4 7 6) (8)
        for (std::size_t i = 1; i < Degree + 1; ++i) {
            auto coeffs_d = this->coefficients.get() + i * 8;
            {
                double t = coeffs_d[1];
                coeffs_d[1] = coeffs_d[4];
                coeffs_d[4] = coeffs_d[2];
                coeffs_d[2] = t;
            }

            {
                double t = coeffs_d[3];
                coeffs_d[3] = coeffs_d[5];
                coeffs_d[5] = coeffs_d[6];
                coeffs_d[6] = t;
            }
        }
    }

    void compute_kernel(__m512d z, double const *dd, __m512d &v1, __m512d &v2) const {
        __m512d k = horner_polynomial_evaluation<Degree>(z, coefficients.get());

        // Load real and imaginary coefficients
        __m512d w_re = _mm512_set1_pd(dd[0]);
        __m512d w_im = _mm512_set1_pd(dd[1]);

        // Multiply by coefficients in lane.
        __m512d k_re = _mm512_mul_pd(k, w_re);
        __m512d k_im = _mm512_mul_pd(k, w_im);

        // To finish, we need to write out the results in interleaved format
        // Note that we rely on pre-shuffled coefficients to ensure that
        // the results are unpacked correctly.
        v1 = _mm512_unpacklo_pd(k_re, k_im);
        v2 = _mm512_unpackhi_pd(k_re, k_im);
    }

    void accumulate_strengths(double *du, int64_t i, __m512d const &v1, __m512d const &v2) const {
        int64_t i_aligned = i & ~3;
        int i_remainder = i - i_aligned;

        __m512d v_lo, v_mid, v_hi;
        // Offset multiplied by 2 to account for the fact that we are
        // working with interleaved complex values
        split_unaligned_vector(v1, v2, 2 * i_remainder, v_lo, v_mid, v_hi);

        __m512d out_lo = _mm512_load_pd(du + 2 * i_aligned);
        __m512d out_mid = _mm512_load_pd(du + 2 * i_aligned + 8);
        __m512d out_hi = _mm512_load_pd(du + 2 * i_aligned + 16);

        out_lo = _mm512_add_pd(out_lo, v_lo);
        out_mid = _mm512_add_pd(out_mid, v_mid);
        out_hi = _mm512_add_pd(out_hi, v_hi);

        _mm512_store_pd(du + 2 * i_aligned, out_lo);
        _mm512_store_pd(du + 2 * i_aligned + 8, out_mid);
        _mm512_store_pd(du + 2 * i_aligned + 16, out_hi);
    }

    /** Function for a single unrolled iteration of the subproblem.
     *
     * This is the core loop of the kernel, and operates on 8 points at a time
     * in order to leverage vectorization in the computation of the index into the grid.
     *
     */
    void process_4(
        double *__restrict output, double const *coord_x, double const *strengths, int64_t offset,
        std::size_t i) const {
        // Load position of 8 non-uniform points, compute grid and subgrid offsets (vectorized)
        __m256d x = _mm256_load_pd(coord_x + i);
        __m256d x_ceil = _mm256_ceil_pd(_mm256_sub_pd(x, _mm256_set1_pd(0.5 * kernel_width)));
        __m256i x_ceili = _mm256_cvtpd_epi64(x_ceil);
        __m256d xi = _mm256_sub_pd(x_ceil, x);

        // Normalized subgrid position for each point
        // [z_0, z_1, ...]
        __m256d z = _mm256_add_pd(_mm256_add_pd(xi, xi), _mm256_set1_pd(kernel_width - 1.0));

        // Prepare zd register so that we can obtain pairs using vpermilps
        // This vector now contains the subgrid offsets for the 8 points,
        // duplicated once in order to facilitate future shuffling.
        // [z_0, z_2, ..., z_0, z_2, ..., z_1, z_3, ..., z_1, z_3, ...]
        __m512d zd = _mm512_permutexvar_pd(
            _mm512_setr_epi64(0, 1, 2, 3, 0, 1, 2, 3), _mm512_castpd256_pd512(z));

        __m512d v1;
        __m512d v2;

        // Store integer coordinates for accumulation later (adjust indices to account for offset)
        alignas(16) int64_t indices[4];
        _mm256_store_epi64(indices, _mm256_sub_epi64(x_ceili, _mm256_set1_epi64x(offset)));

        // Unrolled loop to compute 4 values, one at a time.
        compute_kernel(_mm512_permutex_pd(zd, 0b00000000), strengths + 2 * i, v1, v2);
        accumulate_strengths(output, indices[0], v1, v2);

        compute_kernel(_mm512_permutex_pd(zd, 0b01010101), strengths + 2 * i + 2, v1, v2);
        accumulate_strengths(output, indices[1], v1, v2);

        compute_kernel(_mm512_permutex_pd(zd, 0b10101010), strengths + 2 * i + 4, v1, v2);
        accumulate_strengths(output, indices[2], v1, v2);

        compute_kernel(_mm512_permutex_pd(zd, 0b11111111), strengths + 2 * i + 6, v1, v2);
        accumulate_strengths(output, indices[3], v1, v2);
    }

    void operator()(
        nu_point_collection<1, double const *> const &input, grid_specification<1> const &grid,
        double *__restrict output) const {

        std::fill_n(output, grid.num_elements(), 0.0f);

        double const *coord_x = input.coordinates[0];
        double const *strengths = input.strengths;

        auto offset = grid.offsets[0];

        for (std::size_t i = 0; i < input.num_points; i += 4) {
            process_4(output, coord_x, strengths, offset, i);
        }
    }

    std::size_t num_points_multiple() const { return 4; }
    std::size_t extent_multiple() const { return 1; }
    std::pair<double, double> target_padding() const {
        return {0.5 * kernel_width, 0.5 * kernel_width + 4};
    }
};

extern template struct SpreadSubproblemPolyW8F64<7>;
extern template struct SpreadSubproblemPolyW8F64<8>;
extern template struct SpreadSubproblemPolyW8F64<9>;
extern template struct SpreadSubproblemPolyW8F64<10>;
extern template struct SpreadSubproblemPolyW8F64<11>;

} // namespace avx512
} // namespace spreading
} // namespace finufft
