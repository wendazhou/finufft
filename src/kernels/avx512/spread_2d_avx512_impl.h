#pragma once

#include <cstddef>

#include "align_split_routines.h"
#include "poly_eval_routines.h"

#include "../reference/spreading_reference.h"

/** @file
 *
 * Implementation for 2-dimensional spreading subproblem with avx-512 intrinsics.
 * The public interface for obtaining functors leveraging these implementations
 * may be found in `spread_axv512.h`.
 *
 */

namespace finufft {
namespace spreading {
namespace avx512 {

/** 2-Dimension width-8 spread subproblem kernel
 *
 * This functor implements an AVX-512 vectorized strategy for width 8
 * kernels based on a polynomial approximation.
 *
 * This implementation packs the evaluation of the x and y polynomials
 * into a single 16-wide vector. The computation of the base index is
 * additionally vectorized, so that the inner loop processes 4 elements.
 *
 */
template <std::size_t Degree> struct SpreadSubproblemPoly2DW8 {
    finufft::spreading::aligned_unique_array<float> coefficients;
    float kernel_width;
    std::size_t writeout_width;

    template <typename U>
    SpreadSubproblemPoly2DW8(U const *coefficients, std::size_t width)
        : coefficients(allocate_aligned_array<float>((Degree + 1) * 16, 64)),
          kernel_width(static_cast<float>(width)), writeout_width(width) {

        // Duplicate the polynomial coefficients as we will be evaluating the same
        // polynomial twice, one in each 256-bit lane.
        fill_polynomial_coefficients(Degree, coefficients, width, this->coefficients.get(), 8, 16);
        fill_polynomial_coefficients(
            Degree, coefficients, width, this->coefficients.get() + 8, 8, 16);
    }

    /** Computes kernel for both x and y axis.
     *
     * This function computes the kernel for both x and y axis,
     * and pre-multiplies the x-axis values by the complex weights.
     *
     * @param vx[out] The x-axis kernel, multiplied by the complex strength, in interleaved format.
     * @param vy[out] The y-axis kernel without multiplication.
     *
     */
    void compute_kernel(
        __m512 x, __m512 y, float const *strengths, __m512 &vx1, __m512 &vx2, float *vy) const {
        poly_eval_multiply_strengths_2x8<Degree>(x, this->coefficients.get(), strengths, vx1, vx2);
        __m512 k_y = horner_polynomial_evaluation<Degree>(y, this->coefficients.get());
        _mm512_store_ps(vy, k_y);
    }

    /** Accumulates kernel from pre-multiplied x value and plain y value,
     * with splitting across the fastest varying dimension.
     *
     * This function accumulates the kernel into the output array.
     * The kernel is provided pre-multiplied by the complex weights for `vx`,
     * but without the pre-multiplication for `vy`.
     * To ensure best performance in the contiguous x-axis, the vector is split
     * into its aligned component according to the offset. Each segment is
     * then written into the corresponding array slice.
     *
     */
    void accumulate_strengths(
        float *out, std::size_t remainder, std::size_t stride_y, __m512 vx,
        float const *vy) const {
        // Compute index as base index to aligned location
        // and offset from aligned location to actual index.

        // Split using double operation in order to shuffle pairs
        // of fp32 values (representing a complex number).
        __m512d v_lod, v_hid;
        split_unaligned_vector(_mm512_castps_pd(vx), remainder, v_lod, v_hid);

        __m512 v_lo = _mm512_castpd_ps(v_lod);
        __m512 v_hi = _mm512_castpd_ps(v_hid);

        // Only loop up to writeout_width, as there is no
        // need to write out values beyond that (they are zero).
        // We also save a little bit on padding.
        for (std::size_t i = 0; i < writeout_width; ++i) {
            auto out_base = out + i * stride_y;

            __m512 out_lo = _mm512_load_ps(out_base);
            __m512 out_hi = _mm512_load_ps(out_base + 16);

            __m512 yi = _mm512_set1_ps(vy[i]);
            out_lo = _mm512_fmadd_ps(v_lo, yi, out_lo);
            out_hi = _mm512_fmadd_ps(v_hi, yi, out_hi);

            _mm512_store_ps(out_base, out_lo);
            _mm512_store_ps(out_base + 16, out_hi);
        }
    }

    void process_4(
        float *__restrict output, float const *coord_x, float const *coord_y,
        float const *strengths, int offset_x, int offset_y, std::size_t stride_y) const {

        // Load position of 4 non-uniform points, compute grid and subgrid offsets (vectorized)
        // For better efficiency, we jointly process the x and y coordinates of the 4 points
        // in a 8-wide vector, with the x coordinates in the lower half and the y coordinates
        // in the upper half.
        __m128 x = _mm_load_ps(coord_x);
        __m128 y = _mm_load_ps(coord_y);
        __m256 xy = _mm256_setr_m128(x, y);

        __m256 xy_ceil = _mm256_ceil_ps(_mm256_sub_ps(xy, _mm256_set1_ps(0.5f * kernel_width)));
        __m256i xy_ceili = _mm256_cvtps_epi32(xy_ceil);
        __m256 xyi = _mm256_sub_ps(xy_ceil, xy);

        // Normalized subgrid position for each point
        // [z_0, z_1, ...]
        __m256 xy_n = _mm256_add_ps(_mm256_add_ps(xyi, xyi), _mm256_set1_ps(kernel_width - 1.0f));

        // Prepare zd register so that we can obtain pairs using vpermilps
        __m512 xy_nd = _mm512_permutexvar_ps(
            _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 1, 3, 5, 7, 1, 3, 5, 7),
            _mm512_castps256_ps512(xy_n));

        // Store integer coordinates for accumulation later (adjust indices to account for offset)
        // First 4 correspond to x indices, second 4 correspond to y indices.
        alignas(16) uint32_t indices_base[4];
        alignas(16) uint32_t indices_offset[4];

        {
            __m128i idx_x =
                _mm_sub_epi32(_mm256_castsi256_si128(xy_ceili), _mm_set1_epi32(offset_x));
            __m128i idx_y =
                _mm_sub_epi32(_mm256_extracti128_si256(xy_ceili, 1), _mm_set1_epi32(offset_y));
            __m128i idx_x_aligned = _mm_and_si128(_mm_set1_epi32(~7), idx_x);
            __m128i idx_x_remainder = _mm_sub_epi32(idx_x, idx_x_aligned);
            // base = 2 * idx_x_aligned + idx_y * stride_y
            __m128i idx_base = _mm_add_epi32(
                _mm_add_epi32(idx_x_aligned, idx_x_aligned),
                _mm_mullo_epi32(idx_y, _mm_set1_epi32(stride_y)));
            _mm_store_epi32(indices_base, idx_base);
            _mm_store_epi32(indices_offset, idx_x_remainder);
        }

        __m512 vx1, vx2;
        alignas(64) float vy[16];

        // Unrolled loop to compute 4 values, two at once.
        compute_kernel(
            _mm512_permute_ps(xy_nd, 0b0000'0000),
            _mm512_permute_ps(xy_nd, 0b1010'1010),
            strengths,
            vx1,
            vx2,
            vy);
        accumulate_strengths(output + indices_base[0], indices_offset[0], stride_y, vx1, vy);
        accumulate_strengths(output + indices_base[1], indices_offset[1], stride_y, vx2, vy + 8);

        compute_kernel(
            _mm512_permute_ps(xy_nd, 0b0101'0101),
            _mm512_permute_ps(xy_nd, 0b1111'1111),
            strengths + 4,
            vx1,
            vx2,
            vy);
        accumulate_strengths(output + indices_base[2], indices_offset[2], stride_y, vx1, vy);
        accumulate_strengths(output + indices_base[3], indices_offset[3], stride_y, vx2, vy + 8);
    }

    void operator()(
        // Main loop of the spreading subproblem.
        // This loop is unrolled to process 8 points at a time.
        nu_point_collection<2, float const *> const &input, grid_specification<2> const &grid,
        float *__restrict output) const {

        std::fill_n(output, 2 * grid.num_elements(), 0.0f);

        float const *coord_x = input.coordinates[0];
        float const *coord_y = input.coordinates[1];
        float const *strengths = input.strengths;

        auto offset_x = grid.offsets[0];
        auto offset_y = grid.offsets[1];
        auto stride_y = 2 * grid.extents[0];

        for (std::size_t i = 0; i < input.num_points; i += 4) {
            process_4(
                output, coord_x + i, coord_y + i, strengths + 2 * i, offset_x, offset_y, stride_y);
        }
    }

    std::size_t num_points_multiple() const { return 4; }
    std::array<std::size_t, 2> extent_multiple() const { return {8, 1}; }
    std::array<std::pair<double, double>, 2> target_padding() const {
        double ns2 = 0.5 * kernel_width;
        return {std::pair<double, double>{ns2, ns2 + 8}, {ns2, ns2}};
    }
};

extern template struct SpreadSubproblemPoly2DW8<4>;
extern template struct SpreadSubproblemPoly2DW8<5>;
extern template struct SpreadSubproblemPoly2DW8<6>;
extern template struct SpreadSubproblemPoly2DW8<7>;
extern template struct SpreadSubproblemPoly2DW8<8>;
extern template struct SpreadSubproblemPoly2DW8<9>;
extern template struct SpreadSubproblemPoly2DW8<10>;
extern template struct SpreadSubproblemPoly2DW8<11>;

template <std::size_t Degree> struct SpreadSubproblemPoly2DW8F64 {
    finufft::spreading::aligned_unique_array<double> coefficients;
    double kernel_width;
    std::size_t writeout_width;

    template <typename U>
    SpreadSubproblemPoly2DW8F64(U const *coefficients, std::size_t width)
        : coefficients(allocate_aligned_array<double>((Degree + 1) * 8, 64)),
          kernel_width(static_cast<double>(width)), writeout_width(width) {

        fill_polynomial_coefficients(Degree, coefficients, width, this->coefficients.get(), 8);
    }

    /** Computes kernel for both x and y axis.
     *
     * This function computes the kernel for both x and y axis,
     * and pre-multiplies the x-axis values by the complex weights.
     *
     * @param vx1[out] The lower half of the x-axis kernel, multiplied by the complex strength, in
     * interleaved format.
     * @param vx2[out] The upper half of the x-axis kernel, multiplied by the complex strength, in
     * interleaved format.
     * @param vy[out] The y-axis kernel without multiplication.
     *
     */
    void compute_kernel(
        __m512d z, double const *strengths, __m512d &vx1, __m512d &vx2, __m512d &vy) const {
        // Extract x and y coordinates from z.
        // Input format: z = [x y x y x y x y]
        __m512d zx = _mm512_permute_pd(z, 0b00000000);
        __m512d zy = _mm512_permute_pd(z, 0b11111111);

        // Evaluate kernels for x then y.
        __m512d kx = horner_polynomial_evaluation<Degree>(zx, coefficients.get());
        __m512d ky = horner_polynomial_evaluation<Degree>(zy, coefficients.get());

        // Compute pre-multiplied x
        __m512d kx_re = _mm512_mul_pd(kx, _mm512_set1_pd(strengths[0]));
        __m512d kx_im = _mm512_mul_pd(kx, _mm512_set1_pd(strengths[1]));

        const int re = 0b0000;
        const int im = 0b1000;

        // Interleave real and imaginary parts of x-axis kernel.
        // Note: we could pre-shuffle polynomial weights to use more efficient (latency-wise)
        // shuffle instructions such as unpacklo_pd, but this would incur an additional shuffle
        // for the y-axis kernel. There is currently no advantage on Intel Ice-Lake and previous
        // architectures are all shuffles contend for port 5.
        vx1 = _mm512_permutex2var_pd(
            kx_re,
            _mm512_setr_epi64(re | 0, im | 0, re | 1, im | 1, re | 2, im | 2, re | 3, im | 3),
            kx_im);
        vx2 = _mm512_permutex2var_pd(
            kx_re,
            _mm512_setr_epi64(re | 4, im | 4, re | 5, im | 5, re | 6, im | 6, re | 7, im | 7),
            kx_im);

        vy = ky;
    }

    void accumulate_strengths(
        double *output, int ix, int iy, std::size_t stride_y, __m512d vx1, __m512d vx2,
        __m512d vy) const {
        // Compute index as base index to aligned location
        // and offset from aligned location to actual index.
        int i_aligned = ix & ~3;
        int i_remainder = ix - i_aligned;

        double *out = output + iy * stride_y + 2 * i_aligned;

        // Split using double operation in order to shuffle pairs
        // of fp32 values (representing a complex number).
        __m512d v_lo, v_mid, v_hi;
        split_unaligned_vector(vx1, vx2, 2 * i_remainder, v_lo, v_mid, v_hi);

        alignas(64) double vyf[8];
        _mm512_store_pd(vyf, vy);

        // Only loop up to writeout_width, as there is no
        // need to write out values beyond that (they are zero).
        // We also save a little bit on padding.
        for (std::size_t i = 0; i < writeout_width; ++i) {
            __m512d out_lo = _mm512_load_pd(out + i * stride_y);
            __m512d out_mid = _mm512_load_pd(out + i * stride_y + 8);
            __m512d out_hi = _mm512_load_pd(out + i * stride_y + 16);

            __m512d yi = _mm512_set1_pd(vyf[i]);
            out_lo = _mm512_fmadd_pd(v_lo, yi, out_lo);
            out_mid = _mm512_fmadd_pd(v_mid, yi, out_mid);
            out_hi = _mm512_fmadd_pd(v_hi, yi, out_hi);

            _mm512_store_pd(out + i * stride_y, out_lo);
            _mm512_store_pd(out + i * stride_y + 8, out_mid);
            _mm512_store_pd(out + i * stride_y + 16, out_hi);
        }
    }

    void process_4(
        double *__restrict output, double const *coord_x, double const *coord_y,
        double const *strengths, int offset_x, int offset_y, std::size_t stride_y) const {

        // Load position of 4 non-uniform points, compute grid and subgrid offsets (vectorized)
        // For better efficiency, we jointly process the x and y coordinates of the 4 points
        // in a 8-wide vector, with the x coordinates in the lower half and the y coordinates
        // in the upper half.
        __m256d x = _mm256_load_pd(coord_x);
        __m256d y = _mm256_load_pd(coord_y);
        __m512d xy = _mm512_insertf64x4(_mm512_castpd256_pd512(x), y, 1);

        __m512d xy_ceil = _mm512_ceil_pd(_mm512_sub_pd(xy, _mm512_set1_pd(0.5 * kernel_width)));
        __m512i xy_ceili = _mm512_cvtpd_epi64(xy_ceil);
        __m512d xyi = _mm512_sub_pd(xy_ceil, xy);

        // Normalized subgrid position for each point
        // [z_0, z_1, ...]
        __m512d z = _mm512_add_pd(_mm512_add_pd(xyi, xyi), _mm512_set1_pd(kernel_width - 1.0));

        // Prepare z1 and z2 register so that we can process each point using vector instructions.
        // In order to avoid full-vector shuffles, we perform them here so that we only require
        // within 256-bit lane shuffles to dispatch for each point.
        //
        // z1 = [x1, y1, x2, y2, x1, y1, x2, y2]
        // z2 = [x3, y3, x4, y4, x3, y3, x4, y4]
        __m512d z1 = _mm512_permutexvar_pd(_mm512_setr_epi64(0, 4, 1, 5, 0, 4, 1, 5), z);
        __m512d z2 = _mm512_permutexvar_pd(_mm512_setr_epi64(2, 6, 3, 7, 2, 6, 3, 7), z);

        // Store integer coordinates for accumulation later (adjust indices to account for offset)
        // First 4 correspond to x indices, second 4 correspond to y indices.
        alignas(64) int64_t indices[8];
        _mm512_store_epi64(
            indices,
            _mm512_sub_epi64(
                xy_ceili,
                _mm512_inserti64x4(_mm512_set1_epi64(offset_x), _mm256_set1_epi64x(offset_y), 1)));

        __m512d vx1, vx2, vy;

        // Unrolled loop to compute 4 points, one pair of x-y coordinates at a time.
        // At each stage, we permute from the z1 or z2 register into a register
        // of the form z = [x y x y x y x y], selecting the appropriate pair
        // corresponding to the current point.
        compute_kernel(_mm512_permutex_pd(z1, 0b01000100), strengths, vx1, vx2, vy);
        accumulate_strengths(output, indices[0], indices[4], stride_y, vx1, vx2, vy);

        compute_kernel(_mm512_permutex_pd(z1, 0b11101110), strengths + 2, vx1, vx2, vy);
        accumulate_strengths(output, indices[1], indices[5], stride_y, vx1, vx2, vy);

        compute_kernel(_mm512_permutex_pd(z2, 0b01000100), strengths + 4, vx1, vx2, vy);
        accumulate_strengths(output, indices[2], indices[6], stride_y, vx1, vx2, vy);

        compute_kernel(_mm512_permutex_pd(z2, 0b11101110), strengths + 6, vx1, vx2, vy);
        accumulate_strengths(output, indices[3], indices[7], stride_y, vx1, vx2, vy);
    }

    void operator()(
        // Main loop of the spreading subproblem.
        // This loop is unrolled to process 8 points at a time.
        nu_point_collection<2, double const *> const &input, grid_specification<2> const &grid,
        double *__restrict output) const {

        std::fill_n(output, 2 * grid.num_elements(), 0.0);

        double const *coord_x = input.coordinates[0];
        double const *coord_y = input.coordinates[1];
        double const *strengths = input.strengths;

        auto offset_x = grid.offsets[0];
        auto offset_y = grid.offsets[1];
        auto stride_y = 2 * grid.extents[0];

        for (std::size_t i = 0; i < input.num_points; i += 4) {
            process_4(
                output, coord_x + i, coord_y + i, strengths + 2 * i, offset_x, offset_y, stride_y);
        }
    }

    std::size_t num_points_multiple() const { return 4; }
    std::array<std::size_t, 2> extent_multiple() const { return {4, 1}; }
    std::array<std::pair<double, double>, 2> target_padding() const {
        double ns2 = 0.5 * kernel_width;
        return {std::pair<double, double>{ns2, ns2 + 4}, {ns2, ns2}};
    }
};

extern template struct SpreadSubproblemPoly2DW8F64<4>;
extern template struct SpreadSubproblemPoly2DW8F64<5>;
extern template struct SpreadSubproblemPoly2DW8F64<6>;
extern template struct SpreadSubproblemPoly2DW8F64<7>;
extern template struct SpreadSubproblemPoly2DW8F64<8>;
extern template struct SpreadSubproblemPoly2DW8F64<9>;
extern template struct SpreadSubproblemPoly2DW8F64<10>;
extern template struct SpreadSubproblemPoly2DW8F64<11>;

} // namespace avx512

} // namespace spreading
} // namespace finufft
