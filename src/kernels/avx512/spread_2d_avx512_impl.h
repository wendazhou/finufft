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

template <std::size_t Degree> struct SpreadSubproblemPoly2DW8 {
    finufft::spreading::aligned_unique_array<float> coefficients;
    float kernel_width;

    template <typename U>
    SpreadSubproblemPoly2DW8(U const *coefficients, std::size_t width)
        : coefficients(allocate_aligned_array<float>((Degree + 1) * 16, 64)),
          kernel_width(static_cast<float>(width)) {

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
    void compute_kernel(__m512 z, float const *strengths, __m512 &vx, __m256 &vy) const {
        // Jointly evaluate kernel for x and y coordinates.
        // x coordinates held in the lower half, y coordinates in the upper half.
        __m512 k = horner_polynomial_evaluation<Degree>(z, coefficients.get());

        // Extract and duplicate kernel for x-axis
        __m512 kxd = _mm512_insertf32x8(k, _mm512_castps512_ps256(k), 1);
        __m512 w_re_im =
            _mm512_insertf32x8(_mm512_set1_ps(strengths[0]), _mm256_set1_ps(strengths[1]), 1);
        __m512 kx = _mm512_mul_ps(kxd, w_re_im);

        // shuffle kx to obtain the output pre-multiplied x-axis kernel values
        vx = _mm512_permutexvar_ps(
            _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15), kx);
        // Write out kernel values for y-axis
        vy = _mm512_extractf32x8_ps(k, 1);
    }

    void accumulate_strengths(
        float *output, int ix, int iy, std::size_t ldx, __m512 vx, __m256 vy) const {
        // Compute index as base index to aligned location
        // and offset from aligned location to actual index.
        int i_aligned = ix & ~7;
        int i_remainder = ix - i_aligned;

        float *out = output + iy * ldx + 2 * i_aligned;

        // Split using double operation in order to shuffle pairs
        // of fp32 values (representing a complex number).
        __m512d v_lod, v_hid;
        split_unaligned_vector(_mm512_castps_pd(vx), i_remainder, v_lod, v_hid);

        __m512 v_lo = _mm512_castpd_ps(v_lod);
        __m512 v_hi = _mm512_castpd_ps(v_hid);

        alignas(64) float vyf[8];
        _mm256_store_ps(vyf, vy);

        // Note: check performance impact of using
        // exact width
        for (std::size_t i = 0; i < 8; ++i) {
            __m512 out_lo = _mm512_load_ps(out + i * ldx);
            __m512 out_hi = _mm512_load_ps(out + i * ldx + 16);

            __m512 yi = _mm512_set1_ps(vyf[i]);
            out_lo = _mm512_fmadd_ps(v_lo, yi, out_lo);
            out_hi = _mm512_fmadd_ps(v_hi, yi, out_hi);

            _mm512_store_ps(out + i * ldx, out_lo);
            _mm512_store_ps(out + i * ldx + 16, out_hi);
        }
    }

    void process_4(
        float *__restrict output, float const *coord_x, float const *coord_y,
        float const *strengths, int offset_x, int offset_y, std::size_t ldx) const {

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
        __m256 z = _mm256_add_ps(_mm256_add_ps(xyi, xyi), _mm256_set1_ps(kernel_width - 1.0f));

        // Prepare zd register so that we can obtain pairs using vpermilps
        __m512 zd = _mm512_permutexvar_ps(
            _mm512_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7),
            _mm512_castps256_ps512(z));

        // Store integer coordinates for accumulation later (adjust indices to account for offset)
        // First 4 correspond to x indices, second 4 correspond to y indices.
        alignas(8) int indices[8];
        _mm256_store_epi32(
            indices,
            _mm256_sub_epi32(
                xy_ceili, _mm256_setr_m128i(_mm_set1_epi32(offset_x), _mm_set1_epi32(offset_y))));

        __m512 vx;
        __m256 vy;

        // Unrolled loop to compute 8 values, two at once.
        // At each stage, we permute from xid into a register
        compute_kernel(_mm512_permute_ps(zd, 0b00000000), strengths, vx, vy);
        accumulate_strengths(output, indices[0], indices[4], ldx, vx, vy);

        compute_kernel(_mm512_permute_ps(zd, 0b01010101), strengths + 2, vx, vy);
        accumulate_strengths(output, indices[1], indices[5], ldx, vx, vy);

        compute_kernel(_mm512_permute_ps(zd, 0b10101010), strengths + 4, vx, vy);
        accumulate_strengths(output, indices[2], indices[6], ldx, vx, vy);

        compute_kernel(_mm512_permute_ps(zd, 0b11111111), strengths + 8, vx, vy);
        accumulate_strengths(output, indices[3], indices[7], ldx, vx, vy);
    }

    void operator()(
        // Main loop of the spreading subproblem.
        // This loop is unrolled to process 8 points at a time.
        nu_point_collection<2, float const *> const &input, grid_specification<2> const &grid,
        float *__restrict output) const {

        std::fill_n(output, grid.num_elements(), 0.0f);

        float const *coord_x = input.coordinates[0];
        float const *coord_y = input.coordinates[1];
        float const *strengths = input.strengths;

        auto offset_x = grid.offsets[0];
        auto offset_y = grid.offsets[1];
        auto ldx = 2 * grid.extents[0];

        for (std::size_t i = 0; i < input.num_points; i += 4) {
            process_4(output, coord_x + i, coord_y + i, strengths + 2 * i, offset_x, offset_y, ldx);
        }
    }

    std::size_t num_points_multiple() const { return 4; }
    std::array<std::size_t, 2> extent_multiple() const { return {8, 1}; }
    std::array<std::pair<double, double>, 2> target_padding() const {
        double ns2 = 0.5 * kernel_width;
        return {std::pair<double, double>{ns2, ns2 + 8}, {ns2, -ns2 + 8}};
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

} // namespace avx512

} // namespace spreading
} // namespace finufft
