#pragma once

#include <cstddef>
#include <cstring>
#include <immintrin.h>

#include "align_split_routines.h"
#include "poly_eval_routines.h"

#include "../reference/spread_subproblem_reference.h"

namespace finufft {
namespace spreading {
namespace avx512 {

template <std::size_t Degree> struct SpreadSubproblemPoly3DW8 {
    aligned_unique_array<float> coefficients;
    float kernel_width;
    std::size_t writeout_width;

    template <typename U>
    SpreadSubproblemPoly3DW8(U const *coefficients, std::size_t width)
        : coefficients(allocate_aligned_array<float>((Degree + 1) * 16, 64)),
          kernel_width(static_cast<float>(width)), writeout_width(width) {

        // Duplicate the polynomial coefficients as we will be evaluating the same
        // polynomial twice, one in each 256-bit lane.
        fill_polynomial_coefficients(Degree, coefficients, width, this->coefficients.get(), 8, 16);
        fill_polynomial_coefficients(
            Degree, coefficients, width, this->coefficients.get() + 8, 8, 16);
    }

    /** Computes kernel for x, y, z axes, and pre-multiplies strengths into x-axis.
     *
     * This function computes the kernel for the x, y, and z axes.
     * and pre-multiplies the x-axis values by the complex weights.
     * In order to fit the 3 8-wide kernels into the 16-wide machine vectors,
     * it evaluates two points at a time.
     *
     *
     */
    void compute_kernel(
        __m512 x, __m512 y, __m512 z, float const *strengths, __m512 &vx1, __m512 &vx2, float *vy,
        float *vz) const {
        poly_eval_multiply_strengths_2x8<Degree>(x, coefficients.get(), strengths, vx1, vx2);

        __m512 k_y = horner_polynomial_evaluation<Degree>(y, coefficients.get());
        _mm512_store_ps(vy, k_y);

        __m512 k_z = horner_polynomial_evaluation<Degree>(z, coefficients.get());
        _mm512_store_ps(vz, k_z);
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
        float *__restrict output, int ix, int iy, int iz, std::size_t stride_y,
        std::size_t stride_z, __m512 vx, float const *vy, float const *vz) const {
        // Compute index as base index to aligned location
        // and offset from aligned location to actual index.
        int i_aligned = ix & ~7;
        int i_remainder = ix - i_aligned;

        float *out = output + iz * stride_z + iy * stride_y + 2 * i_aligned;

        // Split using double operation in order to shuffle pairs
        // of fp32 values (representing a complex number).
        __m512d v_lod, v_hid;
        split_unaligned_vector(_mm512_castps_pd(vx), i_remainder, v_lod, v_hid);

        __m512 v_lo = _mm512_castpd_ps(v_lod);
        __m512 v_hi = _mm512_castpd_ps(v_hid);

        // Only loop up to writeout_width, as there is no
        // need to write out values beyond that (they are zero).
        // We also save a little bit on padding.
        for (std::size_t dz = 0; dz < writeout_width; ++dz) {
            for (std::size_t dy = 0; dy < writeout_width; ++dy) {
                auto out_base = out + dz * stride_z + dy * stride_y;
                __m512 out_lo = _mm512_load_ps(out_base);
                __m512 out_hi = _mm512_load_ps(out_base + 16);

                __m512 k_yz = _mm512_set1_ps(vy[dy] * vz[dz]);
                out_lo = _mm512_fmadd_ps(v_lo, k_yz, out_lo);
                out_hi = _mm512_fmadd_ps(v_hi, k_yz, out_hi);

                _mm512_store_ps(out_base, out_lo);
                _mm512_store_ps(out_base + 16, out_hi);
            }
        }
    }

    void process_4(
        float *__restrict output, float const *coord_x, float const *coord_y, float const *coord_z,
        float const *strengths, int offset_x, int offset_y, int offset_z, std::size_t stride_y,
        std::size_t stride_z) const {

        // Load position of 4 non-uniform points, compute grid and subgrid offsets (vectorized)
        // For better efficiency, we jointly process the x and y coordinates of the 4 points
        // in a 8-wide vector, with the x coordinates in the lower half and the y coordinates
        // in the upper half.
        __m128 x = _mm_load_ps(coord_x);
        __m128 y = _mm_load_ps(coord_y);
        __m128 z = _mm_load_ps(coord_z);
        __m512 xyz = _mm512_insertf32x4(_mm512_castps256_ps512(_mm256_setr_m128(x, y)), z, 2);

        __m512 xyz_ceil = _mm512_ceil_ps(_mm512_sub_ps(xyz, _mm512_set1_ps(0.5f * kernel_width)));
        __m512i xyz_ceili = _mm512_cvtps_epi32(xyz_ceil);
        __m512 xyzi = _mm512_sub_ps(xyz_ceil, xyz);

        // Normalized subgrid position for each point
        // [z_0, z_1, ...]
        __m512 xyz_n =
            _mm512_add_ps(_mm512_add_ps(xyzi, xyzi), _mm512_set1_ps(kernel_width - 1.0f));

        // Prepare duplicated registers to obtain coordinates
        __m512 xy_d = _mm512_permutexvar_ps(
            _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 1, 3, 5, 7, 1, 3, 5, 7), xyz_n);
        __m512 z_d = _mm512_permutexvar_ps(
            _mm512_setr_epi32(8, 10, 12, 14, 8, 10, 12, 14, 9, 11, 13, 15, 9, 11, 13, 15), xyz_n);

        // Store integer coordinates for accumulation later (adjust indices to account for offset)
        // First 4 correspond to x indices, second 4 correspond to y indices.
        alignas(64) int indices[16];
        __m512i offsets = _mm512_inserti32x4(
            _mm512_castsi256_si512(
                _mm256_setr_m128i(_mm_set1_epi32(offset_x), _mm_set1_epi32(offset_y))),
            _mm_set1_epi32(offset_z),
            2);
        _mm512_store_epi32(indices, _mm512_sub_epi32(xyz_ceili, offsets));

        __m512 vx1, vx2;
        alignas(64) float vyf[16], vzf[16];

        // Unrolled loop to compute 4 values, two at once.
        compute_kernel(
            _mm512_permute_ps(xy_d, 0b0000'0000),
            _mm512_permute_ps(xy_d, 0b1010'1010),
            _mm512_permute_ps(z_d, 0b0000'0000),
            strengths,
            vx1,
            vx2,
            vyf,
            vzf);
        accumulate_strengths(
            output, indices[0], indices[4], indices[8], stride_y, stride_z, vx1, vyf, vzf);
        accumulate_strengths(
            output, indices[1], indices[5], indices[9], stride_y, stride_z, vx2, vyf + 8, vzf + 8);

        compute_kernel(
            _mm512_permute_ps(xy_d, 0b0101'0101),
            _mm512_permute_ps(xy_d, 0b1111'1111),
            _mm512_permute_ps(z_d, 0b0101'0101),
            strengths + 4,
            vx1,
            vx2,
            vyf,
            vzf);
        accumulate_strengths(
            output, indices[2], indices[6], indices[10], stride_y, stride_z, vx1, vyf, vzf);
        accumulate_strengths(
            output, indices[3], indices[7], indices[11], stride_y, stride_z, vx2, vyf + 8, vzf + 8);
    }

    void operator()(
        // Main loop of the spreading subproblem.
        // This loop is unrolled to process 4 points at a time.
        nu_point_collection<3, float const> const &input, subgrid_specification<3> const &grid,
        float *__restrict output) const {

        std::memset(output, 0, 2 * grid.num_elements() * sizeof(float));

        float const *coord_x = input.coordinates[0];
        float const *coord_y = input.coordinates[1];
        float const *coord_z = input.coordinates[2];
        float const *strengths = input.strengths;

        auto offset_x = grid.offsets[0];
        auto offset_y = grid.offsets[1];
        auto offset_z = grid.offsets[2];

        auto stride_y = 2 * grid.strides[1];
        auto stride_z = 2 * grid.strides[2];

        for (std::size_t i = 0; i < input.num_points; i += 4) {
            process_4(
                output,
                coord_x + i,
                coord_y + i,
                coord_z + i,
                strengths + 2 * i,
                offset_x,
                offset_y,
                offset_z,
                stride_y,
                stride_z);
        }
    }

    std::size_t num_points_multiple() const { return 4; }
    std::array<std::size_t, 3> extent_multiple() const { return {8, 1, 1}; }
    std::array<KernelWriteSpec<float>, 3> target_padding() const {
        float ns2 = 0.5 * kernel_width;
        return {
            KernelWriteSpec<float>{ns2, 0, 16},
            {ns2, 0, static_cast<int>(writeout_width)},
            {ns2, 0, static_cast<int>(writeout_width)}};
    }
};

extern template struct SpreadSubproblemPoly3DW8<4>;
extern template struct SpreadSubproblemPoly3DW8<5>;
extern template struct SpreadSubproblemPoly3DW8<6>;
extern template struct SpreadSubproblemPoly3DW8<7>;
extern template struct SpreadSubproblemPoly3DW8<8>;
extern template struct SpreadSubproblemPoly3DW8<9>;
extern template struct SpreadSubproblemPoly3DW8<10>;
extern template struct SpreadSubproblemPoly3DW8<11>;

template <std::size_t Degree> struct SpreadSubproblemPoly3DW8F64 {
    aligned_unique_array<double> coefficients;
    double kernel_width;
    std::size_t writeout_width;

    template <typename U>
    SpreadSubproblemPoly3DW8F64(U const *coefficients, std::size_t width)
        : coefficients(allocate_aligned_array<double>((Degree + 1) * 8, 64)),
          kernel_width(static_cast<float>(width)), writeout_width(width) {

        // Duplicate the polynomial coefficients as we will be evaluating the same
        // polynomial twice, one in each 256-bit lane.
        fill_polynomial_coefficients(Degree, coefficients, width, this->coefficients.get(), 8);
    }

    /** Computes kernel for x, y, z axes, and pre-multiplies strengths into x-axis.
     *
     * This function computes the kernel for the x, y, and z axes.
     * and pre-multiplies the x-axis values by the complex weights.
     * In order to fit the 3 8-wide kernels into the 16-wide machine vectors,
     * it evaluates two points at a time.
     *
     *
     */
    void compute_kernel(
        __m512d x, __m512d y, __m512d z, double const *strengths, __m512d &vx_lo, __m512d &vx_hi,
        double *vy, double *vz) const {
        __m512d k_y = horner_polynomial_evaluation<Degree>(y, coefficients.get());
        _mm512_store_pd(vy, k_y);

        __m512d k_z = horner_polynomial_evaluation<Degree>(z, coefficients.get());
        _mm512_store_pd(vz, k_z);

        __m512d k_x = horner_polynomial_evaluation<Degree>(x, coefficients.get());

        __m512d kx_re = _mm512_mul_pd(k_x, _mm512_set1_pd(strengths[0]));
        __m512d kx_im = _mm512_mul_pd(k_x, _mm512_set1_pd(strengths[1]));

        interleave_real_imaginary(kx_re, kx_im, vx_lo, vx_hi);
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
        double *__restrict output, int64_t ix, int64_t iy, int64_t iz, std::size_t stride_y,
        std::size_t stride_z, __m512d vx_lo, __m512d vx_hi, double const *vy,
        double const *vz) const {
        // Compute index as base index to aligned location
        // and offset from aligned location to actual index.
        int64_t i_aligned = ix & ~3;
        int64_t i_remainder = ix - i_aligned;

        double *out = output + iz * stride_z + iy * stride_y + 2 * i_aligned;

        // Split using double operation in order to shuffle pairs
        // of fp32 values (representing a complex number).
        __m512d v_lo, v_mid, v_hi;
        split_unaligned_vector(vx_lo, vx_hi, 2 * i_remainder, v_lo, v_mid, v_hi);

        // Only loop up to writeout_width, as there is no
        // need to write out values beyond that (they are zero).
        // We also save a little bit on padding.
        for (std::size_t dz = 0; dz < writeout_width; ++dz) {
            for (std::size_t dy = 0; dy < writeout_width; ++dy) {
                auto out_base = out + dz * stride_z + dy * stride_y;
                __m512d out_lo = _mm512_load_pd(out_base);
                __m512d out_mid = _mm512_load_pd(out_base + 8);
                __m512d out_hi = _mm512_load_pd(out_base + 16);

                __m512d k_yz = _mm512_set1_pd(vy[dy] * vz[dz]);
                out_lo = _mm512_fmadd_pd(v_lo, k_yz, out_lo);
                out_mid = _mm512_fmadd_pd(v_mid, k_yz, out_mid);
                out_hi = _mm512_fmadd_pd(v_hi, k_yz, out_hi);

                _mm512_store_pd(out_base, out_lo);
                _mm512_store_pd(out_base + 8, out_mid);
                _mm512_store_pd(out_base + 16, out_hi);
            }
        }
    }

    void process_4(
        double *__restrict output, double const *coord_x, double const *coord_y,
        double const *coord_z, double const *strengths, int offset_x, int offset_y, int offset_z,
        std::size_t stride_y, std::size_t stride_z) const {

        // Load position of 4 non-uniform points, compute grid and subgrid offsets (vectorized)
        // For better efficiency, we jointly process the x and y coordinates of the 4 points
        // in a 8-wide vector, with the x coordinates in the lower half and the y coordinates
        // in the upper half.
        __m256d x = _mm256_load_pd(coord_x);
        __m256d y = _mm256_load_pd(coord_y);
        __m512d z = _mm512_castpd256_pd512(_mm256_load_pd(coord_z));

        __m512d xy = _mm512_insertf64x4(_mm512_castpd256_pd512(x), y, 1);

        __m512d xy_ceil = _mm512_ceil_pd(_mm512_sub_pd(xy, _mm512_set1_pd(0.5 * kernel_width)));
        __m512d z_ceil = _mm512_ceil_pd(_mm512_sub_pd(z, _mm512_set1_pd(0.5 * kernel_width)));

        __m512i xy_ceili = _mm512_cvtpd_epi64(xy_ceil);
        __m512i z_ceili = _mm512_cvtpd_epi64(z_ceil);

        __m512d xy_i = _mm512_sub_pd(xy_ceil, xy);
        __m512d z_i = _mm512_sub_pd(z_ceil, z);

        // Normalized subgrid position for each point
        // [z_0, z_1, ...]
        __m512d xy_n = _mm512_add_pd(_mm512_add_pd(xy_i, xy_i), _mm512_set1_pd(kernel_width - 1.0));
        __m512d z_n = _mm512_add_pd(_mm512_add_pd(z_i, z_i), _mm512_set1_pd(kernel_width - 1.0));

        // Prepare duplicated registers to obtain coordinates
        __m512d x_d = _mm512_permutexvar_pd(_mm512_setr_epi64(0, 1, 2, 3, 0, 1, 2, 3), xy_n);
        __m512d y_d = _mm512_permutexvar_pd(_mm512_setr_epi64(4, 5, 6, 7, 4, 5, 6, 7), xy_n);
        __m512d z_d = _mm512_permutexvar_pd(_mm512_setr_epi64(0, 1, 2, 3, 0, 1, 2, 3), z_n);

        // Store integer coordinates for accumulation later (adjust indices to account for offset)
        // First 4 correspond to x indices, second 4 correspond to y indices.
        alignas(64) int64_t indices[12];
        _mm512_store_epi64(
            indices,
            _mm512_sub_epi64(
                xy_ceili,
                _mm512_inserti64x4(_mm512_set1_epi64(offset_x), _mm256_set1_epi64x(offset_y), 1)));
        _mm256_store_epi64(
            indices + 8,
            _mm256_sub_epi64(_mm512_castsi512_si256(z_ceili), _mm256_set1_epi64x(offset_z)));

        __m512d vx_lo, vx_hi;
        alignas(64) double vy[8], vz[8];

        // Unrolled loop to compute 4 points, one at a time
        compute_kernel(
            _mm512_permutex_pd(x_d, 0b0000'0000),
            _mm512_permutex_pd(y_d, 0b0000'0000),
            _mm512_permutex_pd(z_d, 0b0000'0000),
            strengths,
            vx_lo,
            vx_hi,
            vy,
            vz);
        accumulate_strengths(
            output, indices[0], indices[4], indices[8], stride_y, stride_z, vx_lo, vx_hi, vy, vz);

        compute_kernel(
            _mm512_permutex_pd(x_d, 0b0101'0101),
            _mm512_permutex_pd(y_d, 0b0101'0101),
            _mm512_permutex_pd(z_d, 0b0101'0101),
            strengths + 2,
            vx_lo,
            vx_hi,
            vy,
            vz);
        accumulate_strengths(
            output, indices[1], indices[5], indices[9], stride_y, stride_z, vx_lo, vx_hi, vy, vz);

        compute_kernel(
            _mm512_permutex_pd(x_d, 0b1010'1010),
            _mm512_permutex_pd(y_d, 0b1010'1010),
            _mm512_permutex_pd(z_d, 0b1010'1010),
            strengths + 4,
            vx_lo,
            vx_hi,
            vy,
            vz);
        accumulate_strengths(
            output, indices[2], indices[6], indices[10], stride_y, stride_z, vx_lo, vx_hi, vy, vz);

        compute_kernel(
            _mm512_permutex_pd(x_d, 0b1111'1111),
            _mm512_permutex_pd(y_d, 0b1111'1111),
            _mm512_permutex_pd(z_d, 0b1111'1111),
            strengths + 6,
            vx_lo,
            vx_hi,
            vy,
            vz);
        accumulate_strengths(
            output, indices[3], indices[7], indices[11], stride_y, stride_z, vx_lo, vx_hi, vy, vz);
    }

    void operator()(
        // Main loop of the spreading subproblem.
        // This loop is unrolled to process 4 points at a time.
        nu_point_collection<3, double const> const &input, subgrid_specification<3> const &grid,
        double *__restrict output) const {

        std::memset(output, 0, 2 * grid.num_elements() * sizeof(double));

        double const *coord_x = input.coordinates[0];
        double const *coord_y = input.coordinates[1];
        double const *coord_z = input.coordinates[2];
        double const *strengths = input.strengths;

        auto offset_x = grid.offsets[0];
        auto offset_y = grid.offsets[1];
        auto offset_z = grid.offsets[2];

        auto stride_y = 2 * grid.strides[1];
        auto stride_z = 2 * grid.strides[2];

        for (std::size_t i = 0; i < input.num_points; i += 4) {
            process_4(
                output,
                coord_x + i,
                coord_y + i,
                coord_z + i,
                strengths + 2 * i,
                offset_x,
                offset_y,
                offset_z,
                stride_y,
                stride_z);
        }
    }

    std::size_t num_points_multiple() const { return 4; }
    std::array<std::size_t, 3> extent_multiple() const { return {8, 1, 1}; }
    std::array<KernelWriteSpec<double>, 3> target_padding() const {
        // Write up to additional half width of vector
        // Due to alignment
        double ns2 = 0.5 * kernel_width;
        return {
            KernelWriteSpec<double>{ns2, 0, 12},
            {ns2, 0, static_cast<int>(writeout_width)},
            {ns2, 0, static_cast<int>(writeout_width)}};
    }
};

extern template struct SpreadSubproblemPoly3DW8F64<4>;
extern template struct SpreadSubproblemPoly3DW8F64<5>;
extern template struct SpreadSubproblemPoly3DW8F64<6>;
extern template struct SpreadSubproblemPoly3DW8F64<7>;
extern template struct SpreadSubproblemPoly3DW8F64<8>;
extern template struct SpreadSubproblemPoly3DW8F64<9>;
extern template struct SpreadSubproblemPoly3DW8F64<10>;
extern template struct SpreadSubproblemPoly3DW8F64<11>;

} // namespace avx512
} // namespace spreading
} // namespace finufft
