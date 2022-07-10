#pragma once

/** @file
 * 
 * Implementation of fold-rescale with AVX512 intrinsics.
 * 
 */

#include "../reference/gather_fold_reference.h"
#include <immintrin.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

namespace finufft {
namespace spreading {
namespace avx512 {

template <typename T> struct FoldRescaleIdentityAvx512;

template <typename T> struct FoldRescalePiAvx512;

template <> struct FoldRescaleIdentityAvx512<float> : FoldRescaleIdentity<float> {
    using FoldRescaleIdentity<float>::operator();

    void operator()(__m512 &v, float extent) const {
        __m512 extent_v = _mm512_set1_ps(extent);

        // Branchless folding operation

        // Compute masks indicating location of element (to the left or to the right)
        auto mask_smaller = _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_LT_OQ);
        auto mask_larger = _mm512_cmp_ps_mask(v, extent_v, _CMP_GT_OQ);

        // Compute shifted versions of the input data
        auto one_left = _mm512_sub_ps(v, extent_v);
        auto one_right = _mm512_add_ps(v, extent_v);

        // Select final result by blending between input and shifted version
        // based on result of comparison
        v = _mm512_mask_blend_ps(mask_smaller, v, one_right);
        v = _mm512_mask_blend_ps(mask_larger, v, one_left);
    }
};

template <> struct FoldRescaleIdentityAvx512<double> : FoldRescaleIdentity<double> {
    using FoldRescaleIdentity<double>::operator();

    void operator()(__m512d &v, double extent) {
        __m512d extent_v = _mm512_set1_pd(extent);

        // Branchless folding operation

        // Compute masks indicating location of element (to the left or to the right)
        auto mask_smaller = _mm512_cmp_pd_mask(v, _mm512_setzero_pd(), _CMP_LT_OQ);
        auto mask_larger = _mm512_cmp_pd_mask(v, extent_v, _CMP_GT_OQ);

        // Compute shifted versions of the input data
        auto one_left = _mm512_sub_pd(v, extent_v);
        auto one_right = _mm512_add_pd(v, extent_v);

        // Select final result by blending between input and shifted version
        // based on result of comparison
        v = _mm512_mask_blend_pd(mask_smaller, v, one_right);
        v = _mm512_mask_blend_pd(mask_larger, v, one_left);
    }
};

template <> struct FoldRescalePiAvx512<float> : FoldRescalePi<float> {
    using FoldRescalePi<float>::operator();

    void operator()(__m512 &v, float extent) const {
        __m512 extent_v = _mm512_set1_ps(extent);

        __m512 pi = _mm512_set1_ps(M_PI);
        __m512 n_pi = _mm512_set1_ps(-M_PI);
        __m512 two_pi = _mm512_set1_ps(2 * M_PI);

        auto mask_smaller = _mm512_cmp_ps_mask(v, n_pi, _CMP_LT_OQ);
        auto mask_larger = _mm512_cmp_ps_mask(v, pi, _CMP_GT_OQ);

        auto one_left = _mm512_sub_ps(v, two_pi);
        auto one_right = _mm512_add_ps(v, two_pi);

        v = _mm512_mask_blend_ps(mask_smaller, v, one_right);
        v = _mm512_mask_blend_ps(mask_larger, v, one_left);

        v = _mm512_add_ps(v, pi);
        v = _mm512_mul_ps(v, extent_v);
        v = _mm512_mul_ps(v, _mm512_set1_ps(0.5 * M_1_PI));
    }
};

template <> struct FoldRescalePiAvx512<double> : FoldRescalePi<double> {
    using FoldRescalePi<double>::operator();

    void operator()(__m512d &v, double extent) const {
        __m512d extent_v = _mm512_set1_pd(extent);

        __m512d pi = _mm512_set1_pd(M_PI);
        __m512d n_pi = _mm512_set1_pd(-M_PI);
        __m512d two_pi = _mm512_set1_pd(2 * M_PI);

        auto mask_smaller = _mm512_cmp_pd_mask(v, n_pi, _CMP_LT_OQ);
        auto mask_larger = _mm512_cmp_pd_mask(v, pi, _CMP_GT_OQ);

        auto one_left = _mm512_sub_pd(v, two_pi);
        auto one_right = _mm512_add_pd(v, two_pi);

        v = _mm512_mask_blend_pd(mask_smaller, v, one_right);
        v = _mm512_mask_blend_pd(mask_larger, v, one_left);

        v = _mm512_add_pd(v, pi);
        v = _mm512_mul_pd(v, extent_v);
        v = _mm512_mul_pd(v, _mm512_set1_pd(0.5 * M_1_PI));
    }
};

} // namespace avx512
} // namespace spreading
} // namespace finufft
