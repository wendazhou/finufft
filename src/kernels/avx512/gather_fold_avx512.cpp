#include "gather_fold_avx512.h"

#include <cmath>
#include <immintrin.h>

namespace finufft {
namespace spreading {

namespace {

struct FoldRescaleIdentityAvx512Float : FoldRescaleIdentity<float> {
    using FoldRescaleIdentity<float>::operator();

    void operator()(__m512 &v, __m512 const &extent) const {
        // Branchless folding operation

        // Compute masks indicating location of element (to the left or to the right)
        auto mask_smaller = _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_LT_OQ);
        auto mask_larger = _mm512_cmp_ps_mask(v, extent, _CMP_GT_OQ);

        // Compute shifted versions of the input data
        auto one_left = _mm512_sub_ps(v, extent);
        auto one_right = _mm512_add_ps(v, extent);

        // Select final result by blending between input and shifted version
        // based on result of comparison
        v = _mm512_mask_blend_ps(mask_smaller, v, one_right);
        v = _mm512_mask_blend_ps(mask_larger, v, one_left);
    }
};

struct FoldRescalePiAvx512Float : FoldRescalePi<float> {
    using FoldRescalePi<float>::operator();

    void operator()(__m512 &v, __m512 const &extent) const {
        __m512 pi = _mm512_set1_ps(M_PI);
        __m512 n_pi = _mm512_set1_ps(-M_PI);
        __m512 two_pi = _mm512_set1_ps(M_2PI);

        auto mask_smaller = _mm512_cmp_ps_mask(v, n_pi, _CMP_LT_OQ);
        auto mask_larger = _mm512_cmp_ps_mask(v, pi, _CMP_GT_OQ);

        auto one_left = _mm512_sub_ps(v, two_pi);
        auto one_right = _mm512_add_ps(v, two_pi);

        v = _mm512_mask_blend_ps(mask_smaller, v, one_right);
        v = _mm512_mask_blend_ps(mask_larger, v, one_left);

        v = _mm512_add_ps(v, pi);
        v = _mm512_mul_ps(v, extent);
        v = _mm512_mul_ps(v, _mm512_set1_ps(M_1_2PI));
    }
};

/** Generic vectorized implementation for gathering and folding.
 * 
 * This version operates on a set of points corresponding to the vector width at a time.
 * 
 */
template <std::size_t Dim, typename FoldRescale>
void gather_and_fold_avx512_rescale(
    SpreaderMemoryInput<Dim, float> const &memory,
    nu_point_collection<Dim, float const *> const &input, std::array<int64_t, Dim> const &sizes,
    std::int64_t const *sort_indices, FoldRescale &&fold_rescale) {

    std::array<float, Dim> sizes_floating;
    std::copy(sizes.begin(), sizes.end(), sizes_floating.begin());

    std::size_t i = 0;

    for (; i < memory.num_points - 15; i += 16) {
        // We are using 64-bit indices for 32-bit data.
        // Need to load two registers of indices to cover one register of data.
        auto addr1 = _mm512_load_epi64(sort_indices + i);
        auto addr2 = _mm512_load_epi64(sort_indices + i + 8);

        for (std::size_t dim = 0; dim < Dim; ++dim) {
            // Load values from memory, and combine into single 16-wide register.
            __m256 v1 = _mm512_i64gather_ps(addr1, input.coordinates[dim], sizeof(float));
            __m256 v2 = _mm512_i64gather_ps(addr2, input.coordinates[dim], sizeof(float));
            __m512 v = _mm512_insertf32x8(_mm512_castps256_ps512(v1), v2, 1);

            // Perform folding and store result
            __m512 range_max = _mm512_set1_ps(sizes_floating[dim]);
            fold_rescale(v, range_max);
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

} // namespace

void GatherFoldAvx512::operator()(
    SpreaderMemoryInput<1, float> const &memory, nu_point_collection<1, float const *> const &input,
    std::array<int64_t, 1> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) const {

    if (rescale_range == FoldRescaleRange::Identity) {
        gather_and_fold_avx512_rescale(
            memory, input, sizes, sort_indices, FoldRescaleIdentityAvx512Float{});
    } else {
        gather_and_fold_avx512_rescale(
            memory, input, sizes, sort_indices, FoldRescalePiAvx512Float{});
    }
}

void GatherFoldAvx512::operator()(
    SpreaderMemoryInput<2, float> const &memory, nu_point_collection<2, float const *> const &input,
    std::array<int64_t, 2> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) const {

    if (rescale_range == FoldRescaleRange::Identity) {
        gather_and_fold_avx512_rescale(
            memory, input, sizes, sort_indices, FoldRescaleIdentityAvx512Float{});
    } else {
        gather_and_fold_avx512_rescale(
            memory, input, sizes, sort_indices, FoldRescalePiAvx512Float{});
    }
}

void GatherFoldAvx512::operator()(
    SpreaderMemoryInput<3, float> const &memory, nu_point_collection<3, float const *> const &input,
    std::array<int64_t, 3> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) const {

    if (rescale_range == FoldRescaleRange::Identity) {
        gather_and_fold_avx512_rescale(
            memory, input, sizes, sort_indices, FoldRescaleIdentityAvx512Float{});
    } else {
        gather_and_fold_avx512_rescale(
            memory, input, sizes, sort_indices, FoldRescalePiAvx512Float{});
    }
}

const GatherFoldAvx512 gather_and_fold_avx512;

} // namespace spreading

} // namespace finufft
