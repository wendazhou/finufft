#include "gather_fold_avx2.h"

#include <cmath>
#include <immintrin.h>

namespace finufft {

namespace spreading {

namespace {

/** Branchless implementation of folding.
 * 
 * Implements folding by ensuring that data one period to the right
 * or to the left of the main period is folded back.
 * 
 * Our implementation is branchless to enable vectorization, and
 * computes both the left-shifted and right-shifted versions of the
 * input data.
 * Those versions are then masked according to whether they are
 * to the left, within, or to the right of the main period, with
 * only the correct version being not zeroed.
 * The final output is assembled by summing over all the versions
 * after they have been masked.
 * 
 */
struct FoldRescaleIdentityAvx2Float : FoldRescaleIdentity<float> {
    using FoldRescaleIdentity<float>::operator();

    void operator()(__m256 &v, __m256 const &extent) const {
        // Compute masks to determine whether folding is needed
        auto mask_smaller = _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_LT_OQ);
        auto mask_larger = _mm256_cmp_ps(v, extent, _CMP_GT_OQ);

        // Compute offset versions of the input data
        auto one_left = _mm256_sub_ps(v, extent);
        auto one_right = _mm256_add_ps(v, extent);

        // Mask shifted versions according to comparison
        one_left = _mm256_and_ps(one_left, mask_larger);
        one_right = _mm256_and_ps(one_right, mask_smaller);

        // Mask original versions according to comparison
        v = _mm256_andnot_ps(mask_smaller, v);
        v = _mm256_andnot_ps(mask_larger, v);

        // Assemble output by summing over all versions
        v = _mm256_add_ps(v, one_right);
        v = _mm256_add_ps(v, one_left);
    }
};

struct FoldRescalePiAvx2Float : FoldRescalePi<float> {
    using FoldRescalePi<float>::operator();

    void operator()(__m256 &v, __m256 const &extent) const {
        __m256 pi = _mm256_set1_ps(M_PI);
        __m256 n_pi = _mm256_set1_ps(-M_PI);
        __m256 two_pi = _mm256_set1_ps(M_2PI);

        auto mask_smaller = _mm256_cmp_ps(v, n_pi, _CMP_LT_OQ);
        auto mask_larger = _mm256_cmp_ps(v, pi, _CMP_GT_OQ);

        auto one_left = _mm256_sub_ps(v, two_pi);
        auto one_right = _mm256_add_ps(v, two_pi);

        one_left = _mm256_and_ps(one_left, mask_larger);
        one_right = _mm256_and_ps(one_right, mask_smaller);

        v = _mm256_andnot_ps(mask_smaller, v);
        v = _mm256_andnot_ps(mask_larger, v);

        v = _mm256_add_ps(v, one_right);
        v = _mm256_add_ps(v, one_left);

        v = _mm256_add_ps(v, pi);
        v = _mm256_mul_ps(v, extent);
        v = _mm256_mul_ps(v, _mm256_set1_ps(M_1_2PI));
    }
};

template <std::size_t Dim, typename FoldRescale>
void gather_and_fold_avx2_rescale(
    SpreaderMemoryInput<Dim, float> const &memory,
    nu_point_collection<Dim, float const *> const &input, std::array<int64_t, Dim> const &sizes,
    std::int64_t const *sort_indices, FoldRescale &&fold_rescale) {

    std::array<float, Dim> sizes_floating;
    std::copy(sizes.begin(), sizes.end(), sizes_floating.begin());

    std::size_t i = 0;

    for (; i < memory.num_points - 7; i += 8) {
        auto addr1 = _mm256_load_si256(reinterpret_cast<__m256i const *>(sort_indices + i));
        auto addr2 = _mm256_load_si256(reinterpret_cast<__m256i const *>(sort_indices + i + 4));

        for (std::size_t dim = 0; dim < Dim; ++dim) {
            __m128 v1 = _mm256_i64gather_ps(input.coordinates[dim], addr1, sizeof(float));
            __m128 v2 = _mm256_i64gather_ps(input.coordinates[dim], addr2, sizeof(float));
            __m256 v = _mm256_insertf128_ps(_mm256_castps128_ps256(v1), v2, 1);

            __m256 range_max = _mm256_set1_ps(sizes_floating[dim]);
            fold_rescale(v, range_max);
            _mm256_store_ps(memory.coordinates[dim] + i, v);
        }

        // Load strengths which are complex32 as a single 64-bit element.
        auto strengths1 = _mm256_i64gather_epi64(
            reinterpret_cast<long long const *>(input.strengths), addr1, sizeof(double));
        auto strengths2 = _mm256_i64gather_epi64(
            reinterpret_cast<long long const *>(input.strengths), addr2, sizeof(double));
        _mm256_store_si256(reinterpret_cast<__m256i*>(memory.strengths + 2 * i), strengths1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(memory.strengths + 2 * i + 8), strengths2);
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

void GatherFoldAvx2::operator()(
    SpreaderMemoryInput<1, float> const &memory, nu_point_collection<1, float const *> const &input,
    std::array<int64_t, 1> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) const {

    if (rescale_range == FoldRescaleRange::Identity) {
        gather_and_fold_avx2_rescale(
            memory, input, sizes, sort_indices, FoldRescaleIdentityAvx2Float{});
    } else {
        gather_and_fold_avx2_rescale(memory, input, sizes, sort_indices, FoldRescalePiAvx2Float{});
    }
}

void GatherFoldAvx2::operator()(
    SpreaderMemoryInput<2, float> const &memory, nu_point_collection<2, float const *> const &input,
    std::array<int64_t, 2> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) const {

    if (rescale_range == FoldRescaleRange::Identity) {
        gather_and_fold_avx2_rescale(
            memory, input, sizes, sort_indices, FoldRescaleIdentityAvx2Float{});
    } else {
        gather_and_fold_avx2_rescale(memory, input, sizes, sort_indices, FoldRescalePiAvx2Float{});
    }
}

void GatherFoldAvx2::operator()(
    SpreaderMemoryInput<3, float> const &memory, nu_point_collection<3, float const *> const &input,
    std::array<int64_t, 3> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range) const {

    if (rescale_range == FoldRescaleRange::Identity) {
        gather_and_fold_avx2_rescale(
            memory, input, sizes, sort_indices, FoldRescaleIdentityAvx2Float{});
    } else {
        gather_and_fold_avx2_rescale(memory, input, sizes, sort_indices, FoldRescalePiAvx2Float{});
    }
}

const GatherFoldAvx2 gather_and_fold_avx2;

} // namespace spreading

} // namespace finufft
