#pragma once

#include "../../spreading.h"

namespace finufft {

namespace spreading {

/** Main functor implementing AVX-512 gather and fold.
 *
 */
struct GatherFoldAvx512 {
    void operator()(
        SpreaderMemoryInput<1, float> const &memory,
        nu_point_collection<1, float const> const &input, std::array<int64_t, 1> const &sizes,
        std::int64_t const *sort_indices, FoldRescaleRange rescale_range) const;

    void operator()(
        SpreaderMemoryInput<2, float> const &memory,
        nu_point_collection<2, float const> const &input, std::array<int64_t, 2> const &sizes,
        std::int64_t const *sort_indices, FoldRescaleRange rescale_range) const;

    void operator()(
        SpreaderMemoryInput<3, float> const &memory,
        nu_point_collection<3, float const> const &input, std::array<int64_t, 3> const &sizes,
        std::int64_t const *sort_indices, FoldRescaleRange rescale_range) const;
};

/** Function object which encapsulates the AVX-512 gather and fold implementation.
 *
 * This function encapsulates the implementation of the AVX-512 gather and fold for
 * dimension 1, 2, 3 in single and double precision.
 *
 */
extern const GatherFoldAvx512 gather_and_fold_avx512;

} // namespace spreading

} // namespace finufft
