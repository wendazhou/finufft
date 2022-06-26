#pragma once

#include "../../spreading.h"

namespace finufft {

namespace spreading {

void gather_and_fold_avx512(
    SpreaderMemoryInput<1, float> const &memory, nu_point_collection<1, float const *> const &input,
    std::array<int64_t, 1> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range);

void gather_and_fold_avx512(
    SpreaderMemoryInput<2, float> const &memory, nu_point_collection<2, float const *> const &input,
    std::array<int64_t, 2> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range);

void gather_and_fold_avx512(
    SpreaderMemoryInput<3, float> const &memory, nu_point_collection<3, float const *> const &input,
    std::array<int64_t, 3> const &sizes, std::int64_t const *sort_indices,
    FoldRescaleRange rescale_range);


} // namespace spreading

} // namespace finufft
