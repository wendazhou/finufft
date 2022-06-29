#pragma once

#include "../../spreading.h"

namespace finufft {
namespace spreading {
/** Obtain an implementation of the subproblem for the given kernel specification.
 *
 * This function searches through the pre-computed polynomial approximations,
 * and instantiates a kernel using a pre-computed polynomial if it exists and is supported.
 * Otherwise, a std::runtime_error is thrown.
 *
 */
SpreadSubproblemFunctor<float, 1>
get_subproblem_polynomial_avx512_1d_fp32_functor(kernel_specification const &kernel);

} // namespace spreading
} // namespace finufft
