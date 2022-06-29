#pragma once

#include <stdexcept>

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

/** Utility generic function to obtain AVX-512 implementation of the subproblem.
 * 
 * Note that this function template dispatches to supported implementations,
 * and throws a std::runtime_error if no implementation is found.
 * 
 */
template <typename T, std::size_t Dim>
SpreadSubproblemFunctor<T, Dim>
get_subproblem_polynomial_avx512_functor(kernel_specification const &kernel) {
    throw std::runtime_error("No AVX-512 implementation found for given kernel.");
}

template <>
SpreadSubproblemFunctor<float, 1>
get_subproblem_polynomial_avx512_functor<float, 1>(kernel_specification const &kernel) {
    return get_subproblem_polynomial_avx512_1d_fp32_functor(kernel);
}

} // namespace spreading
} // namespace finufft
