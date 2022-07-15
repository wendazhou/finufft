#pragma once

#include <stdexcept>

#include "../spreading.h"
#include "../../tracing.h"

struct finufft_spread_opts;

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

SpreadSubproblemFunctor<double, 1>
get_subproblem_polynomial_avx512_1d_fp64_functor(kernel_specification const &kernel);

SpreadSubproblemFunctor<float, 2>
get_subproblem_polynomial_avx512_2d_fp32_functor(kernel_specification const &kernel);

SpreadSubproblemFunctor<double, 2>
get_subproblem_polynomial_avx512_2d_fp64_functor(kernel_specification const &kernel);

SpreadSubproblemFunctor<float, 3>
get_subproblem_polynomial_avx512_3d_fp32_functor(kernel_specification const &kernel);

SpreadSubproblemFunctor<double, 3>
get_subproblem_polynomial_avx512_3d_fp64_functor(kernel_specification const &kernel);

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
inline SpreadSubproblemFunctor<float, 1>
get_subproblem_polynomial_avx512_functor<float, 1>(kernel_specification const &kernel) {
    return get_subproblem_polynomial_avx512_1d_fp32_functor(kernel);
}

template <>
inline SpreadSubproblemFunctor<float, 2>
get_subproblem_polynomial_avx512_functor<float, 2>(kernel_specification const &kernel) {
    return get_subproblem_polynomial_avx512_2d_fp32_functor(kernel);
}

template <>
inline SpreadSubproblemFunctor<float, 3>
get_subproblem_polynomial_avx512_functor<float, 3>(kernel_specification const &kernel) {
    return get_subproblem_polynomial_avx512_3d_fp32_functor(kernel);
}

template <>
inline SpreadSubproblemFunctor<double, 1>
get_subproblem_polynomial_avx512_functor<double, 1>(kernel_specification const &kernel) {
    return get_subproblem_polynomial_avx512_1d_fp64_functor(kernel);
}

template <>
inline SpreadSubproblemFunctor<double, 2>
get_subproblem_polynomial_avx512_functor<double, 2>(kernel_specification const &kernel) {
    return get_subproblem_polynomial_avx512_2d_fp64_functor(kernel);
}


template <>
inline SpreadSubproblemFunctor<double, 3>
get_subproblem_polynomial_avx512_functor<double, 3>(kernel_specification const &kernel) {
    return get_subproblem_polynomial_avx512_3d_fp64_functor(kernel);
}

/** Get spread configuration for AVX-512 implementation.
 * 
 */
template<typename T, std::size_t Dim>
SpreadFunctorConfiguration<T, Dim> get_spread_configuration_avx512(finufft_spread_opts const&);

/** Create a spread functor based on a packed sort and blocked spread strategy.
 * 
 * This spread functor is based on a packed sorting strategy, into blocks
 * whose size is computed based on the L1 cache size of the CPU.
 * 
 */
template<typename T, std::size_t Dim>
SpreadFunctor<T, Dim> make_avx512_blocked_spread_functor(
    kernel_specification const& kernel,
    tcb::span<const std::size_t, Dim> target_size,
    FoldRescaleRange input_range,
    finufft::Timer const& timer = {});


} // namespace spreading
} // namespace finufft
