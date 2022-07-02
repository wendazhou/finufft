#include "spread_reference.h"
#include "spread_subproblem_reference.h"

namespace finufft {
namespace spreading {

// Explicit instantiation of common functor types and dimensions.
template SpreadSubproblemFunctor<float, 1>
get_subproblem_polynomial_reference_functor<float, 1>(kernel_specification const &);
template SpreadSubproblemFunctor<float, 2>
get_subproblem_polynomial_reference_functor<float, 2>(kernel_specification const &);
template SpreadSubproblemFunctor<float, 3>
get_subproblem_polynomial_reference_functor<float, 3>(kernel_specification const &);
template SpreadSubproblemFunctor<double, 1>
get_subproblem_polynomial_reference_functor<double, 1>(kernel_specification const &);
template SpreadSubproblemFunctor<double, 2>
get_subproblem_polynomial_reference_functor<double, 2>(kernel_specification const &);
template SpreadSubproblemFunctor<double, 3>
get_subproblem_polynomial_reference_functor<double, 3>(kernel_specification const &);

template SpreadFunctorConfiguration<float, 1>
get_spread_configuration_reference<float, 1>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<float, 2>
get_spread_configuration_reference<float, 2>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<float, 3>
get_spread_configuration_reference<float, 3>(finufft_spread_opts const &);

template SpreadFunctorConfiguration<double, 1>
get_spread_configuration_reference<double, 1>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<double, 2>
get_spread_configuration_reference<double, 2>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<double, 3>
get_spread_configuration_reference<double, 3>(finufft_spread_opts const &);

} // namespace spreading

} // namespace finufft
