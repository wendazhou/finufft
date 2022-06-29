#include "spreading_reference.h"

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

} // namespace spreading

} // namespace finufft
