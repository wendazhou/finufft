#include "spread_reference.h"
#include "spread_subproblem_reference.h"

#include "gather_fold_reference.h"
#include "spread_subproblem_reference.h"
#include "synchronized_accumulate_reference.h"

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

/** Get default configuration based on reference implementation.
 *
 */
template <typename T, std::size_t Dim>
SpreadFunctorConfiguration<T, Dim>
get_spread_configuration_reference(finufft_spread_opts const &opts) {
    GatherAndFoldReferenceFunctor gather_rescale{
        opts.pirange ? FoldRescaleRange::Pi : FoldRescaleRange::Identity};

    kernel_specification kernel_spec{opts.ES_beta, opts.nspread};
    auto spread_subproblem = get_subproblem_polynomial_reference_functor<T, Dim>(kernel_spec);

    // Use block locking accumulator.
    // Locking overhead should be minimal when number of threads is low, so
    // don't bother special casing a non-synchronized version for single-threaded problem.
    auto accumulate_subgrid_factory = get_reference_block_locking_accumulator<T, Dim>();

    return SpreadFunctorConfiguration<T, Dim>{
        std::move(gather_rescale),
        std::move(spread_subproblem),
        std::move(accumulate_subgrid_factory),
    };
}

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
