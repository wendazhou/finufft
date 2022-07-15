#include "spread_reference.h"
#include "spread_subproblem_reference.h"

#include "gather_fold_reference.h"
#include "spread_subproblem_reference.h"
#include "synchronized_accumulate_reference.h"

#include "../legacy/synchronized_accumulate_legacy.h"

namespace finufft {
namespace spreading {

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

#define INSTANTIATE(T, Dim)                                                                        \
    template SpreadFunctorConfiguration<T, Dim> get_spread_configuration_reference(                \
        finufft_spread_opts const &opts);                                                          \
    template SpreadSubproblemFunctor<T, Dim> get_subproblem_polynomial_reference_functor<T, Dim>(  \
        kernel_specification const &);

INSTANTIATE(float, 1);
INSTANTIATE(float, 2);
INSTANTIATE(float, 3);

INSTANTIATE(double, 1);
INSTANTIATE(double, 2);
INSTANTIATE(double, 3);

#undef INSTANTIATE

} // namespace spreading

} // namespace finufft
