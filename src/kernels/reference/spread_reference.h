#pragma once

#include "../../spreading.h"
#include <finufft_spread_opts.h>

#include "../legacy/synchronized_accumulate_legacy.h"
#include "gather_fold_reference.h"
#include "spread_subproblem_reference.h"

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

    // Note: no reference implementation for accumulate yet, using
    // legacy implementation for now.
    auto accumulate_subgrid_factory = get_legacy_locking_accumulator<T, Dim>();

    return SpreadFunctorConfiguration<T, Dim>{
        std::move(gather_rescale),
        std::move(spread_subproblem),
        std::move(accumulate_subgrid_factory),
    };
}

// Explicit instantiation for commonly used settings.
extern template SpreadFunctorConfiguration<float, 1>
get_spread_configuration_reference<float, 1>(finufft_spread_opts const &);
extern template SpreadFunctorConfiguration<float, 2>
get_spread_configuration_reference<float, 2>(finufft_spread_opts const &);
extern template SpreadFunctorConfiguration<float, 3>
get_spread_configuration_reference<float, 3>(finufft_spread_opts const &);

extern template SpreadFunctorConfiguration<double, 1>
get_spread_configuration_reference<double, 1>(finufft_spread_opts const &);
extern template SpreadFunctorConfiguration<double, 2>
get_spread_configuration_reference<double, 2>(finufft_spread_opts const &);
extern template SpreadFunctorConfiguration<double, 3>
get_spread_configuration_reference<double, 3>(finufft_spread_opts const &);

} // namespace spreading
} // namespace finufft
