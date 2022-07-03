#pragma once

#include "../../spreading.h"
#include <finufft_spread_opts.h>

#include "../reference/gather_fold_reference.h"
#include "spread_subproblem_legacy.h"
#include "synchronized_accumulate_legacy.h"

namespace finufft {
namespace spreading {

/** Get default configuration which mimics current functionality.
 *
 */
template <typename T, std::size_t Dim>
SpreadFunctorConfiguration<T, Dim>
get_spread_configuration_legacy(finufft_spread_opts const &opts) {
    GatherAndFoldReferenceFunctor gather_rescale{
        opts.pirange ? FoldRescaleRange::Pi : FoldRescaleRange::Identity};

    kernel_specification kernel_spec{opts.ES_beta, opts.nspread};
    SpreadSubproblemLegacyFunctor<T, Dim> spread_subproblem{kernel_spec};

//     auto accumulate_subgrid_factory = (opts.nthreads > opts.atomic_threshold || opts.nthreads == 0)
//                                           ? get_legacy_atomic_accumulator<T, Dim>()
//                                           : get_legacy_locking_accumulator<T, Dim>();
// 
    auto accumulate_subgrid_factory = get_legacy_atomic_accumulator<T, Dim>();

    return SpreadFunctorConfiguration<T, Dim>{
        std::move(gather_rescale),
        std::move(spread_subproblem),
        std::move(accumulate_subgrid_factory),
    };
}

} // namespace spreading
} // namespace finufft
