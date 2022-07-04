#include "spread_legacy.h"

#include "../reference/gather_fold_reference.h"
#include "spread_subproblem_legacy.h"
#include "synchronized_accumulate_legacy.h"

namespace finufft {
namespace spreading {

template <typename T, std::size_t Dim>
SpreadFunctorConfiguration<T, Dim>
get_spread_configuration_legacy(finufft_spread_opts const &opts) {
    GatherAndFoldReferenceFunctor gather_rescale{
        opts.pirange ? FoldRescaleRange::Pi : FoldRescaleRange::Identity};

    kernel_specification kernel_spec{opts.ES_beta, opts.nspread};
    SpreadSubproblemLegacyFunctor<T, Dim> spread_subproblem{kernel_spec};

    auto accumulate_subgrid_factory = (opts.nthreads > opts.atomic_threshold || opts.nthreads == 0)
                                          ? get_legacy_atomic_accumulator<T, Dim>()
                                          : get_legacy_locking_accumulator<T, Dim>();

    return SpreadFunctorConfiguration<T, Dim>{
        std::move(gather_rescale),
        std::move(spread_subproblem),
        std::move(accumulate_subgrid_factory),
    };
}


template SpreadFunctorConfiguration<float, 1> get_spread_configuration_legacy<float, 1>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<float, 2> get_spread_configuration_legacy<float, 2>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<float, 3> get_spread_configuration_legacy<float, 3>(finufft_spread_opts const &);

template SpreadFunctorConfiguration<double, 1> get_spread_configuration_legacy<double, 1>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<double, 2> get_spread_configuration_legacy<double, 2>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<double, 3> get_spread_configuration_legacy<double, 3>(finufft_spread_opts const &);

} // namespace spreading
} // namespace finufft
