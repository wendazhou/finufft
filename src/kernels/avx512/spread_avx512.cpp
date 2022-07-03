#include "spread_avx512.h"

#include "../legacy/synchronized_accumulate_legacy.h"
#include "gather_fold_avx512.h"

#include <finufft_spread_opts.h>

namespace finufft {
namespace spreading {

template <typename T, std::size_t Dim>
SpreadFunctorConfiguration<T, Dim>
get_spread_configuration_avx512(finufft_spread_opts const &opts) {
    GatherFoldAvx512Functor gather_rescale{
        opts.pirange ? FoldRescaleRange::Pi : FoldRescaleRange::Identity};
    auto spread_supbroblem =
        get_subproblem_polynomial_avx512_functor<T, Dim>({opts.ES_beta, opts.nspread});
    auto accumulate_subgrid_factory = get_legacy_atomic_accumulator<T, Dim>();

    return SpreadFunctorConfiguration<T, Dim>{
        std::move(gather_rescale),
        std::move(spread_supbroblem),
        std::move(accumulate_subgrid_factory),
    };
}

template SpreadFunctorConfiguration<float, 1>
get_spread_configuration_avx512<float, 1>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<float, 2>
get_spread_configuration_avx512<float, 2>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<float, 3>
get_spread_configuration_avx512<float, 3>(finufft_spread_opts const &);

template SpreadFunctorConfiguration<double, 1>
get_spread_configuration_avx512<double, 1>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<double, 2>
get_spread_configuration_avx512<double, 2>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<double, 3>
get_spread_configuration_avx512<double, 3>(finufft_spread_opts const &);

} // namespace spreading
} // namespace finufft
