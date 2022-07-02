#pragma once

#include <finufft/defs.h>

#include "kernels/legacy/spreading_legacy.h"
#include "kernels/legacy/synchronized_accumulate_legacy.h"
#include "kernels/reference/gather_fold_reference.h"
#include "kernels/reference/spread_processor_reference.h"
#include "spreading.h"

namespace finufft {
namespace spreading {

template <std::size_t Dim, typename T, typename IdxT>
inline void spread(
    IdxT const *sort_indices, std::array<std::int64_t, Dim> const &sizes, std::size_t num_points,
    std::array<T const *, Dim> const &coordinates, T const *strengths, T *output,
    const finufft_spread_opts &opts) {

    OmpSpreadProcessor processor{opts.nthreads, static_cast<std::size_t>(opts.max_subproblem_size)};

    auto accumulate_subgrid_factory = get_legacy_locking_accumulator<T, Dim>();
    kernel_specification kernel_spec{opts.ES_beta, opts.nspread};
    auto spread_subproblem = SpreadSubproblemLegacyFunctor{kernel_spec};
    auto gather_rescale = GatherAndFoldReferenceFunctor{
        opts.pirange ? FoldRescaleRange::Pi : FoldRescaleRange::Identity};

    auto config = SpreadFunctorConfiguration<T, Dim>{
        std::move(gather_rescale),
        std::move(spread_subproblem),
        std::move(accumulate_subgrid_factory),
    };

    nu_point_collection<Dim, const T> input{num_points, coordinates, strengths};

    processor(config, input, sort_indices, sizes, output);
}

} // namespace spreading
} // namespace finufft
