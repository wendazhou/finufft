#pragma once

#include <finufft/defs.h>

#include "kernels/legacy/spread_legacy.h"
#include "kernels/reference/spread_processor_reference.h"

namespace finufft {
namespace spreading {

template <std::size_t Dim, typename T, typename IdxT>
inline void spread(
    IdxT const *sort_indices, std::array<std::int64_t, Dim> const &sizes, std::size_t num_points,
    std::array<T const *, Dim> const &coordinates, T const *strengths, T *output,
    const finufft_spread_opts &opts) {

    OmpSpreadProcessor processor{opts.nthreads, static_cast<std::size_t>(opts.max_subproblem_size)};
    auto config = get_legacy_spread_configuration<T, Dim>(opts);

    nu_point_collection<Dim, const T> input{num_points, coordinates, strengths};

    processor(config, input, sort_indices, sizes, output);
}

} // namespace spreading
} // namespace finufft
