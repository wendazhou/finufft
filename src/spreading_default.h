#pragma once

#include <finufft/defs.h>

#include "kernels/legacy/spread_legacy.h"
#include "kernels/reference/spread_processor_reference.h"

#include "kernels/avx512/spread_avx512.h"

namespace finufft {
namespace spreading {

/** This is the main entry point for the spreading subproblem as currently
 * configured by default for use in finufft. It makes use of the legacy configuration,
 * although other configurations can be selected by editing the code.
 *
 * TODO: API in planning to select correct configuration at runtime.
 *
 */
template <std::size_t Dim, typename T, typename IdxT>
inline void spread(
    IdxT const *sort_indices, std::array<std::int64_t, Dim> const &sizes, std::size_t num_points,
    std::array<T const *, Dim> const &coordinates, T const *strengths, T *output,
    const finufft_spread_opts &opts) {

    OmpSpreadProcessor processor{opts.nthreads, static_cast<std::size_t>(opts.max_subproblem_size)};
    auto config = get_spread_configuration_legacy<T, Dim>(opts);
    // auto config = get_spread_configuration_avx512<T, Dim>(opts);

    nu_point_collection<Dim, const T> input{num_points, coordinates, strengths};

    processor(config, input, sort_indices, sizes, output);
}

} // namespace spreading
} // namespace finufft
