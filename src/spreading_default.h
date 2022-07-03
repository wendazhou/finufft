#pragma once

#include <finufft/defs.h>

#include "kernels/legacy/spread_legacy.h"
#include "kernels/reference/spread_processor_reference.h"

#include "kernels/avx512/spread_avx512.h"

namespace finufft {
namespace spreading {

template <std::size_t Dim, typename T, typename IdxT, typename Configuration>
inline void spread_with_configuration(
    IdxT const *sort_indices,
    nu_point_collection<Dim, typename identity<T>::type const> const &input,
    std::array<std::int64_t, Dim> const &sizes, T *output, const finufft_spread_opts &opts,
    Configuration const &config) {

    OmpSpreadProcessor processor{opts.nthreads, static_cast<std::size_t>(opts.max_subproblem_size)};
    processor(config, input, sort_indices, sizes, output);
}

/** This is the main entry point for the spreading subproblem as currently
 * configured by default for use in finufft. It makes use of the legacy configuration,
 * although other configurations can be selected by editing the code.
 *
 * TODO: API in planning to select correct configuration at runtime.
 *
 */
template <std::size_t Dim, typename T, typename IdxT>
inline void spread(
    IdxT const *sort_indices, nu_point_collection<Dim, T const> const &input,
    std::array<std::int64_t, Dim> const &sizes, T *output, const finufft_spread_opts &opts) {

    // auto config = get_spread_configuration_legacy<T, Dim>(opts);
    auto config = get_spread_configuration_avx512<T, Dim>(opts);


    spread_with_configuration<Dim, T>(sort_indices, input, sizes, output, opts, config);
}

} // namespace spreading
} // namespace finufft
