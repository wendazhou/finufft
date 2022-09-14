#pragma once

#include "../../plan.h"
#include "../../tracing.h"
#include "../spreading.h"

#include <finufft_spread_opts.h>

#include <tcb/span.hpp>

namespace finufft {
namespace spreading {

/** Get default configuration which mimics current functionality.
 *
 */
template <typename T, std::size_t Dim>
SpreadFunctorConfiguration<T, Dim> get_spread_configuration_legacy(finufft_spread_opts const &opts);

namespace legacy {

/** Spread functor which calls into the current implementation of bin-sorting
 * and spreading. This is mostly implemented for testing purposes.
 *
 */
template <typename T, std::size_t Dim>
SpreadFunctor<T, Dim> make_spread_functor(
    kernel_specification const &kernel_spec, FoldRescaleRange input_range,
    tcb::span<const std::size_t, Dim> size, finufft::Timer const &timer = {});

/** Construct legacy options from kernel specification.
 *
 * In order to call reference implementation, we construct a legacy options struct
 * from the bare kernel specification. Note that the reverse engineering is not unique,
 * and depends on the specific implementation of setup_spreader.
 *
 */
finufft_spread_opts
construct_opts_from_kernel(const kernel_specification &kernel, std::size_t dim = 0);

/** Create type-1 plan
 *
 */
template <typename T, std::size_t Dim>
Type1Plan<T, Dim> make_legacy_type1_plan(Type1TransformConfiguration<Dim> const &configuration);

} // namespace legacy

} // namespace spreading
} // namespace finufft
