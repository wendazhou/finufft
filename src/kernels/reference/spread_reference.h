#pragma once

#include "../spreading.h"
#include <finufft_spread_opts.h>

namespace finufft {
namespace spreading {

/** Get default configuration based on reference implementation.
 *
 */
template <typename T, std::size_t Dim>
SpreadFunctorConfiguration<T, Dim>
get_spread_configuration_reference(finufft_spread_opts const &opts);


} // namespace spreading
} // namespace finufft
