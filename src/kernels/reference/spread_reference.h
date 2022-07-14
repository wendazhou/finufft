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
