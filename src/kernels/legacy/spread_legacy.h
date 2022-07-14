#pragma once

#include "../spreading.h"
#include <finufft_spread_opts.h>


namespace finufft {
namespace spreading {

/** Get default configuration which mimics current functionality.
 *
 */
template <typename T, std::size_t Dim>
SpreadFunctorConfiguration<T, Dim>
get_spread_configuration_legacy(finufft_spread_opts const &opts);

extern template SpreadFunctorConfiguration<float, 1> get_spread_configuration_legacy<float, 1>(finufft_spread_opts const &);
extern template SpreadFunctorConfiguration<float, 2> get_spread_configuration_legacy<float, 2>(finufft_spread_opts const &);
extern template SpreadFunctorConfiguration<float, 3> get_spread_configuration_legacy<float, 3>(finufft_spread_opts const &);

extern template SpreadFunctorConfiguration<double, 1> get_spread_configuration_legacy<double, 1>(finufft_spread_opts const &);
extern template SpreadFunctorConfiguration<double, 2> get_spread_configuration_legacy<double, 2>(finufft_spread_opts const &);
extern template SpreadFunctorConfiguration<double, 3> get_spread_configuration_legacy<double, 3>(finufft_spread_opts const &);


} // namespace spreading
} // namespace finufft
