#pragma once

#include "../spreading.h"
#include "../sorting.h"

#include <finufft_spread_opts.h>

namespace finufft {
namespace spreading {

/** Get default configuration based on reference implementation.
 *
 */
template <typename T, std::size_t Dim>
SpreadFunctorConfiguration<T, Dim>
get_spread_configuration_reference(finufft_spread_opts const &opts);

/** Create a spread functor which assembles the given sub-operations
 * to perform the spread operation through an indirect sort strategy.
 * 
 */
template <typename T, std::size_t Dim>
SpreadFunctor<T, Dim> make_indirect_omp_spread_functor(
    BinSortFunctor<T, Dim>&& bin_sort_functor,
    GatherRescaleFunctor<T, Dim>&& gather,
    SpreadSubproblemFunctor<T, Dim>&& spread_subproblem,
    SynchronizedAccumulateFactory<T, Dim>&& accumulate_factory,
    tcb::span<const std::size_t, Dim> target_size,
    FoldRescaleRange input_range,
    finufft::Timer const& timer = {});


namespace reference {

/** Get a default instantiation of a spread functor for the reference implementation,
 * based on an indirect sort strategy.
 * 
 */
template <typename T, std::size_t Dim>
SpreadFunctor<T, Dim> get_indirect_spread_functor(
    kernel_specification const &kernel_spec,
    tcb::span<const std::size_t, Dim> target_size,
    FoldRescaleRange input_range,
    finufft::Timer const& timer = {});

}


} // namespace spreading
} // namespace finufft
