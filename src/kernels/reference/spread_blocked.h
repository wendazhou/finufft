#pragma once

#include "../../tracing.h"
#include "../spreading.h"

namespace finufft {
namespace spreading {
namespace reference {

struct SpreadBlockedTimers {
    SpreadBlockedTimers(finufft::Timer &timer)
        : gather(timer.make_timer("gather")), subproblem(timer.make_timer("subproblem")),
          accumulate(timer.make_timer("accumulate")) {}

    finufft::Timer gather;
    finufft::Timer subproblem;
    finufft::Timer accumulate;
};

/** Reference implementation of a blocked spread operation.
 *
 * This function builds a blocked spread implementation from the given
 * subproblem and accumulate implementations.
 *
 */
template <typename T, std::size_t Dim>
SpreadBlockedFunctor<T, Dim> make_omp_spread_blocked(
    SpreadSubproblemFunctor<T, Dim> &&spread_subproblem,
    SynchronizedAccumulateFactory<T, Dim> &&accumulate_factory,
    SpreadBlockedTimers const &timers_ref = {});

} // namespace reference
} // namespace spreading
} // namespace finufft
