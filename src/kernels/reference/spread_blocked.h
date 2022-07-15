#pragma once

#include "../../tracing.h"
#include "../sorting.h"
#include "../spreading.h"

namespace finufft {
namespace spreading {
namespace reference {

/** Timers for blocked spreading.
 *
 */
struct SpreadBlockedTimers {
    SpreadBlockedTimers() = default;
    SpreadBlockedTimers(finufft::Timer &timer)
        : gather(timer.make_timer("gather")), subproblem(timer.make_timer("subproblem")),
          accumulate(timer.make_timer("accumulate")) {}

    finufft::Timer gather;     ///< Timer for copying point data into local buffer.
    finufft::Timer subproblem; ///< Timer for spreading subproblem.
    finufft::Timer accumulate; ///< Timer for accumulating results into output buffer.
};

struct SpreadTimers {
    finufft::Timer sort_packed;
    finufft::Timer spread_blocked;

    SpreadBlockedTimers spread_blocked_timers;

    SpreadTimers() = default;
    SpreadTimers(finufft::Timer &timer)
        : sort_packed(timer.make_timer("sp")), spread_blocked(timer.make_timer("sb")),
          spread_blocked_timers(spread_blocked) {}
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

/** Implements a spread operation using a packed sort and a blocked spread.
 *
 */
template <typename T, std::size_t Dim>
SpreadFunctor<T, Dim> make_packed_sort_spread_blocked(
    SortPointsFunctor<T, Dim> &&sort_points, SpreadSubproblemFunctor<T, Dim> &&spread_subproblem,
    SynchronizedAccumulateFactory<T, Dim> &&accumulate,
    FoldRescaleRange range,
    tcb::span<const std::size_t, Dim> target_size,
    tcb::span<const std::size_t, Dim> grid_size, SpreadTimers const &timers = {});

} // namespace reference
} // namespace spreading
} // namespace finufft
