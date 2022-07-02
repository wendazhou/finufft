#pragma once

#include <array>
#include <memory>
#include <mutex>

#include "../../spreading.h"

namespace finufft {
namespace spreading {

/** Simple implementation of globally locked synchronized accumulate.
 *
 */
template <typename T, std::size_t Dim, typename Impl> struct GlobalLockedSynchronizedAccumulate {
    T *output_;
    std::array<std::size_t, Dim> sizes;
    std::unique_ptr<std::mutex> mutex_;

    GlobalLockedSynchronizedAccumulate(T *output, std::array<std::size_t, Dim> const &sizes)
        : output_(output), sizes(sizes), mutex_(std::make_unique<std::mutex>()) {}
    GlobalLockedSynchronizedAccumulate(GlobalLockedSynchronizedAccumulate const &) = delete;
    GlobalLockedSynchronizedAccumulate(GlobalLockedSynchronizedAccumulate &&) = default;

    void operator()(T const *data, grid_specification<Dim> const &grid) const {
        std::scoped_lock lock(*mutex_);
        Impl{}(data, grid, output_, sizes);
    }
};

/** Implementation of a non-locked accumulate.
 * 
 * This functor requires that either:
 * 1) the processor uses the accumulate non-concurrently (e.g. single thread), or
 * 2) the underlying implementation is thread-safe (e.g. using atomics)
 * 
 */
template <typename T, std::size_t Dim, typename Impl> struct NonLockedSynchronizedAccumulate {
    T* output_;
    std::array<std::size_t, Dim> sizes;

    void operator()(T const *data, grid_specification<Dim> const &grid) const {
        Impl{}(data, grid, output_, sizes);
    }
};

template <typename T, std::size_t Dim, typename Fn> struct LambdaSynchronizedAccumulateFactory {
    Fn fn;

    SynchronizedAccumulateFunctor<T, Dim>
    operator()(T *output, std::array<std::size_t, Dim> const &sizes) const {
        return fn(output, sizes);
    }
};

template <typename T, std::size_t Dim, typename Fn>
LambdaSynchronizedAccumulateFactory<T, Dim, Fn>
make_lambda_synchronized_accumulate_factory(Fn &&fn) {
    return {std::forward<Fn>(fn)};
}

} // namespace spreading
} // namespace finufft
