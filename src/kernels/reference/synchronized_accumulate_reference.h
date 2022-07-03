#pragma once

#include <array>
#include <memory>
#include <mutex>

#include "../../spreading.h"

#include <tcb/span.hpp>

namespace finufft {
namespace spreading {

namespace detail {

/** Recursive implementation of wrapped subgrid accumulation.
 *
 * This template implements a functor which accumulates a subgrid
 * into the main grid, with wrapping. The implementation is performs
 * the wrapping in three parts, by splitting the subgrid into the elements
 * which are to the left, within, and to the right of the main grid.
 *
 */
template <typename T, std::size_t Dim> struct WrappedSubgridAccumulator {
    void operator()(
        T const *input, tcb::span<const int64_t, Dim> offsets,
        tcb::span<const std::size_t, Dim> extents, tcb::span<const std::size_t, Dim> input_strides,
        T *__restrict output, tcb::span<const std::size_t, Dim> sizes,
        tcb::span<const std::size_t, Dim> strides) const noexcept {

        // Do all arithmetic in signed integers to avoid surprises.
        auto offset = offsets[Dim - 1];
        auto extent = static_cast<int64_t>(extents[Dim - 1]);
        auto size = static_cast<int64_t>(sizes[Dim - 1]);

        // Adjust offset if necessary
        offset = offset % size;
        if ((extent > size) && (offset >= size / 2)) {
            // Need to start on the left because we are adding
            // a large subgrid which requires wrapping on both ends
            offset -= size;
        }

        auto subgrid_end = offset + extent;

        // Loop through left wrapping
        for (int64_t i = offset; i < std::min<int64_t>(subgrid_end, 0); ++i) {
            auto index = i - offset;

            WrappedSubgridAccumulator<T, Dim - 1>{}(
                input + index * input_strides[Dim - 1],
                offsets.template subspan<0, Dim - 1>(),
                extents.template subspan<0, Dim - 1>(),
                input_strides.template subspan<0, Dim - 1>(),
                output + (i + size) * strides[Dim - 1],
                sizes.template subspan<0, Dim - 1>(),
                strides.template subspan<0, Dim - 1>());
        }

        // Loop through right wrapping
        for (int64_t i = size; i < subgrid_end; ++i) {
            auto index = i - offset;

            WrappedSubgridAccumulator<T, Dim - 1>{}(
                input + index * input_strides[Dim - 1],
                offsets.template subspan<0, Dim - 1>(),
                extents.template subspan<0, Dim - 1>(),
                input_strides.template subspan<0, Dim - 1>(),
                output + (i - size) * strides[Dim - 1],
                sizes.template subspan<0, Dim - 1>(),
                strides.template subspan<0, Dim - 1>());
        }

        // Loop through main part
        for (int64_t i = std::max<int64_t>(offset, 0); i < std::min(size, subgrid_end); ++i) {
            auto index = i - offset;

            WrappedSubgridAccumulator<T, Dim - 1>{}(
                input + index * input_strides[Dim - 1],
                offsets.template subspan<0, Dim - 1>(),
                extents.template subspan<0, Dim - 1>(),
                input_strides.template subspan<0, Dim - 1>(),
                output + i * strides[Dim - 1],
                sizes.template subspan<0, Dim - 1>(),
                strides.template subspan<0, Dim - 1>());
        }
    }
};

template <typename T> struct WrappedSubgridAccumulator<T, 0> {
    void operator()(
        T const *input, tcb::span<const int64_t, 0> offsets,
        tcb::span<const std::size_t, 0> extents, tcb::span<const std::size_t, 0> input_strides,
        T *__restrict output, tcb::span<const std::size_t, 0> sizes,
        tcb::span<const std::size_t, 0> strides) const noexcept {
        // We work with pair of values (interleaved complex).
        output[0] += input[0];
        output[1] += input[1];
    }
};
} // namespace detail

template <typename T, std::size_t Dim> struct NonSynchronizedAccumulateWrappedSubgridReference {
    /** Non thread-safe generic reference implementation for wrapped add subgrid.
     *
     */
    void operator()(
        T const *input, grid_specification<Dim> const &grid, T *output,
        tcb::span<const std::size_t, Dim> sizes) const noexcept {

        std::array<std::size_t, Dim> strides_output;
        {
            strides_output[0] = 2;
            for (std::size_t i = 1; i < Dim; ++i) {
                strides_output[i] = strides_output[i - 1] * sizes[i - 1];
            }
        }

        std::array<std::size_t, Dim> strides_input;
        {
            strides_input[0] = 2;
            for (std::size_t i = 1; i < Dim; ++i) {
                strides_input[i] = strides_input[i - 1] * grid.extents[i - 1];
            }
        }

        detail::WrappedSubgridAccumulator<T, Dim>{}(
            input, grid.offsets, grid.extents, strides_input, output, sizes, strides_output);
    }
};

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
    T *output_;
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

template <typename T, std::size_t Dim>
SynchronizedAccumulateFactory<T, Dim> get_reference_locking_accumulator() {
    return make_lambda_synchronized_accumulate_factory<T, Dim>(
        [](T *output, std::array<std::size_t, Dim> const &sizes) {
            return SynchronizedAccumulateFunctor<T, Dim>(
                GlobalLockedSynchronizedAccumulate<
                    T,
                    Dim,
                    NonSynchronizedAccumulateWrappedSubgridReference<T, Dim>>(output, sizes));
        });
}

} // namespace spreading
} // namespace finufft
