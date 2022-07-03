#pragma once

#include <array>
#include <memory>
#include <mutex>

#include "../../spreading.h"

#include <tcb/span.hpp>

namespace finufft {
namespace spreading {

namespace detail {

/** Simple contiguous accumulate implementation (no locking).
 *
 * @tparam T The type of the data to accumulate.
 * @tparam PackSize The number of elements of type T constituting a logical element.
 *      lengths and offsets are multiplied by the packsize to address the correct memory location.
 *      This enable this accumulator to directly support e.g. interleaved complex elements.
 *
 */
template <typename T, std::size_t PackSize = 1> struct SimpleContiguousAccumulate {
    T const *input;
    T *__restrict output;

    void operator()(std::size_t offset_input, std::size_t offset_output, std::size_t length) const {
        offset_input *= PackSize;
        offset_output *= PackSize;
        length *= PackSize;

        for (std::size_t i = 0; i < length; ++i) {
            output[offset_output + i] += input[offset_input + i];
        }
    }
};

template <typename T, std::size_t Dim, typename ContiguousAccumulate>
struct WrappedSubgridAccumulator;

/** Implementation of accumulation for a contiguous range in given dimension.
 *
 * This class is a helper to implement the contiguous aspect in a given dimension.
 * When working on a general dimension (not the last one), this implementation
 * simply loops over each of the valid slices of the current dimension, and dispatches
 * to the implementation at dimension D - 1 for the slice.
 * However, when working on the last dimension, this implementation instead directly
 * dispatches to the underlying contiguous accumulate implementation.
 *
 * Note: this can be significantly simplified with a C++17 constexpr if expression.
 *
 */
template <typename T, std::size_t Dim, typename ContiguousAccumulate> struct AccumulateNextDim {
    ContiguousAccumulate const &accumulate;
    tcb::span<const int64_t, Dim> offsets;
    tcb::span<const std::size_t, Dim> extents;
    tcb::span<const std::size_t, Dim> input_strides;
    tcb::span<const std::size_t, Dim> sizes;
    tcb::span<const std::size_t, Dim> output_strides;

    void operator()(int64_t length, std::size_t input_offset, std::size_t output_offset) {
        if (length <= 0) {
            return;
        }
        for (std::size_t i = 0; i < length; ++i) {
            WrappedSubgridAccumulator<T, Dim - 1, ContiguousAccumulate>{accumulate}(
                input_offset + i * input_strides[Dim - 1],
                offsets.template subspan<0, Dim - 1>(),
                extents.template subspan<0, Dim - 1>(),
                input_strides.template subspan<0, Dim - 1>(),
                output_offset + i * output_strides[Dim - 1],
                sizes.template subspan<0, Dim - 1>(),
                output_strides.template subspan<0, Dim - 1>());
        }
    }
};

template <typename T, typename ContiguousAccumulate>
struct AccumulateNextDim<T, 1, ContiguousAccumulate> {
    ContiguousAccumulate const &accumulate;
    tcb::span<const int64_t, 1> offsets;
    tcb::span<const std::size_t, 1> extents;
    tcb::span<const std::size_t, 1> input_strides;
    tcb::span<const std::size_t, 1> sizes;
    tcb::span<const std::size_t, 1> output_sizes;

    void operator()(int64_t length, std::size_t input_offset, std::size_t output_offset) {
        if (length <= 0) {
            return;
        }

        accumulate(input_offset, output_offset, length);
    }
};

/** Recursive implementation of wrapped subgrid accumulation.
 *
 * This template implements a functor which accumulates a subgrid
 * into the main grid, with wrapping. The implementation is performs
 * the wrapping in three parts, by splitting the subgrid into the elements
 * which are to the left, within, and to the right of the main grid.
 *
 */
template <typename T, std::size_t Dim, typename ContiguousAccumulate>
struct WrappedSubgridAccumulator {
    ContiguousAccumulate const &contiguous_accumulate;

    void operator()(
        std::size_t current_input_offset, tcb::span<const int64_t, Dim> offsets,
        tcb::span<const std::size_t, Dim> extents, tcb::span<const std::size_t, Dim> input_strides,
        std::size_t current_output_offset, tcb::span<const std::size_t, Dim> sizes,
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

        AccumulateNextDim<T, Dim, ContiguousAccumulate> accumulate_contiguous{
            contiguous_accumulate, offsets, extents, input_strides, sizes, strides};

        // Loop through left wrapping
        // This loop takes the initial part of the subgrid which does not yet overlap
        // with the main grid, and adds it to the end of the main grid.
        accumulate_contiguous(
            std::min<int64_t>(subgrid_end, 0) - offset,
            current_input_offset,
            current_output_offset + (size + offset) * strides[Dim - 1]);

        // Loop through right wrapping
        // This loop takes the final part of the subgrid which overhangs the main grid,
        // and adds it to the start of the main grid.
        accumulate_contiguous(
            subgrid_end - size,
            current_input_offset + (size - offset) * input_strides[Dim - 1],
            current_output_offset);

        // Loop through main part
        // This loop takes the intersection of the subgrid and the main grid.
        accumulate_contiguous(
            std::min(size, subgrid_end) - std::max<int64_t>(offset, 0),
            current_input_offset - std::min<int64_t>(offset, 0) * input_strides[Dim - 1],
            current_output_offset + std::max<int64_t>(offset, 0) * strides[Dim - 1]);
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
            strides_output[0] = 1;
            for (std::size_t i = 1; i < Dim; ++i) {
                strides_output[i] = strides_output[i - 1] * sizes[i - 1];
            }
        }

        std::array<std::size_t, Dim> strides_input;
        {
            strides_input[0] = 1;
            for (std::size_t i = 1; i < Dim; ++i) {
                strides_input[i] = strides_input[i - 1] * grid.extents[i - 1];
            }
        }

        // Note: accumulating complex interleaved data, so using packsize = 2
        detail::SimpleContiguousAccumulate<T, 2> contiguous_accumulate{input, output};
        detail::WrappedSubgridAccumulator<T, Dim, detail::SimpleContiguousAccumulate<T, 2>>{
            contiguous_accumulate}(
            0, grid.offsets, grid.extents, strides_input, 0, sizes, strides_output);
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
