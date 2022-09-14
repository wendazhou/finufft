#pragma once

#include <cstddef>
#include <function2/function2.h>

namespace finufft {
namespace interpolation {

enum class ModeOrdering {
    ZeroMiddle = 0,
    ZeroFirst = 1,
};

/** Generic function pointer to an interpolation function.
 *
 */
template <typename T, std::size_t Dim>
struct InterpolationFunctor : fu2::function<void(T const *, T *) const noexcept> {
    using fu2::function<void(T const *, T *) const noexcept>::function;
    using fu2::function<void(T const *, T *) const noexcept>::operator=;
};

/** This functor represents an utility which fills a given array with the given
 * kernel values for the given dimension.
 *
 */
template <typename T>
struct InterpolationKernelFactory : fu2::function<void(T *, std::size_t) const noexcept> {
    using fu2::function<void(T *, std::size_t) const noexcept>::function;
    using fu2::function<void(T *, std::size_t) const noexcept>::operator=;
};

} // namespace interpolation
} // namespace finufft
