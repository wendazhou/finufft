#pragma once

#include <function2/function2.h>

namespace finufft {
namespace interpolation {

/** Generic function pointer to an interpolation function.
 *
 */
template <typename T, std::size_t Dim>
struct InterpolationFunctor : fu2::function<void(T const *, T *) const noexcept> {
    using fu2::function<void(T const *, T *) const noexcept>::function;
    using fu2::function<void(T const *, T *) const noexcept>::operator=;
};

} // namespace interpolation
} // namespace finufft
