#pragma once

#include <cstddef>
#include <function2/function2.h>
#include <tcb/span.hpp>

namespace finufft {
namespace fft {

enum class FourierTransformDirection {
    Forward,
    Backward,
};

#define F(NAME, ...)                                                                               \
    template <typename T, std::size_t Dim> struct NAME : fu2::unique_function<__VA_ARGS__> {       \
        using fu2::unique_function<__VA_ARGS__>::unique_function;                                  \
        using fu2::unique_function<__VA_ARGS__>::operator=;                                        \
    };

template <typename T> struct PlannedFourierTransformation : fu2::function<void(T *data)> {
    using fu2::function<void(T *data)>::function;
    using fu2::function<void(T *data)>::operator=;
};

/** This function pointer represents a facility to plan an in-place Fourier transform
 * with the given parameters.
 *
 */
F(FourierTransformFactory,
  PlannedFourierTransformation<T>(
      FourierTransformDirection direction, tcb::span<const std::size_t, Dim> size,
      tcb::span<const std::size_t, Dim> stride, T *data, std::size_t n_batch,
      std::size_t stride_batch) const);

#undef F

template <typename T, std::size_t Dim>
PlannedFourierTransformation<T> make_fftw_planned_transform(
    FourierTransformDirection direction, tcb::span<const std::size_t, Dim> size,
    tcb::span<const std::size_t, Dim> stride, T *data, std::size_t n_batch = 1,
    std::size_t stride_batch = 0);

} // namespace fft
} // namespace finufft
