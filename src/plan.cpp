#include "plan.h"

namespace finufft {

namespace {
template <typename T, std::size_t Dim> struct KernelType1Plan {
    finufft::aligned_unique_array<T> uniform_buffer_;
    finufft::spreading::SpreadFunctor<T, Dim> spread_blocked_;
    finufft::fft::PlannedFourierTransformation<T> fft_;
    finufft::interpolation::InterpolationFunctor<T, Dim> interpolate_;

    void execute(tcb::span<const T *, Dim> coordinates, T const *weights, T *result) {
        finufft::spreading::nu_point_collection<Dim, const T> input;
        std::copy(coordinates.begin(), coordinates.end(), input.coordinates.begin());
        input.strengths = weights;

        spread_blocked_(input, uniform_buffer_.get());
        fft_(uniform_buffer_.get());
        interpolate_(uniform_buffer_.get(), result);
    }
};
} // namespace

template <typename T, std::size_t Dim>
Type1Plan<T, Dim> make_type1_plan(
    finufft::aligned_unique_array<T> uniform_buffer,
    finufft::spreading::SpreadFunctor<T, Dim> spread_blocked,
    finufft::fft::PlannedFourierTransformation<T> fft,
    finufft::interpolation::InterpolationFunctor<T, Dim> interpolate) {
    return KernelType1Plan<T, Dim>(
        std::move(uniform_buffer),
        std::move(spread_blocked),
        std::move(fft),
        std::move(interpolate));
}

#define INSTANTIATE(T, Dim)                                                                        \
    template Type1Plan<T, Dim> make_type1_plan(                                                    \
        finufft::aligned_unique_array<T> uniform_buffer,                                           \
        finufft::spreading::SpreadFunctor<T, Dim>                                                  \
            spread_blocked,                                                                        \
        finufft::fft::PlannedFourierTransformation<T>                                              \
            fft,                                                                                   \
        finufft::interpolation::InterpolationFunctor<T, Dim>                                       \
            interpolate);

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)
INSTANTIATE(float, 3)

INSTANTIATE(double, 1)
INSTANTIATE(double, 2)
INSTANTIATE(double, 3)

#undef INSTANTIATE

} // namespace finufft
