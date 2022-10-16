#include "plan.h"

namespace finufft {

namespace {
template <typename T, std::size_t Dim> struct KernelType1Plan {
    finufft::aligned_unique_array<T> uniform_buffer_;
    finufft::spreading::SpreadFunctor<T, Dim> spread_blocked_;
    finufft::fft::PlannedFourierTransformation<T> fft_;
    finufft::interpolation::InterpolationFunctor<T, Dim> interpolate_;

    void operator()(
        std::size_t num_points, tcb::span<const T *const, Dim> coordinates, T const *weights,
        T *result) {
        finufft::spreading::nu_point_collection<Dim, const T> input;
        std::copy(coordinates.begin(), coordinates.end(), input.coordinates.begin());
        input.strengths = weights;
        input.num_points = num_points;

        spread_blocked_(input, uniform_buffer_.get());
        fft_(uniform_buffer_.get());
        interpolate_(uniform_buffer_.get(), result);
    }
};

template <typename T, std::size_t Dim> struct LoopBatchedType1Plan {
    Type1Plan<T, Dim> plan_;
    std::size_t weight_stride_;
    std::size_t result_stride_;

    void operator()(
        std::size_t num_points, tcb::span<const T *const, Dim> coordinates,
        std::size_t num_transforms, T const *weights, T *result) {
        for (std::size_t i = 0; i < num_transforms; ++i) {
            // Note: strides adjusted by 2 to account for complex elements
            plan_(
                num_points,
                coordinates,
                weights + i * 2 * weight_stride_,
                result + i * 2 * result_stride_);
        }
    }
};

} // namespace

template <typename T, std::size_t Dim>
Type1Plan<T, Dim> make_type1_plan_from_parts(
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

template <typename T, std::size_t Dim>
BatchedType1Plan<T, Dim>
batch_type1_plan(Type1Plan<T, Dim> &&plan, std::size_t weight_stride, std::size_t output_stride) {
    return LoopBatchedType1Plan<T, Dim>(std::move(plan), weight_stride, output_stride);
}

#define INSTANTIATE(T, Dim)                                                                        \
    template Type1Plan<T, Dim> make_type1_plan_from_parts(                                         \
        finufft::aligned_unique_array<T> uniform_buffer,                                           \
        finufft::spreading::SpreadFunctor<T, Dim>                                                  \
            spread_blocked,                                                                        \
        finufft::fft::PlannedFourierTransformation<T>                                              \
            fft,                                                                                   \
        finufft::interpolation::InterpolationFunctor<T, Dim>                                       \
            interpolate);                                                                          \
    template BatchedType1Plan<T, Dim> batch_type1_plan(                                            \
        Type1Plan<T, Dim> &&plan, std::size_t weight_stride, std::size_t output_stride);

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)
INSTANTIATE(float, 3)

INSTANTIATE(double, 1)
INSTANTIATE(double, 2)
INSTANTIATE(double, 3)

#undef INSTANTIATE

} // namespace finufft
