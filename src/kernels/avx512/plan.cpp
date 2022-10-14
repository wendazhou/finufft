#include "plan.h"

#include <array>

#include <finufft/utils_precindep.h>

#include "../../memory.h"
#include "../extents.h"
#include "../fft.h"
#include "../legacy/interpolation_legacy.h"
#include "../reference/spread_kernel.h"
#include "spread_avx512.h"

namespace finufft {
namespace avx512 {

template <typename T, std::size_t Dim>
Type1Plan<T, Dim> make_type1_plan(Type1TransformConfiguration<Dim> const &configuration) {
    auto kernel_spec = spreading::reference::get_default_kernel_specification(
        configuration.tolerance_, configuration.upsampling_factor_);

    std::array<std::size_t, Dim> buffer_size;

    for (std::size_t i = 0; i < Dim; ++i) {
        buffer_size[i] =
            static_cast<std::size_t>(configuration.modes_[i] * configuration.upsampling_factor_);
        buffer_size[i] = std::max(buffer_size[i], std::size_t(2 * kernel_spec.width));
        buffer_size[i] = utils::next235even(buffer_size[i]);
    }

    auto buffer_stride = strides_from_sizes<Dim>(buffer_size);
    std::size_t num_buffer_elements = std::accumulate(
        buffer_size.begin(), buffer_size.end(), std::size_t(1), std::multiplies<std::size_t>());

    auto buffer = finufft::allocate_aligned_array<T>(2 * num_buffer_elements, 64);

    auto spread_functor = spreading::avx512::get_blocked_spread_functor<T, Dim>(
        kernel_spec, buffer_size, spreading::FoldRescaleRange::Pi);

    auto fft = fft::make_fftw_planned_transform<T, Dim>(
        fft::FourierTransformDirection::Backward, buffer_size, buffer_stride, buffer.get());

    auto interpolation = interpolation::legacy::make_legacy_interpolation_functor<T, Dim>(
        configuration.modes_,
        strides_from_sizes<Dim>(configuration.modes_),
        buffer_size,
        buffer_stride,
        spreading::reference::make_interpolation_kernel_factory<T>(kernel_spec),
        interpolation::ModeOrdering(configuration.mode_ordering_));

    return make_type1_plan(
        std::move(buffer), std::move(spread_functor), std::move(fft), std::move(interpolation));
}

#define INSTANTIATE(T, Dim)                                                                        \
    template Type1Plan<T, Dim> make_type1_plan(                                                    \
        Type1TransformConfiguration<Dim> const &configuration);

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)
INSTANTIATE(float, 3)

#undef INSTANTIATE

} // namespace avx512
} // namespace finufft
