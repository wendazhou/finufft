#include "plan.h"

#include <cmath>
#include <complex>

namespace finufft {
namespace reference {

namespace {

std::int64_t get_mode_frequency(std::size_t mode_idx, std::size_t num_modes, int mode_ordering) {
    if (mode_ordering == 0) {
        return static_cast<std::int64_t>(mode_idx) - num_modes / 2;
    } else {
        return mode_idx >= num_modes / 2 ? static_cast<std::int64_t>(mode_idx) - num_modes
                                        : static_cast<std::int64_t>(mode_idx);
    }
}

template <typename T, std::size_t Dim, std::size_t DimRemaining> struct NuftType1ExactImpl {
    void operator()(
        std::size_t num_points, tcb::span<const T *const, Dim> coordinates,
        std::complex<T> const *weights, tcb::span<const std::size_t, DimRemaining> num_modes,
        std::complex<T> *output, int mode_ordering,
        tcb::span<const T, Dim - DimRemaining> phase) const noexcept {

        auto modes_stride = std::accumulate(
            num_modes.begin(), num_modes.end() - 1, std::size_t(1), std::multiplies<std::size_t>());

        auto num_modes_last = num_modes.back();
        auto num_modes_remaining = num_modes.template subspan<0, DimRemaining - 1>();

        NuftType1ExactImpl<T, Dim, DimRemaining - 1> impl_;
        std::array<T, Dim - DimRemaining + 1> phase_;
        std::copy(phase.begin(), phase.end(), phase_.begin() + 1);

        for (std::size_t ki = 0; ki < num_modes_last; ++ki) {
            phase_[0] = static_cast<T>(get_mode_frequency(ki, num_modes_last, mode_ordering));

            impl_(
                num_points,
                coordinates,
                weights,
                num_modes_remaining,
                output + modes_stride * ki,
                mode_ordering,
                phase_);
        }
    }
};

template <typename T, std::size_t Dim> struct NuftType1ExactImpl<T, Dim, 1> {
    void operator()(
        std::size_t num_points, tcb::span<const T *const, Dim> coordinates,
        std::complex<T> const *weights, tcb::span<const std::size_t, 1> num_modes,
        std::complex<T> *output, int mode_ordering,
        tcb::span<const T, Dim - 1> phase) const noexcept {

        for (std::size_t k = 0; k < num_modes[0]; ++k) {
            for (std::size_t j = 0; j < num_points; ++j) {
                T kx = 0;
                for (std::size_t d = 0; d < Dim - 1; ++d) {
                    kx += coordinates[Dim - d - 1][j] * phase[d];
                }
                kx += coordinates[0][j] *
                      static_cast<T>(get_mode_frequency(k, num_modes[0], mode_ordering));

                auto factor = std::complex<T>(std::cos(kx), std::sin(kx));
                output[k] += weights[j] * factor;
            }
        }
    }
};

template <typename T, std::size_t Dim> struct NuftType1ExactPlan {
    std::array<std::size_t, Dim> num_modes_;
    int mode_ordering_;

    void operator()(
        std::size_t num_points, tcb::span<const T *const, Dim> coordinates, T const *weights,
        T *output) const noexcept {
        NuftType1ExactImpl<T, Dim, Dim> impl_;
        impl_(
            num_points,
            coordinates,
            reinterpret_cast<std::complex<T> const *>(weights),
            num_modes_,
            reinterpret_cast<std::complex<T> *>(output),
            mode_ordering_,
            std::array<T, 0>());
    }
};

} // namespace

template <typename T, std::size_t Dim>
Type1Plan<T, Dim> make_exact_type1_plan(Type1TransformConfiguration<Dim> const &configuration) {
    return Type1Plan<T, Dim>(
        NuftType1ExactPlan<T, Dim>{configuration.modes_, configuration.mode_ordering_});
}

#define INSTANTIATE(T, Dim)                                                                        \
    template Type1Plan<T, Dim> make_exact_type1_plan(                                              \
        Type1TransformConfiguration<Dim> const &configuration);

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)
INSTANTIATE(float, 3)

INSTANTIATE(double, 1)
INSTANTIATE(double, 2)
INSTANTIATE(double, 3)

#undef INSTANTIATE

} // namespace reference
} // namespace finufft
