#include "interpolation_legacy.h"

#include <numeric>

#include "../../memory.h"

#include <fftw3.h>

namespace finufft {

namespace common {

void deconvolveshuffle1d(
    int dir, float prefac, float *ker, int64_t ms, float *fk, int64_t nf1, fftwf_complex *fw,
    int modeord);
void deconvolveshuffle1d(
    int dir, double prefac, double *ker, int64_t ms, double *fk, int64_t nf1, fftw_complex *fw,
    int modeord);
void deconvolveshuffle2d(
    int dir, float prefac, float *ker1, float *ker2, int64_t ms, int64_t mt, float *fk, int64_t nf1,
    int64_t nf2, fftwf_complex *fw, int modeord);
void deconvolveshuffle2d(
    int dir, double prefac, double *ker1, double *ker2, int64_t ms, int64_t mt, double *fk,
    int64_t nf1, int64_t nf2, fftw_complex *fw, int modeord);
void deconvolveshuffle3d(
    int dir, float prefac, float *ker1, float *ker2, float *ker3, int64_t ms, int64_t mt,
    int64_t mu, float *fk, int64_t nf1, int64_t nf2, int64_t nf3, fftwf_complex *fw, int modeord);
void deconvolveshuffle3d(
    int dir, double prefac, double *ker1, double *ker2, double *ker3, int64_t ms, int64_t mt,
    int64_t mu, double *fk, int64_t nf1, int64_t nf2, int64_t nf3, fftw_complex *fw, int modeord);

} // namespace common

namespace interpolation {
namespace legacy {

namespace {

template <typename T> struct fftw_complex_generic;
template <> struct fftw_complex_generic<float> { using type = fftwf_complex; };
template <> struct fftw_complex_generic<double> { using type = fftw_complex; };

template <typename T> using fftw_complex_t = typename fftw_complex_generic<T>::type;

template <typename T, std::size_t Dim> struct LegacyInterpolationFunctor {
    std::array<std::size_t, Dim> input_size_;
    std::array<std::size_t, Dim> output_size_;
    aligned_unique_array<T> kernel_;
    ModeOrdering mode_ordering_;

    LegacyInterpolationFunctor(
        tcb::span<const std::size_t, Dim> output_size, tcb::span<const std::size_t, Dim> input_size,
        tcb::span<const T *const, Dim> kernel, ModeOrdering mode_ordering)
        : kernel_(finufft::allocate_aligned_array<T>(
              std::accumulate(
                  output_size.begin(), output_size.end(), std::size_t(0),
                  [](std::size_t acc, std::size_t s) {
                      return acc + round_to_next_multiple((s / 2) + 1, 64 / sizeof(T));
                  }),
              64)),
          mode_ordering_(mode_ordering) {

        {
            std::size_t offset = 0;
            for (std::size_t i = 0; i < kernel.size(); ++i) {
                std::size_t size = round_to_next_multiple((output_size[i] / 2) + 1, 64 / sizeof(T));
                std::copy(kernel[i], kernel[i] + size, kernel_.get() + offset);
                offset += size;
            }
        }

        std::copy(input_size.begin(), input_size.end(), input_size_.begin());
        std::copy(output_size.begin(), output_size.end(), output_size_.begin());
    }

    LegacyInterpolationFunctor(LegacyInterpolationFunctor const &other)
        : input_size_(other.input_size_), output_size_(other.output_size_),
          kernel_(finufft::allocate_aligned_array<T>(
              std::accumulate(
                  output_size_.begin(), output_size_.end(), std::size_t(0),
                  [](std::size_t acc, std::size_t s) {
                      return acc + round_to_next_multiple((s / 2) + 1, 64 / sizeof(T));
                  }),
              64)),
          mode_ordering_(other.mode_ordering_) {
        std::copy(other.kernel_.get(), other.kernel_.get() + kernel_offset(Dim), kernel_.get());
    }
    LegacyInterpolationFunctor(LegacyInterpolationFunctor &&other) noexcept = default;

    std::size_t kernel_offset(std::size_t i) const noexcept {
        return std::accumulate(
            output_size_.begin(),
            output_size_.begin() + i,
            std::size_t(0),
            [](std::size_t acc, std::size_t s) {
                return acc + round_to_next_multiple((s / 2) + 1, 64 / sizeof(T));
            });
    }

    void operator()(T const *input, T *output) const noexcept {
        static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Only 1D, 2D, and 3D are supported");

        if (Dim == 1) {
            common::deconvolveshuffle1d(
                1,
                1.0,
                kernel_.get() + kernel_offset(0),
                input_size_[0],
                const_cast<T *>(input),
                output_size_[0],
                reinterpret_cast<fftw_complex_t<T> *>(output),
                static_cast<int>(mode_ordering_));
        } else if (Dim == 2) {
            common::deconvolveshuffle2d(
                1,
                1.0,
                kernel_.get() + kernel_offset(0),
                kernel_.get() + kernel_offset(1),
                input_size_[0],
                input_size_[1],
                const_cast<T *>(input),
                output_size_[0],
                output_size_[1],
                reinterpret_cast<fftw_complex_t<T> *>(output),
                static_cast<int>(mode_ordering_));
        } else if (Dim == 3) {
            common::deconvolveshuffle3d(
                1,
                1.0,
                kernel_.get() + kernel_offset(0),
                kernel_.get() + kernel_offset(1),
                kernel_.get() + kernel_offset(2),
                input_size_[0],
                input_size_[1],
                input_size_[2],
                const_cast<T *>(input),
                output_size_[0],
                output_size_[1],
                output_size_[2],
                reinterpret_cast<fftw_complex_t<T> *>(output),
                static_cast<int>(mode_ordering_));
        }
    }
};

} // namespace

template <typename T, std::size_t Dim>
InterpolationFunctor<T, Dim> make_legacy_interpolation_functor(
    T *output, tcb::span<const std::size_t, Dim> output_size,
    tcb::span<const std::size_t, Dim> output_stride, T const *input,
    tcb::span<const std::size_t, Dim> input_size, tcb::span<const std::size_t, Dim> input_stride,
    tcb::span<const T *const, Dim> kernel, ModeOrdering mode_ordering) {
    return LegacyInterpolationFunctor<T, Dim>(output_size, input_size, kernel, mode_ordering);
}

#define INSTANTIATE(T, Dim)                                                                        \
    template InterpolationFunctor<T, Dim> make_legacy_interpolation_functor(                       \
        T *output,                                                                                 \
        tcb::span<const std::size_t, Dim>                                                          \
            output_size,                                                                           \
        tcb::span<const std::size_t, Dim>                                                          \
            output_stride,                                                                         \
        T const *input,                                                                            \
        tcb::span<const std::size_t, Dim>                                                          \
            input_size,                                                                            \
        tcb::span<const std::size_t, Dim>                                                          \
            input_stride,                                                                          \
        tcb::span<const T *const, Dim>                                                             \
            kernel,                                                                                \
        ModeOrdering mode_ordering);

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)
INSTANTIATE(float, 3)

INSTANTIATE(double, 1)
INSTANTIATE(double, 2)
INSTANTIATE(double, 3)

#undef INSTANTIATE

} // namespace legacy
} // namespace interpolation
} // namespace finufft
